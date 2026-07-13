import Foundation
import MLX
import MLXLMCommon
import SpikeCore

private final class CompileTraceCounter {
    var count = 0
}

private enum BatchShapeProbeError: Error, CustomStringConvertible {
    case unsupportedArchitecture(String)
    case tokenMismatch(mode: String, batch: Int, phase: String, expected: [Int], actual: [Int])
    case unexpectedTraceCount(mode: String, batch: Int, expected: Int, actual: Int)

    var description: String {
        switch self {
        case .unsupportedArchitecture(let modelType):
            return "batch probe supports only dense qwen3 cache layout; model_type=\(modelType)"
        case .tokenMismatch(let mode, let batch, let phase, let expected, let actual):
            return "batch probe token mismatch mode=\(mode) B=\(batch) phase=\(phase) expected=\(expected) actual=\(actual)"
        case .unexpectedTraceCount(let mode, let batch, let expected, let actual):
            return "batch probe trace count mode=\(mode) B=\(batch) expected=\(expected) actual=\(actual)"
        }
    }
}

private struct BatchShapeProbeResult {
    let mode: String
    let batchSize: Int
    let promptLengths: [Int]
    let compiledSteps: Int
    let mainTraceCount: Int
    let membershipTraceCount: Int
    let firstCompiledMilliseconds: Double
    let steadyP50Milliseconds: Double
    let maxInitialLogitDelta: Float
}

/// Bench-only proof for the pinned Swift model API. It exercises a real dense model with
/// independent scalar caches and the proposed left-padded batch cache side-by-side.
func batchShapeProbe(
    modelPath: String, compiledSteps: Int, compileMode: String, batchSizes: [Int]
) async {
    do {
        guard compiledSteps > 0 else {
            print("batch-probe FAILED: --steps must be positive")
            exit(1)
        }
        let modes: [Bool]
        switch compileMode {
        case "fixed": modes = [false]
        case "shapeless": modes = [true]
        case "both": modes = [false, true]
        default:
            print("batch-probe FAILED: --compile-mode must be fixed, shapeless, or both")
            exit(1)
        }
        guard !batchSizes.isEmpty,
            batchSizes.allSatisfy({ [1, 2, 4, 7, 8].contains($0) })
        else {
            print("batch-probe FAILED: --batch-sizes must use a comma-separated subset of 1,2,4,7,8")
            exit(1)
        }

        try requireSupportedDenseModel(modelPath: modelPath)

        // Keep the allocator cache bounded even when the bench host's wired limit is raised.
        Memory.cacheLimit = 8 << 30
        let (model, tokenizer, _) = try await loadModelAndTokenizer(modelPath: modelPath)
        let promptText = [
            "Batch probe alpha.",
            "Batch probe beta has a different prompt length.",
            "Batch probe gamma asks for one compact answer about Metal kernels.",
            "Batch probe delta: explain unified memory in two concise clauses.",
            "Batch probe epsilon includes another deliberately distinct token sequence for row identity.",
            "Batch probe zeta asks why bounded caches prevent compile-shape churn.",
            "Batch probe eta requests a short note about rotary position embeddings.",
            "Batch probe theta is the longest row and checks left padding across eight concurrent dense requests.",
        ]
        let prompts = promptText.map { tokenizer.encode(text: $0) }

        var results: [BatchShapeProbeResult] = []
        for shapeless in modes {
            for batchSize in batchSizes {
                // Ragged B=7 is the exact shape produced by removing the middle of B=8.
                let rows = batchSize == 7
                    ? [0, 1, 2, 3, 5, 6, 7]
                    : Array(0 ..< batchSize)
                results.append(
                    try runBatchShapeProbe(
                        model: model,
                        prompts: rows.map { prompts[$0] },
                        compiledSteps: compiledSteps,
                        shapeless: shapeless))
                Memory.clearCache()
            }
        }

        print(
            "compile_mode,batch_size,prompt_lengths,compiled_steps,main_trace_count,membership_trace_count,first_compiled_ms,steady_p50_ms,max_initial_logit_delta,status"
        )
        for result in results {
            let lengths = result.promptLengths.map(String.init).joined(separator: ":")
            print(
                "\(result.mode),\(result.batchSize),\(lengths),\(result.compiledSteps),\(result.mainTraceCount),\(result.membershipTraceCount),\(String(format: "%.3f", result.firstCompiledMilliseconds)),\(String(format: "%.3f", result.steadyP50Milliseconds)),\(String(format: "%.6f", result.maxInitialLogitDelta)),PASS"
            )
        }
    } catch {
        print("batch-probe FAILED: \(error)")
        exit(1)
    }
}

private func runBatchShapeProbe(
    model: any LanguageModel,
    prompts: [[Int]],
    compiledSteps: Int,
    shapeless: Bool
) throws -> BatchShapeProbeResult {
    let batchSize = prompts.count
    let mode = shapeless ? "shapeless" : "fixed"
    let layerCount = model.newCache(parameters: nil).count
    let capacity = (prompts.map(\.count).max() ?? 0) + compiledSteps + 8

    func makeScalarCaches() -> [CompiledKVCache] {
        (0 ..< layerCount).map { _ in CompiledKVCache(capacity: capacity) }
    }

    let referenceCaches = prompts.map { _ in makeScalarCaches() }
    let mergeSourceCaches = prompts.map { _ in makeScalarCaches() }
    var current: [Int] = []
    current.reserveCapacity(batchSize)

    // Prefill both independent reference caches and the scalar histories that will be
    // merged. Their first greedy picks must already agree before batching is involved.
    for row in prompts.indices {
        let referenceFirst = forwardToken(
            model: model, input: prompts[row], caches: referenceCaches[row])
        let mergeFirst = forwardToken(
            model: model, input: prompts[row], caches: mergeSourceCaches[row])
        guard referenceFirst == mergeFirst else {
            throw BatchShapeProbeError.tokenMismatch(
                mode: mode, batch: batchSize, phase: "prefill-row-\(row)",
                expected: [referenceFirst], actual: [mergeFirst])
        }
        current.append(referenceFirst)
    }

    let batchCaches = try (0 ..< layerCount).map { layer in
        try BatchedCompiledKVCache.merging(
            mergeSourceCaches.map { $0[layer] },
            lengths: prompts.map(\.count))
    }
    try batchCaches.forEach { try $0.requireCapacity(for: compiledSteps + 2) }
    let batchModelCaches: [any KVCache] = batchCaches.map { $0 }

    // One uncompiled shared forward checks full-logit numerical proximity as well as
    // exact greedy-token parity. Subsequent forwards exercise compiled replay.
    var scalarLogits: [[Float]] = []
    var scalarNext: [Int] = []
    scalarLogits.reserveCapacity(batchSize)
    scalarNext.reserveCapacity(batchSize)
    for row in prompts.indices {
        let logits = forwardLogits(
            model: model, input: [current[row]], caches: referenceCaches[row])
        scalarLogits.append(logits)
        scalarNext.append(argmaxIndex(logits))
    }
    let batchInput = MLXArray(current.map(Int32.init)).reshaped([batchSize, 1])
    let batchLast = model(batchInput, cache: batchModelCaches)[0..., -1, 0...]
        .asType(.float32)
    let batchShape = batchLast.shape
    let flattenedBatchLogits = batchLast.asArray(Float.self)
    let vocabularySize = batchShape[1]
    var batchNext: [Int] = []
    var maxLogitDelta: Float = 0
    for row in prompts.indices {
        let start = row * vocabularySize
        let rowLogits = Array(flattenedBatchLogits[start ..< start + vocabularySize])
        batchNext.append(argmaxIndex(rowLogits))
        for (lhs, rhs) in zip(scalarLogits[row], rowLogits) {
            maxLogitDelta = max(maxLogitDelta, abs(lhs - rhs))
        }
    }
    guard scalarNext == batchNext else {
        throw BatchShapeProbeError.tokenMismatch(
            mode: mode, batch: batchSize, phase: "uncompiled-logits",
            expected: scalarNext, actual: batchNext)
    }
    current = scalarNext

    let mainCounter = CompileTraceCounter()
    let mainStep = makeCompiledBatchStep(
        model: model,
        caches: batchCaches,
        batchSize: batchSize,
        shapeless: shapeless,
        counter: mainCounter)
    var stepMilliseconds: [Double] = []
    stepMilliseconds.reserveCapacity(compiledSteps)

    for step in 0 ..< compiledSteps {
        var expected: [Int] = []
        var expectedLogits: [[Float]] = []
        expected.reserveCapacity(batchSize)
        expectedLogits.reserveCapacity(batchSize)
        for row in prompts.indices {
            let logits = forwardLogits(
                model: model, input: [current[row]], caches: referenceCaches[row])
            expectedLogits.append(logits)
            expected.append(argmaxIndex(logits))
        }

        let started = ContinuousClock.now
        let stepOutput = mainStep([MLXArray(current.map(Int32.init))])
        let output = stepOutput[0]
            .asType(.int32).asArray(Int32.self).map(Int.init)
        let elapsed = ContinuousClock.now - started
        stepMilliseconds.append(
            Double(elapsed.components.seconds) * 1_000
                + Double(elapsed.components.attoseconds) / 1e15)
        if expected != output {
            let batchLogits = stepOutput[1].asType(.float32).asArray(Float.self)
            let vocabularySize = expectedLogits[0].count
            for row in prompts.indices where expected[row] != output[row] {
                let rowStart = row * vocabularySize
                let actualRow = Array(batchLogits[rowStart ..< rowStart + vocabularySize])
                let expectedToken = expected[row]
                let actualToken = output[row]
                let maxDelta = zip(expectedLogits[row], actualRow)
                    .map { abs($0 - $1) }.max() ?? 0
                print(
                    "# logit diagnostic mode=\(mode) B=\(batchSize) step=\(step) row=\(row) expected_token=\(expectedToken) actual_token=\(actualToken) scalar_expected_logit=\(expectedLogits[row][expectedToken]) scalar_actual_logit=\(expectedLogits[row][actualToken]) batch_expected_logit=\(actualRow[expectedToken]) batch_actual_logit=\(actualRow[actualToken]) max_abs_delta=\(maxDelta)"
                )
            }
        }
        guard expected == output else {
            throw BatchShapeProbeError.tokenMismatch(
                mode: mode, batch: batchSize, phase: "compiled-step-\(step)",
                expected: expected, actual: output)
        }
        current = expected
    }
    guard mainCounter.count == 1 else {
        throw BatchShapeProbeError.unexpectedTraceCount(
            mode: mode, batch: batchSize, expected: 1, actual: mainCounter.count)
    }

    // Extract one row, remove it from the shared cache, retrace exactly once for the
    // changed membership shape, and prove both paths continue from the correct request.
    let removed = batchSize / 2
    let extractedRows = try prompts.indices.map { row in
        try batchCaches.map { try $0.extract(slot: row) }
    }
    let transitionRows = try prompts.indices.map { row in
        try batchCaches.map { try $0.extract(slot: row) }
    }
    var nextReference: [Int] = []
    nextReference.reserveCapacity(batchSize)
    for row in prompts.indices {
        let expected = forwardToken(
            model: model, input: [current[row]], caches: referenceCaches[row])
        let extractedActual = forwardToken(
            model: model, input: [current[row]], caches: extractedRows[row])
        guard expected == extractedActual else {
            throw BatchShapeProbeError.tokenMismatch(
                mode: mode, batch: batchSize, phase: "pre-filter-extract-row-\(row)",
                expected: [expected], actual: [extractedActual])
        }
        nextReference.append(expected)
    }
    var membershipTraceCount = 0
    if batchSize == 1 {
        // The all-row extraction comparison above is the B=1 membership proof.
    } else {
        let keeping = prompts.indices.filter { $0 != removed }
        try batchCaches.forEach { try $0.filter(keeping: keeping) }

        let keptExpected = keeping.map { nextReference[$0] }

        // Rebuild the same post-removal shape from independently extracted rows. If this
        // and the in-place filter agree, membership indexing is not the source of a drift.
        let rebuiltCaches = try (0 ..< layerCount).map { layer in
            try BatchedCompiledKVCache.merging(
                keeping.map { transitionRows[$0][layer] })
        }
        try rebuiltCaches.forEach { try $0.requireCapacity(for: 1) }
        let rebuiltCounter = CompileTraceCounter()
        let rebuiltStep = makeCompiledBatchStep(
            model: model,
            caches: rebuiltCaches,
            batchSize: keeping.count,
            shapeless: shapeless,
            counter: rebuiltCounter)
        let keptInput = MLXArray(keeping.map { Int32(current[$0]) })
        let rebuiltActual = rebuiltStep([keptInput])[0]
            .asType(.int32).asArray(Int32.self).map(Int.init)

        let membershipCounter = CompileTraceCounter()
        let membershipStep = makeCompiledBatchStep(
            model: model,
            caches: batchCaches,
            batchSize: keeping.count,
            shapeless: shapeless,
            counter: membershipCounter)
        let keptActual = membershipStep([keptInput])[0]
            .asType(.int32).asArray(Int32.self).map(Int.init)
        if keptExpected != rebuiltActual || keptExpected != keptActual {
            print(
                "# transition diagnostic mode=\(mode) B=\(batchSize) expected=\(keptExpected) rebuilt=\(rebuiltActual) filtered=\(keptActual)"
            )
        }
        guard keptExpected == keptActual else {
            throw BatchShapeProbeError.tokenMismatch(
                mode: mode, batch: batchSize, phase: "filter-row-order",
                expected: keptExpected, actual: keptActual)
        }
        guard keptExpected == rebuiltActual else {
            throw BatchShapeProbeError.tokenMismatch(
                mode: mode, batch: batchSize, phase: "rebuilt-row-order",
                expected: keptExpected, actual: rebuiltActual)
        }
        guard rebuiltCounter.count == 1 else {
            throw BatchShapeProbeError.unexpectedTraceCount(
                mode: mode, batch: batchSize, expected: 1, actual: rebuiltCounter.count)
        }
        membershipTraceCount = membershipCounter.count
        guard membershipTraceCount == 1 else {
            throw BatchShapeProbeError.unexpectedTraceCount(
                mode: mode, batch: batchSize, expected: 1, actual: membershipTraceCount)
        }
    }

    let steady = Array(stepMilliseconds.dropFirst()).sorted()
    let steadyP50 = steady.isEmpty ? stepMilliseconds[0] : steady[steady.count / 2]
    return BatchShapeProbeResult(
        mode: mode,
        batchSize: batchSize,
        promptLengths: prompts.map(\.count),
        compiledSteps: compiledSteps,
        mainTraceCount: mainCounter.count,
        membershipTraceCount: membershipTraceCount,
        firstCompiledMilliseconds: stepMilliseconds[0],
        steadyP50Milliseconds: steadyP50,
        maxInitialLogitDelta: maxLogitDelta)
}

private func makeCompiledBatchStep(
    model: any LanguageModel,
    caches: [BatchedCompiledKVCache],
    batchSize: Int,
    shapeless: Bool,
    counter: CompileTraceCounter
) -> ([MLXArray]) -> [MLXArray] {
    let modelCaches: [any KVCache] = caches.map { $0 }
    let state: [any Updatable] = caches.map { $0 }
    let step: ([MLXArray]) -> [MLXArray] = { arguments in
        counter.count += 1
        let input = arguments[0].reshaped([batchSize, 1])
        let logits = model(input, cache: modelCaches)
        let last = logits[0..., -1, 0...]
        return [argMax(last, axis: -1), last]
    }
    return compile(inputs: state, outputs: state, shapeless: shapeless, step)
}

private func forwardToken(
    model: any LanguageModel,
    input: [Int],
    caches: [CompiledKVCache]
) -> Int {
    argmaxIndex(forwardLogits(model: model, input: input, caches: caches))
}

private func forwardLogits(
    model: any LanguageModel,
    input: [Int],
    caches: [CompiledKVCache]
) -> [Float] {
    let modelCaches: [any KVCache] = caches.map { $0 }
    let ids = MLXArray(input.map(Int32.init)).reshaped([1, input.count])
    return model(ids, cache: modelCaches)[0..., -1, 0...]
        .asType(.float32).asArray(Float.self)
}

private func argmaxIndex(_ values: [Float]) -> Int {
    values.indices.max { values[$0] < values[$1] } ?? 0
}

private struct BatchProbeModelConfiguration: Decodable {
    let modelType: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
    }
}

private func requireSupportedDenseModel(modelPath: String) throws {
    let url = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
    let configuration = try JSONDecoder().decode(
        BatchProbeModelConfiguration.self, from: Data(contentsOf: url))
    guard configuration.modelType == "qwen3" else {
        throw BatchShapeProbeError.unsupportedArchitecture(configuration.modelType)
    }
}
