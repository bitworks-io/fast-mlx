import Darwin
import Foundation
import MLX
import SpikeCore

private enum KVarNMemoryProbeCLIError: Error, CustomStringConvertible {
    case invalidArguments
    case dirtyHarnessSHA(String)
    case persistentByteMismatch(expected: Int, actual: Int)
    case materializationByteMismatch(expected: Int, actual: Int)
    case incompleteEvaluation(expected: Int, actual: Int)
    case nonFiniteOutput
    case unreconciledMemory(
        label: String, logical: Int, expectedAllocator: Int,
        active: Int, runtimeBaseline: Int, arrayCount: Int,
        allocatorPageBytes: Int)
    case invalidRetainedMemory(
        logical: Int, active: Int, arrayCount: Int)

    var description: String {
        switch self {
        case .invalidArguments:
            return "invalid or missing KVarN memory-probe argument"
        case .dirtyHarnessSHA(let value):
            return "memory evidence requires a clean 40-hex harness SHA; got \(value)"
        case .persistentByteMismatch(let expected, let actual):
            return "persistent byte mismatch expected=\(expected) actual=\(actual)"
        case .materializationByteMismatch(let expected, let actual):
            return "materialization byte mismatch expected=\(expected) actual=\(actual)"
        case .incompleteEvaluation(let expected, let actual):
            return "evaluated-array mismatch expected=\(expected) actual=\(actual)"
        case .nonFiniteOutput:
            return "KVarN probe produced non-finite output"
        case .unreconciledMemory(
            let label, let logical, let expectedAllocator,
            let active, let runtimeBaseline,
            let arrayCount, let allocatorPageBytes):
            return "\(label) memory did not reconcile logical=\(logical) "
                + "expectedAllocator=\(expectedAllocator) active=\(active) "
                + "runtimeBaseline=\(runtimeBaseline) arrays=\(arrayCount) "
                + "page=\(allocatorPageBytes)"
        case .invalidRetainedMemory(let logical, let active, let arrayCount):
            return "measured end memory is below its retained logical floor "
                + "logical=\(logical) active=\(active) arrays=\(arrayCount)"
        }
    }
}

private struct KVarNMemoryProbeResult: Encodable {
    let schemaVersion: Int
    let harnessSHA: String
    let mlxSwiftVersion: String
    let configuration: KVarNMemoryProbeConfiguration
    let persistentLogicalBytes: Int
    let materializationLogicalBytes: Int
    let controlLogicalBytes: Int
    let evaluatedArrayCount: Int
    let expectedEvaluatedArrayCount: Int
    let valuesFinite: Bool
    let emptyBaseline: KVarNMemoryCounters
    let startReconciliation: KVarNMemoryReconciliation
    let endRetainedAccounting: KVarNRetainedMemoryAccounting
    let postDetachCounters: KVarNMemoryCounters
    let postDetachReconciliation: KVarNMemoryReconciliation
    let cacheBoundaryStructuralMemory: KVarNCacheBoundaryStructuralMemory?
    let highWater: KVarNMemoryHighWater
    let status: String
}

func kvarnMemoryProbe(flags: Flags) {
    do {
        guard let phase = KVarNMemoryProbePhase(
            rawValue: flags.string("phase", default: "")),
            let heads = Int(flags.string("heads", default: "8")),
            let headDimension = Int(flags.string("head-dim", default: "128")),
            let groupSize = Int(flags.string("group-size", default: "128")),
            let iterations = Int(flags.string("iterations", default: "8")),
            let capacity = Int(flags.string("capacity", default: "256")),
            let cacheLimitBytes = Int(flags.string("cache-limit-bytes", default: "0")),
            let run = Int(flags.string("run", default: "1")),
            flags.string("json", default: "true") == "true"
        else { throw KVarNMemoryProbeCLIError.invalidArguments }
        let configuration = try KVarNMemoryProbeConfiguration(
            phase: phase, heads: heads, headDimension: headDimension,
            groupSize: groupSize, iterations: iterations, capacity: capacity,
            cacheLimitBytes: cacheLimitBytes, run: run)
        let harnessSHA = try batchProbeHarnessSHA()
        guard KVarNMemoryEvidence.isCleanHarnessSHA(harnessSHA) else {
            throw KVarNMemoryProbeCLIError.dirtyHarnessSHA(harnessSHA)
        }

        Memory.cacheLimit = configuration.cacheLimitBytes
        Memory.clearCache()
        let allocatorPageBytes = Int(getpagesize())
        let emptyBaseline = try warmedEmptyMemoryBaseline()
        let result: KVarNMemoryProbeResult
        switch configuration.phase {
        case .encode:
            result = try runKVarNEncodeMemoryProbe(
                configuration: configuration, harnessSHA: harnessSHA,
                emptyBaseline: emptyBaseline,
                allocatorPageBytes: allocatorPageBytes)
        case .decode:
            result = try runKVarNDecodeMemoryProbe(
                configuration: configuration, harnessSHA: harnessSHA,
                emptyBaseline: emptyBaseline,
                allocatorPageBytes: allocatorPageBytes)
        case .cacheBoundary:
            result = try runKVarNCacheBoundaryMemoryProbe(
                configuration: configuration, harnessSHA: harnessSHA,
                emptyBaseline: emptyBaseline,
                allocatorPageBytes: allocatorPageBytes)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(result)
        guard let line = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteInapplicableStringEncoding)
        }
        print(line)
    } catch {
        print("kvarn-memory-probe FAILED: \(error)")
        exit(1)
    }
}

private func runKVarNEncodeMemoryProbe(
    configuration: KVarNMemoryProbeConfiguration, harnessSHA: String,
    emptyBaseline: KVarNMemoryCounters, allocatorPageBytes: Int
) throws -> KVarNMemoryProbeResult {
    let inputs = makeKVarNTileInputs(configuration)
    let inputArrays = [inputs.keys, inputs.values]
    eval(inputArrays)
    Memory.clearCache()
    let start = resetPeakAndSnapshot()
    let measurement = try measureKVarNEncode(
        inputs: inputs, configuration: configuration)
    let recordArrays = recordArrays(measurement.record)
    let inputLogicalBytes = try checkedByteSum(inputArrays)
    let persistentLogicalBytes = try checkedByteSum(recordArrays)
    let startAllocatorBytes = try allocatorByteSum(
        inputArrays, pageBytes: allocatorPageBytes)
    let postDetachArrays = inputArrays + recordArrays
    let postDetachAllocatorBytes = try allocatorByteSum(
        postDetachArrays, pageBytes: allocatorPageBytes)
    let endMinimumLogicalBytes = try checkedSum([
        inputLogicalBytes, persistentLogicalBytes,
    ])
    Memory.clearCache()
    let postDetach = resetPeakAndSnapshot()
    return try makeProbeResult(
        configuration: configuration, harnessSHA: harnessSHA,
        persistentLogicalBytes: persistentLogicalBytes,
        materializationLogicalBytes: 0, controlLogicalBytes: 0,
        evaluatedArrayCount: measurement.evaluatedArrayCount,
        expectedEvaluatedArrayCount: 8,
        valuesFinite: true,
        emptyBaseline: emptyBaseline,
        allocatorPageBytes: allocatorPageBytes,
        startLogicalBytes: inputLogicalBytes,
        startExpectedAllocatorBytes: startAllocatorBytes,
        startArrayCount: inputArrays.count,
        endMinimumLogicalBytes: endMinimumLogicalBytes,
        endArrayCount: postDetachArrays.count,
        postDetachExpectedAllocatorBytes: postDetachAllocatorBytes,
        start: start, end: measurement.end,
        postDetach: postDetach,
        retained: (inputs, measurement.record))
}

private func runKVarNDecodeMemoryProbe(
    configuration: KVarNMemoryProbeConfiguration, harnessSHA: String,
    emptyBaseline: KVarNMemoryCounters, allocatorPageBytes: Int
) throws -> KVarNMemoryProbeResult {
    let record = try makeEvaluatedKVarNRecord(configuration)
    let storedArrays = recordArrays(record)
    Memory.clearCache()
    let start = resetPeakAndSnapshot()
    let measurement = try measureKVarNDecode(record)
    let persistentLogicalBytes = try checkedByteSum(storedArrays)
    let materializationLogicalBytes = try checkedByteSum(measurement.outputs)
    let startAllocatorBytes = try allocatorByteSum(
        storedArrays, pageBytes: allocatorPageBytes)
    let postDetachArrays = storedArrays + measurement.outputs
    let postDetachAllocatorBytes = try allocatorByteSum(
        postDetachArrays, pageBytes: allocatorPageBytes)
    let endMinimumLogicalBytes = try checkedSum([
        persistentLogicalBytes, materializationLogicalBytes,
    ])
    Memory.clearCache()
    let postDetach = resetPeakAndSnapshot()
    return try makeProbeResult(
        configuration: configuration, harnessSHA: harnessSHA,
        persistentLogicalBytes: persistentLogicalBytes,
        materializationLogicalBytes: materializationLogicalBytes,
        controlLogicalBytes: 0,
        evaluatedArrayCount: measurement.evaluatedArrayCount,
        expectedEvaluatedArrayCount: 2,
        valuesFinite: true,
        emptyBaseline: emptyBaseline,
        allocatorPageBytes: allocatorPageBytes,
        startLogicalBytes: persistentLogicalBytes,
        startExpectedAllocatorBytes: startAllocatorBytes,
        startArrayCount: storedArrays.count,
        endMinimumLogicalBytes: endMinimumLogicalBytes,
        endArrayCount: postDetachArrays.count,
        postDetachExpectedAllocatorBytes: postDetachAllocatorBytes,
        start: start, end: measurement.end,
        postDetach: postDetach,
        retained: (record, measurement.outputs))
}

private func runKVarNCacheBoundaryMemoryProbe(
    configuration: KVarNMemoryProbeConfiguration, harnessSHA: String,
    emptyBaseline: KVarNMemoryCounters, allocatorPageBytes: Int
) throws -> KVarNMemoryProbeResult {
    let cache = try makePreparedKVarNCache(configuration)
    let trigger = makeKVarNTrigger(configuration)
    let triggerArrays = [trigger.keys, trigger.values]
    let startAccounting = try prepareKVarNCacheProbeStart(
        cache: cache, triggerArrays: triggerArrays,
        allocatorPageBytes: allocatorPageBytes)
    Memory.clearCache()
    let start = resetPeakAndSnapshot()
    let measurement = try measureKVarNCacheBoundary(
        cache: cache, trigger: trigger,
        configuration: configuration)
    let finalStateArrays = cache.innerState()
    let actualPersistent = try checkedByteSum(finalStateArrays)
    guard measurement.snapshot.totalPersistentBytes == actualPersistent else {
        throw KVarNMemoryProbeCLIError.persistentByteMismatch(
            expected: measurement.snapshot.totalPersistentBytes,
            actual: actualPersistent)
    }
    let actualMaterialization = try checkedByteSum(measurement.outputs)
    guard measurement.snapshot.materializationWorkspaceBytes == actualMaterialization else {
        throw KVarNMemoryProbeCLIError.materializationByteMismatch(
            expected: measurement.snapshot.materializationWorkspaceBytes,
            actual: actualMaterialization)
    }
    let endMinimumLogicalBytes = try checkedSum([
        actualPersistent,
        try checkedByteSum(triggerArrays),
        actualMaterialization,
    ])
    let postDetachArrays = finalStateArrays + triggerArrays + measurement.outputs
    let postDetachAllocatorBytes = try allocatorByteSum(
        postDetachArrays, pageBytes: allocatorPageBytes)
    Memory.clearCache()
    let postDetach = resetPeakAndSnapshot()
    return try makeProbeResult(
        configuration: configuration, harnessSHA: harnessSHA,
        persistentLogicalBytes: measurement.snapshot.formatPersistentBytes,
        materializationLogicalBytes: actualMaterialization,
        controlLogicalBytes: measurement.snapshot.controlBytes,
        evaluatedArrayCount: measurement.evaluatedArrayCount,
        expectedEvaluatedArrayCount: 15,
        valuesFinite: true,
        emptyBaseline: emptyBaseline,
        allocatorPageBytes: allocatorPageBytes,
        startLogicalBytes: startAccounting.logicalBytes,
        startExpectedAllocatorBytes: startAccounting.expectedAllocatorBytes,
        startArrayCount: startAccounting.arrayCount,
        endMinimumLogicalBytes: endMinimumLogicalBytes,
        endArrayCount: postDetachArrays.count,
        postDetachExpectedAllocatorBytes: postDetachAllocatorBytes,
        start: start, end: measurement.end,
        postDetach: postDetach,
        retained: (cache, trigger, measurement.outputs))
}

private func prepareKVarNCacheProbeStart(
    cache: KVarNKVCache, triggerArrays: [MLXArray],
    allocatorPageBytes: Int
) throws -> (
    logicalBytes: Int, expectedAllocatorBytes: Int, arrayCount: Int
) {
    let stateArrays = cache.innerState()
    let retainedArrays = triggerArrays + stateArrays
    eval(retainedArrays)
    return (
        try checkedByteSum(retainedArrays),
        try allocatorByteSum(retainedArrays, pageBytes: allocatorPageBytes),
        retainedArrays.count)
}

private func measureKVarNEncode(
    inputs: (keys: MLXArray, values: MLXArray),
    configuration: KVarNMemoryProbeConfiguration
) throws -> (
    record: KVarNMLXRecord, end: KVarNMemoryCounters,
    evaluatedArrayCount: Int
) {
    let record = try KVarNMLXCodec.quantize(
        keys: inputs.keys, values: inputs.values,
        configuration: try codecConfiguration(configuration))
    let arrays = recordArrays(record)
    eval(arrays)
    let end = memoryCounters()
    let detached = try KVarNMLXCodec.detachedStorageCopy(of: record)
    eval(recordArrays(detached))
    return (detached, end, arrays.count)
}

private func measureKVarNDecode(
    _ record: KVarNMLXRecord
) throws -> (
    outputs: [MLXArray], end: KVarNMemoryCounters,
    evaluatedArrayCount: Int
) {
    let reconstruction = try KVarNMLXCodec.dequantize(record)
    let outputs = [reconstruction.keys, reconstruction.values]
    eval(outputs)
    let end = memoryCounters()
    let detached = try outputs.map(detachedArray)
    eval(detached)
    return (detached, end, outputs.count)
}

private func measureKVarNCacheBoundary(
    cache: KVarNKVCache,
    trigger: (keys: MLXArray, values: MLXArray),
    configuration: KVarNMemoryProbeConfiguration
) throws -> (
    outputs: [MLXArray], end: KVarNMemoryCounters,
    evaluatedArrayCount: Int, snapshot: KVarNKVCacheStorageSnapshot
) {
    let materialized = cache.update(keys: trigger.keys, values: trigger.values)
    let outputs = [materialized.0, materialized.1]
    let evaluated = outputs + cache.innerState()
    eval(evaluated)
    let end = memoryCounters()
    guard let snapshot = cache.storageSnapshot(),
        cache.offset == configuration.capacity,
        cache.completedTileCount == configuration.completedTileCapacity
    else { throw KVarNMemoryProbeCLIError.invalidArguments }
    let detachedOutputs = try outputs.map(detachedArray)
    eval(detachedOutputs)
    try KVarNMemoryEvidence.detachCacheStorage(cache)
    return (detachedOutputs, end, evaluated.count, snapshot)
}

private func makePreparedKVarNCache(
    _ configuration: KVarNMemoryProbeConfiguration
) throws -> KVarNKVCache {
    let cache = KVarNKVCache(
        capacity: configuration.capacity, tier: .k4v2G128,
        iterations: configuration.iterations)
    let preparationTokens = configuration.capacity - 1
    let shape = [
        1, configuration.heads, preparationTokens, configuration.headDimension,
    ]
    let keys = MLXArray.zeros(shape, dtype: .float16)
    let values = MLXArray.ones(shape, dtype: .float16) * Float16(0.25)
    eval([keys, values])
    let materialized = cache.update(keys: keys, values: values)
    eval([materialized.0, materialized.1] + cache.innerState())
    guard cache.offset == preparationTokens else {
        throw KVarNMemoryProbeCLIError.invalidArguments
    }
    try KVarNMemoryEvidence.detachCacheStorage(cache)
    return cache
}

private func makeEvaluatedKVarNRecord(
    _ configuration: KVarNMemoryProbeConfiguration
) throws -> KVarNMLXRecord {
    let inputs = makeKVarNTileInputs(configuration)
    eval([inputs.keys, inputs.values])
    let record = try KVarNMLXCodec.quantize(
        keys: inputs.keys, values: inputs.values,
        configuration: try codecConfiguration(configuration))
    eval(recordArrays(record))
    let detached = try KVarNMLXCodec.detachedStorageCopy(of: record)
    eval(recordArrays(detached))
    return detached
}

private func makeKVarNTileInputs(
    _ configuration: KVarNMemoryProbeConfiguration
) -> (keys: MLXArray, values: MLXArray) {
    let count = configuration.tileElementCount
    let keys = (0 ..< count).map {
        Float16(sin(Double($0) * 0.011) + 0.25 * cos(Double($0) * 0.037))
    }
    let values = (0 ..< count).map {
        Float16(cos(Double($0) * 0.019) - 0.3 * sin(Double($0) * 0.023))
    }
    let shape = [
        1, configuration.heads, configuration.groupSize,
        configuration.headDimension,
    ]
    return (
        MLXArray(keys).reshaped(shape),
        MLXArray(values).reshaped(shape))
}

private func makeKVarNTrigger(
    _ configuration: KVarNMemoryProbeConfiguration
) -> (keys: MLXArray, values: MLXArray) {
    let count = configuration.heads * configuration.headDimension
    let shape = [1, configuration.heads, 1, configuration.headDimension]
    return (
        MLXArray([Float16](repeating: 1, count: count)).reshaped(shape),
        MLXArray([Float16](repeating: -0.5, count: count)).reshaped(shape))
}

private func codecConfiguration(
    _ configuration: KVarNMemoryProbeConfiguration
) throws -> KVarNMLXConfiguration {
    try KVarNMLXConfiguration(
        headDimension: configuration.headDimension,
        groupSize: configuration.groupSize,
        keyBits: 4, valueBits: 2,
        iterations: configuration.iterations)
}

private func recordArrays(_ record: KVarNMLXRecord) -> [MLXArray] {
    [
        record.keyPayload, record.keyAbsorbedScale, record.keyAbsorbedBias,
        record.keyTokenScale, record.valuePayload, record.valueChannelScale,
        record.valueAbsorbedScale, record.valueAbsorbedBias,
    ]
}

private func detachedArray(_ array: MLXArray) throws -> MLXArray {
    switch array.dtype {
    case .uint8:
        return MLXArray(array.asArray(UInt8.self)).reshaped(array.shape)
    case .float16:
        return MLXArray(array.asArray(Float16.self)).reshaped(array.shape)
    case .int32:
        return MLXArray(array.asArray(Int32.self)).reshaped(array.shape)
    default:
        throw KVarNMemoryEvidenceError.unsupportedCacheStorageDType
    }
}

private func warmedEmptyMemoryBaseline() throws -> KVarNMemoryCounters {
    warmMLXAllocator()
    Memory.clearCache()
    Memory.peakMemory = 0
    let baseline = memoryCounters()
    guard baseline.cacheBytes == 0 else {
        throw KVarNMemoryEvidenceError.allocatorCacheNotEmpty
    }
    guard baseline.peakActiveBytes == 0 else {
        throw KVarNMemoryEvidenceError.invalidMemoryCounters
    }
    return baseline
}

private func warmMLXAllocator() {
    let warmup = MLXArray([Float16(0)]) + Float16(1)
    eval([warmup])
    withExtendedLifetime(warmup) {}
}

private func resetPeakAndSnapshot() -> KVarNMemoryCounters {
    Memory.peakMemory = 0
    return memoryCounters()
}

private func memoryCounters() -> KVarNMemoryCounters {
    let snapshot = Memory.snapshot()
    return KVarNMemoryCounters(
        activeBytes: snapshot.activeMemory,
        cacheBytes: snapshot.cacheMemory,
        peakActiveBytes: snapshot.peakMemory)
}

private func checkedByteSum(_ arrays: [MLXArray]) throws -> Int {
    try checkedSum(arrays.map(\.nbytes))
}

private func allocatorByteSum(
    _ arrays: [MLXArray], pageBytes: Int
) throws -> Int {
    try checkedSum(try arrays.map {
        try KVarNMemoryEvidence.allocatorBytes(
            forLogicalBytes: $0.nbytes, pageBytes: pageBytes)
    })
}

private func checkedSum(_ values: [Int]) throws -> Int {
    try values.reduce(into: 0) { result, value in
        let (next, overflow) = result.addingReportingOverflow(value)
        guard !overflow else { throw KVarNMemoryEvidenceError.invalidMemoryCounters }
        result = next
    }
}

private func makeProbeResult<T>(
    configuration: KVarNMemoryProbeConfiguration,
    harnessSHA: String,
    persistentLogicalBytes: Int,
    materializationLogicalBytes: Int,
    controlLogicalBytes: Int,
    evaluatedArrayCount: Int,
    expectedEvaluatedArrayCount: Int,
    valuesFinite: Bool,
    emptyBaseline: KVarNMemoryCounters,
    allocatorPageBytes: Int,
    startLogicalBytes: Int,
    startExpectedAllocatorBytes: Int,
    startArrayCount: Int,
    endMinimumLogicalBytes: Int,
    endArrayCount: Int,
    postDetachExpectedAllocatorBytes: Int,
    start: KVarNMemoryCounters,
    end: KVarNMemoryCounters,
    postDetach: KVarNMemoryCounters,
    retained: T
) throws -> KVarNMemoryProbeResult {
    guard evaluatedArrayCount == expectedEvaluatedArrayCount else {
        throw KVarNMemoryProbeCLIError.incompleteEvaluation(
            expected: expectedEvaluatedArrayCount, actual: evaluatedArrayCount)
    }
    guard valuesFinite else { throw KVarNMemoryProbeCLIError.nonFiniteOutput }
    guard emptyBaseline.cacheBytes == 0, postDetach.cacheBytes == 0 else {
        throw KVarNMemoryEvidenceError.allocatorCacheNotEmpty
    }
    guard emptyBaseline.peakActiveBytes == 0,
        postDetach.peakActiveBytes == 0
    else { throw KVarNMemoryEvidenceError.invalidMemoryCounters }
    guard emptyBaseline.activeBytes < allocatorPageBytes else {
        throw KVarNMemoryProbeCLIError.unreconciledMemory(
            label: "empty baseline", logical: 0,
            expectedAllocator: 0, active: emptyBaseline.activeBytes,
            runtimeBaseline: emptyBaseline.activeBytes,
            arrayCount: 0, allocatorPageBytes: allocatorPageBytes)
    }
    let highWater = try KVarNMemoryHighWater(start: start, end: end)
    let startReconciliation = try reconcileMemory(
        label: "start", logicalBytes: startLogicalBytes,
        expectedAllocatorBytes: startExpectedAllocatorBytes,
        activeBytes: start.activeBytes,
        runtimeBaselineBytes: emptyBaseline.activeBytes,
        arrayCount: startArrayCount,
        allocatorPageBytes: allocatorPageBytes)
    let endRetainedAccounting: KVarNRetainedMemoryAccounting
    do {
        endRetainedAccounting = try KVarNRetainedMemoryAccounting(
            minimumLogicalBytes: endMinimumLogicalBytes,
            activeBytes: end.activeBytes, arrayCount: endArrayCount)
    } catch {
        throw KVarNMemoryProbeCLIError.invalidRetainedMemory(
            logical: endMinimumLogicalBytes,
            active: end.activeBytes, arrayCount: endArrayCount)
    }
    let postDetachReconciliation = try reconcileMemory(
        label: "post-detach", logicalBytes: endMinimumLogicalBytes,
        expectedAllocatorBytes: postDetachExpectedAllocatorBytes,
        activeBytes: postDetach.activeBytes,
        runtimeBaselineBytes: emptyBaseline.activeBytes,
        arrayCount: endArrayCount,
        allocatorPageBytes: allocatorPageBytes)
    let structuralMemory: KVarNCacheBoundaryStructuralMemory?
    if configuration.phase == .cacheBoundary {
        structuralMemory = try KVarNCacheBoundaryStructuralMemory(
            configuration: configuration,
            allocatorPageBytes: allocatorPageBytes,
            startActiveBytes: start.activeBytes,
            observedPeakActiveBytes: highWater.observedPeakActiveBytes)
    } else {
        structuralMemory = nil
    }
    return withExtendedLifetime(retained) {
        KVarNMemoryProbeResult(
            schemaVersion: 2, harnessSHA: harnessSHA,
            mlxSwiftVersion: "0.31.6", configuration: configuration,
            persistentLogicalBytes: persistentLogicalBytes,
            materializationLogicalBytes: materializationLogicalBytes,
            controlLogicalBytes: controlLogicalBytes,
            evaluatedArrayCount: evaluatedArrayCount,
            expectedEvaluatedArrayCount: expectedEvaluatedArrayCount,
            valuesFinite: valuesFinite,
            emptyBaseline: emptyBaseline,
            startReconciliation: startReconciliation,
            endRetainedAccounting: endRetainedAccounting,
            postDetachCounters: postDetach,
            postDetachReconciliation: postDetachReconciliation,
            cacheBoundaryStructuralMemory: structuralMemory,
            highWater: highWater,
            status: "PASS")
    }
}

private func reconcileMemory(
    label: String, logicalBytes: Int, expectedAllocatorBytes: Int,
    activeBytes: Int, runtimeBaselineBytes: Int,
    arrayCount: Int, allocatorPageBytes: Int
) throws -> KVarNMemoryReconciliation {
    do {
        return try KVarNMemoryReconciliation(
            logicalBytes: logicalBytes,
            expectedAllocatorBytes: expectedAllocatorBytes,
            activeBytes: activeBytes,
            runtimeBaselineBytes: runtimeBaselineBytes,
            arrayCount: arrayCount, allocatorPageBytes: allocatorPageBytes)
    } catch {
        throw KVarNMemoryProbeCLIError.unreconciledMemory(
            label: label, logical: logicalBytes,
            expectedAllocator: expectedAllocatorBytes,
            active: activeBytes, runtimeBaseline: runtimeBaselineBytes,
            arrayCount: arrayCount,
            allocatorPageBytes: allocatorPageBytes)
    }
}
