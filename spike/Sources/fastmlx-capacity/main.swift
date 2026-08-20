import Foundation
import HarnessCore
import SystemProfiler

/// `fastmlx-capacity` — the operator-facing capacity CLI (spec §5). Wires the real host
/// introspection (`SystemProfiler.probe()`) into the pure capacity model (`HarnessCore.
/// CapacityModel`), printing either a catalog-wide table (default mode) or a full per-term
/// breakdown for one model (`--model`). MLX-free: only `SystemProfiler` + `HarnessCore`.

let gib = 1024.0 * 1024.0 * 1024.0

func gibString(_ bytes: Int, decimals: Int = 2) -> String {
    String(format: "%.\(decimals)f GiB", Double(bytes) / gib)
}

func gibString(_ bytes: Double, decimals: Int = 2) -> String {
    String(format: "%.\(decimals)f GiB", bytes / gib)
}

func printHostHeader(_ report: HostReport) {
    print("== Host profile ==")
    print("chip:                \(report.chip)")
    print("cores:                \(report.pCores)P + \(report.eCores)E")
    print("total RAM:            \(gibString(report.totalRAMBytes))")
    var wiredLimitLine = "wired limit:          \(gibString(report.wiredLimitBytes))"
    if report.wiredLimitIsDefault {
        wiredLimitLine += " (system default, synthesized)"
    }
    print(wiredLimitLine)
    if let alloc = report.currentGPUAllocBytes {
        print("current GPU alloc:    \(gibString(alloc))")
    }
    if let recommended = report.recommendedWorkingSetBytes {
        print("recommended workset:  \(gibString(recommended))")
    }
    if let internalDisk = report.diskInternal {
        print("disk internal:        \(internalDisk)")
    }
    if let free = report.diskFreeBytes {
        print("disk free:            \(gibString(free))")
    }
    print("")
}

func kvDisplayString(_ prediction: CapacityPrediction, verbose: Bool = false) -> String {
    if prediction.derivable { return gibString(prediction.kvBytes) }
    return verbose ? "n/a (not derivable)" : "n/a"
}

func runCatalogTable(profile: SystemProfile) {
    print("== Capacity table (effective default context, concurrency 1, fp16 KV) ==")
    let columns = ["model", "ctx", "KV", "color", "binding", "ceiling"]
    print(columns[0].padding(toLength: 24, withPad: " ", startingAt: 0)
        + columns[1].padding(toLength: 10, withPad: " ", startingAt: 0)
        + columns[2].padding(toLength: 14, withPad: " ", startingAt: 0)
        + columns[3].padding(toLength: 8, withPad: " ", startingAt: 0)
        + columns[4].padding(toLength: 22, withPad: " ", startingAt: 0)
        + columns[5])

    for model in ModelArchProfile.catalog {
        let context = CapacityModel.effectiveDefaultContext(model)
        let prediction = CapacityModel.predictPeakBytes(
            model: model, context: context, concurrency: 1, kvQuant: .fp16, profile: profile)
        let verdict = CapacityModel.classify(
            prediction, profile: profile, weightsBytes: Double(model.weightsBytes4bitEstimate))
        let ceiling = CapacityModel.contextCeiling(
            model: model, profile: profile, kvQuant: .fp16, concurrency: 1)

        let row = model.id.padding(toLength: 24, withPad: " ", startingAt: 0)
            + "\(context)".padding(toLength: 10, withPad: " ", startingAt: 0)
            + kvDisplayString(prediction).padding(toLength: 14, withPad: " ", startingAt: 0)
            + verdict.color.rawValue.padding(toLength: 8, withPad: " ", startingAt: 0)
            + verdict.bindingConstraint.rawValue.padding(toLength: 22, withPad: " ", startingAt: 0)
            + "\(ceiling)"
        print(row)
    }
}

func runDetail(
    modelID: String, context: Int?, concurrency: Int, kvQuant: KVQuantTier, profile: SystemProfile
) {
    guard let model = ModelArchProfile.catalog.first(where: { $0.id == modelID }) else {
        let validIDs = ModelArchProfile.catalog.map(\.id).joined(separator: ", ")
        FileHandle.standardError.write("error: unknown model id '\(modelID)'\nvalid ids: \(validIDs)\n".data(using: .utf8)!)
        exit(1)
    }

    let effectiveDefault = CapacityModel.effectiveDefaultContext(model)
    let resolvedContext = context ?? effectiveDefault

    let prediction = CapacityModel.predictPeakBytes(
        model: model, context: resolvedContext, concurrency: concurrency, kvQuant: kvQuant, profile: profile)
    let verdict = CapacityModel.classify(
        prediction, profile: profile, weightsBytes: Double(model.weightsBytes4bitEstimate))
    let ceiling = CapacityModel.contextCeiling(
        model: model, profile: profile, kvQuant: kvQuant, concurrency: concurrency)

    print("== \(model.id) ==")
    print("modelType:            \(model.modelType.rawValue)")
    print("nativeMaxContext:     \(model.nativeMaxContext)")
    print("context:              \(resolvedContext)")
    print("concurrency:          \(concurrency)")
    print("kvQuant:              \(kvQuant.rawValue)")
    print("")
    print("-- prediction terms --")
    print("weights:              \(gibString(prediction.weightsBytes))")
    print("KV:                   \(kvDisplayString(prediction, verbose: true))")
    print("transient prefill:    \(gibString(prediction.transientPrefillPeakBytes))")
    print("allocator headroom:   \(gibString(prediction.allocatorHeadroomBytes))")
    print("total:                \(gibString(prediction.totalBytes))")
    print("")
    print("-- verdict --")
    print("color:                \(verdict.color.rawValue)")
    print("binding constraint:   \(verdict.bindingConstraint.rawValue)")
    print("ratio:                \(verdict.ratio)")
    print("mitigation:           \(verdict.suggestedMitigation)")
    print("")
    print("-- context tunable --")
    print("effectiveDefault:     \(effectiveDefault)")
    if let advisory = CapacityModel.defaultContextAdvisory(model) {
        print("defaultAdvisory:      \(advisory.rawValue)")
    }
    print("contextCeiling:       \(ceiling)")
}

/// `--sizer`: renders `ModelSizer.report(...)` — the model-sizer moat pillar (llmfit-like:
/// "given this box's RAM, which quantized builds fit and at what context") — as a readable table,
/// or as JSON with `--json`. All numbers come straight out of `ModelSizer`/`CapacityModel`; this
/// function only formats them.
func runSizerTable(box: SystemProfile, context: Int?, kvQuant: KVQuantTier, concurrency: Int, json: Bool) {
    let rows = ModelSizer.report(box: box, context: context, kvQuant: kvQuant, concurrency: concurrency)

    if json {
        print(sizerReportJSON(rows))
        return
    }

    print("== Model sizer (MODELED ESTIMATE, not a measured guarantee) ==")
    print("kvQuant: \(kvQuant.rawValue), concurrency: \(concurrency)")
    if !box.wiredLimitIsMeasured {
        print("NOTE: this box's wired-memory limit is ESTIMATED (not read from hardware) — headroom numbers are approximate.")
    }
    if !rows.isEmpty && !rows[0].estimateIsMeasured && box.wiredLimitIsMeasured {
        print("NOTE: \(kvQuant.rawValue) is an ⚠️ EXPERIMENTAL/UNMEASURED placeholder KV tier — treat fit/ceiling numbers as more speculative than usual.")
    }
    print("")

    let bitsOrder = weightBitOptionsDefault.sorted()
    var header = "model".padding(toLength: 26, withPad: " ", startingAt: 0)
        + "ctx".padding(toLength: 10, withPad: " ", startingAt: 0)
    for bits in bitsOrder {
        header += "\(bits)-bit fit".padding(toLength: 12, withPad: " ", startingAt: 0)
            + "\(bits)-bit maxCtx".padding(toLength: 16, withPad: " ", startingAt: 0)
    }
    print(header)

    for model in ModelArchProfile.catalog {
        guard let first = rows.first(where: { $0.modelID == model.id }) else { continue }
        var row = model.id.padding(toLength: 26, withPad: " ", startingAt: 0)
            + "\(first.requestedContext)".padding(toLength: 10, withPad: " ", startingAt: 0)
        for bits in bitsOrder {
            guard let fit = rows.first(where: { $0.modelID == model.id && $0.weightBits == bits }) else { continue }
            row += (fit.fits ? "yes" : "no").padding(toLength: 12, withPad: " ", startingAt: 0)
                + "\(fit.maxContextThatFits)".padding(toLength: 16, withPad: " ", startingAt: 0)
        }
        print(row)
    }
}

private let weightBitOptionsDefault = [4, 8]

/// Minimal hand-rolled JSON serializer for `[ModelFit]` (kept local to the CLI rather than adding
/// a `Codable`/Foundation-JSON dependency onto `HarnessCore`'s `ModelFit`, which the spec defines
/// as `Equatable, Sendable` only).
func sizerReportJSON(_ rows: [ModelFit]) -> String {
    func field(_ key: String, _ value: String) -> String { "\"\(key)\":\(value)" }
    func str(_ s: String) -> String {
        "\"" + s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }
    let objects = rows.map { row -> String in
        "{"
            + field("modelID", str(row.modelID)) + ","
            + field("weightBits", "\(row.weightBits)") + ","
            + field("weightsBytes", "\(row.weightsBytes)") + ","
            + field("kvBytesAtContext", "\(row.kvBytesAtContext)") + ","
            + field("transientPrefillBytes", "\(row.transientPrefillBytes)") + ","
            + field("totalPeakBytes", "\(row.totalPeakBytes)") + ","
            + field("fits", "\(row.fits)") + ","
            + field("maxContextThatFits", "\(row.maxContextThatFits)") + ","
            + field("requestedContext", "\(row.requestedContext)") + ","
            + field("classification", str(row.classification.rawValue)) + ","
            + field("estimateIsMeasured", "\(row.estimateIsMeasured)")
            + "}"
    }
    return "[" + objects.joined(separator: ",") + "]"
}

// MARK: - arg parsing

func takeValue(for flag: String, in args: inout [String]) -> String? {
    guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
    let value = args[idx + 1]
    args.removeSubrange(idx...(idx + 1))
    return value
}

func takeFlag(_ flag: String, in args: inout [String]) -> Bool {
    guard let idx = args.firstIndex(of: flag) else { return false }
    args.remove(at: idx)
    return true
}

/// Resolve `--box` to the profile capacity is computed against. Defaults to the live host; the
/// named presets let an operator plan for a target box they aren't currently running on (e.g.
/// sizing a 512GB-Ultra deployment from a laptop) — capacity planning, not just current-box report.
func resolveBox(_ arg: String?, hostReport: HostReport) -> (SystemProfile, String) {
    switch arg {
    case nil, "host": return (hostReport.systemProfile, "host")
    case "m5Max128": return (.m5Max128, "m5Max128")
    case "m3Ultra256": return (.m3Ultra256, "m3Ultra256")
    case "m3Ultra512": return (.m3Ultra512, "m3Ultra512")
    default:
        FileHandle.standardError.write(
            "error: unknown --box '\(arg!)'\nvalid: host, m5Max128, m3Ultra256, m3Ultra512\n".data(using: .utf8)!)
        exit(1)
    }
}

var args = Array(CommandLine.arguments.dropFirst())
let modelID = takeValue(for: "--model", in: &args)
let contextArg = takeValue(for: "--context", in: &args).flatMap { Int($0) }
let concurrencyArg = takeValue(for: "--concurrency", in: &args).flatMap { Int($0) } ?? 1
let kvQuantArg = takeValue(for: "--kv-quant", in: &args).flatMap { KVQuantTier(rawValue: $0) } ?? .fp16
let boxArg = takeValue(for: "--box", in: &args)
// `--auto`: target the live host via HarnessCore's pure-Swift `SystemProfile.detectHost()`
// (sysctl RAM/chip + an estimated wired limit) instead of a `--box` preset. `--sizer`: run the
// model-sizer report instead of the default catalog table / `--model` detail view.
let autoFlag = takeFlag("--auto", in: &args)
let sizerFlag = takeFlag("--sizer", in: &args)
let jsonFlag = takeFlag("--json", in: &args)
// `--sizer-matrix`: emit `ModelSizer.report(...)` as a schema-tagged `sizer-matrix/v1` artifact
// (`SizerMatrixArtifact.encodedJSON()`) instead of the table/`--sizer --json` array. This is the
// data source a later increment's public "which model fits which Mac" sizer page consumes — always
// JSON, regardless of `--json` (that flag only gates the existing `--sizer` array output).
let sizerMatrixFlag = takeFlag("--sizer-matrix", in: &args)
// `--quant-reliability <artifact.json>`: render a per-quant tool-call reliability artifact
// (`quant-reliability/v1`, produced off-box by `scripts/bench-tool-calling.py --quants`). This is a
// pure render-and-exit path (no host capacity math), hosted here because the capacity CLI is the
// MLX-free operator surface — it fails closed on a foreign schema tag or malformed JSON.
let quantReliabilityPathArg = takeValue(for: "--quant-reliability", in: &args)

// `--quant-reliability` is a pure artifact render: no host introspection, no capacity math. Handle
// it before probing the host so the output is just the reliability rows (and a clean non-zero exit
// on a bad/foreign artifact).
if let path = quantReliabilityPathArg {
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        for line in try QuantReliabilityArtifactRenderer.renderLines(from: data) {
            print(line)
        }
        exit(0)
    } catch let error as QuantReliabilityArtifactRenderer.RenderError {
        FileHandle.standardError.write("error: \(error.description)\n".data(using: .utf8)!)
        exit(2)
    } catch {
        FileHandle.standardError.write(
            "error: could not read --quant-reliability file '\(path)': \(error.localizedDescription)\n"
                .data(using: .utf8)!)
        exit(2)
    }
}

let hostReport = SystemProfiler.probe()
// `--sizer-matrix` is a machine-consumed data source (its artifact promises "always JSON"): keep its
// stdout pure JSON by suppressing the human host-profile header + the "Capacity computed for" banner,
// the same clean-surface treatment `--quant-reliability` gets via its early exit above. The host is
// still probed (the default target is the live host); only the human framing is elided.
if !sizerMatrixFlag {
    printHostHeader(hostReport)
}

// The live host is always profiled + printed above (spec: "understand the system the tool is
// running on"); `--box` retargets the capacity math to a preset for planning a different box.
// `--auto` retargets it to a fresh `detectHost()` read instead (same host, but sourced from
// HarnessCore's own minimal sysctl probe rather than the `SystemProfiler.probe()` above, and with
// an explicitly ESTIMATED wired limit — see `wiredLimitIsMeasured`).
let targetProfile: SystemProfile
let targetLabel: String
if autoFlag {
    targetProfile = SystemProfile.detectHost()
    targetLabel = "auto"
} else {
    (targetProfile, targetLabel) = resolveBox(boxArg, hostReport: hostReport)
}
if !sizerMatrixFlag {
    if targetLabel == "auto" {
        print("Capacity computed for: auto-detected host — \(gibString(targetProfile.totalRAMBytes)) RAM, "
            + "\(gibString(targetProfile.wiredLimitBytes)) wired (ESTIMATED, not measured)\n")
    } else if targetLabel != "host" {
        print("Capacity computed for: \(targetLabel) — \(gibString(targetProfile.totalRAMBytes)) RAM, "
            + "\(gibString(targetProfile.wiredLimitBytes)) wired — NOT the live host\n")
    }
}

if sizerMatrixFlag {
    let artifact = SizerMatrixArtifact.build(
        box: targetProfile, boxLabel: targetLabel, context: contextArg, kvQuant: kvQuantArg,
        concurrency: concurrencyArg)
    print(artifact.encodedJSON())
} else if sizerFlag {
    runSizerTable(box: targetProfile, context: contextArg, kvQuant: kvQuantArg, concurrency: concurrencyArg, json: jsonFlag)
} else if let modelID {
    runDetail(
        modelID: modelID, context: contextArg, concurrency: concurrencyArg, kvQuant: kvQuantArg,
        profile: targetProfile)
} else {
    runCatalogTable(profile: targetProfile)
}
