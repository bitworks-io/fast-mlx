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

// MARK: - arg parsing

func takeValue(for flag: String, in args: inout [String]) -> String? {
    guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
    let value = args[idx + 1]
    args.removeSubrange(idx...(idx + 1))
    return value
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

let hostReport = SystemProfiler.probe()
printHostHeader(hostReport)

// The live host is always profiled + printed above (spec: "understand the system the tool is
// running on"); `--box` retargets the capacity math to a preset for planning a different box.
let (targetProfile, targetLabel) = resolveBox(boxArg, hostReport: hostReport)
if targetLabel != "host" {
    print("Capacity computed for: \(targetLabel) — \(gibString(targetProfile.totalRAMBytes)) RAM, "
        + "\(gibString(targetProfile.wiredLimitBytes)) wired — NOT the live host\n")
}

if let modelID {
    runDetail(
        modelID: modelID, context: contextArg, concurrency: concurrencyArg, kvQuant: kvQuantArg,
        profile: targetProfile)
} else {
    runCatalogTable(profile: targetProfile)
}
