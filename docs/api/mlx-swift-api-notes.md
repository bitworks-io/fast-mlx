# mlx-swift / mlx-swift-lm API notes (Swift decode-loop spike)

Verified 2026-07-08 on `llmbench@192.168.1.252` (Xcode 26.6, Swift 6.3.3, macOS 26.5.2) by reading
the resolved SwiftPM checkouts under
`~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/SourcePackages/checkouts/`.
Every mlx-swift / mlx-swift-lm call in later spike tasks must match the signatures below —
several differ from the plan's assumptions (noted inline).

## Pinned versions (`spike/Package.resolved`)

| package | version | revision |
|---|---|---|
| `ml-explore/mlx-swift` | 0.31.6 | `0bb916c67f4b9e5c682cbe02a42c701c93ab5021` |
| `ml-explore/mlx-swift-lm` | `main` branch | `702e5a0eaf990e1f6d3db2b6e7d8872858a44055` |
| `huggingface/swift-huggingface` | `from: 0.9.0` | (transitive, needed for `#hubDownloader()` / not used by the local-dir path but part of the macro integration) |
| `huggingface/swift-transformers` | `from: 1.3.0` | (transitive; provides `Tokenizers.AutoTokenizer` backing `#huggingFaceTokenizerLoader()`) |
| `apple/swift-argument-parser` | 1.8.2 | |
| `apple/swift-numerics` | 1.1.1 | |
| `swiftlang/swift-syntax` | 603.0.2 | |

`Package.swift` pins mlx-swift `exact: "0.31.6"` and mlx-swift-lm to the exact `revision` above
(not `branch: "main"`) so the spike is reproducible.

**Build gotcha (not an API issue, but required for Task 1):** `xcodebuild` fails validating the
`CudaBuild` plugin in mlx-swift with `Validate plug-in "CudaBuild" in package "mlx-swift"` unless
you pass `-skipPackagePluginValidation`. Also, a bare Xcode 26.6 install does not ship the Metal
compiler — first build fails every `.metal` file with `cannot execute tool 'metal' due to missing
Metal Toolchain`; fix once with `xcodebuild -downloadComponent MetalToolchain`. Full working build
command:
```
xcodebuild -scheme spike-cli -destination 'platform=macOS' -skipPackagePluginValidation build
```

## 1. Load model + tokenizer from a local path

**Plan's assumption:** `LLMModelFactory.shared.loadContainer(...)` yielding a `ModelContext` with
`.model` and `.tokenizer`. **Reality is more explicit** — mlx-swift-lm's `main` branch (post-2.x
"upgrade") decoupled the concrete `Downloader`/`Tokenizer` implementations from the core package.
For a **local directory** (our case — the model is already unpacked on disk, no download needed)
the call is a **free function**, not a factory method, and does NOT need a `Downloader` at all —
but it does need a concrete `TokenizerLoader`, supplied here via the `MLXHuggingFace` macro
integration (`#huggingFaceTokenizerLoader()`, backed by `swift-transformers`' `Tokenizers.AutoTokenizer`).

```swift
// MLXLMCommon/ModelFactory.swift:386
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> sending ModelContext
```

`ModelContext` (`MLXLMCommon/ModelFactory.swift:75`):
```swift
public struct ModelContext {
    public var configuration: ModelConfiguration
    public var model: any LanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer   // MLXLMCommon.Tokenizer, see §6
}
```

Concrete call used by the spike's `api-check`:
```swift
import MLXHuggingFace   // provides #huggingFaceTokenizerLoader()

let ctx = try await loadModel(
    from: URL(fileURLWithPath: modelPath),
    using: #huggingFaceTokenizerLoader()
)
// ctx.model, ctx.tokenizer
```

There is also a `ModelContainer`-returning sibling (`loadModelContainer(from:using:)`) that wraps
the context in an `actor` for cross-context Sendable use; the spike's single-owner `InferenceActor`
supersedes that, so the raw `ModelContext` is enough.

**New dependency this pulled in that the plan didn't list:** `MLXHuggingFace` product (from
mlx-swift-lm) + `huggingface/swift-huggingface` + `huggingface/swift-transformers` packages, added
to `Package.swift` and to `SpikeCore`'s target dependencies.

## 2. Forward pass returning logits

**Plan's assumption:** `model(inputs, cache: cache) -> MLXArray` logits `[1, seqLen, vocab]`.
**Confirmed correct**, from the `LanguageModel` protocol (`MLXLMCommon/LanguageModel.swift:252`):

```swift
protocol LanguageModel: BaseLanguageModel {
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray
    // ...
}
```

So `model(ids, cache: cache)` (Swift `callAsFunction` sugar) works exactly as assumed, returning
`[1, seqLen, vocab]` logits. (There is also a richer `callAsFunction(_ input: LMInput.Text, cache:
state:) -> LMOutput` for models needing extra state, and a `prepare(_:cache:windowSize:) throws ->
PrepareResult` step used by mlx-swift-lm's own generation loop — the spike intentionally bypasses
`prepare`/the built-in generation loop per the plan's "replace the generation loop" architecture,
calling `callAsFunction` directly.)

## 3. KVCache type + construction

**Plan's assumption:** `MLXLMCommon` `KVCache` array, one per layer. **Confirmed.**

```swift
// MLXLMCommon/KVCache.swift:46
public protocol KVCache: Evaluatable {
    var offset: Int { get }
    var ropeOffset: RoPEOffset { get }
    var maxSize: Int? { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    var state: [MLXArray] { get set }
}
```

Construction, via the model itself (`MLXLMCommon/LanguageModel.swift:256`):
```swift
protocol LanguageModel {
    func newCache(parameters: GenerateParameters?) -> [KVCache]
}
```
Default implementation for models conforming to `KVCacheDimensionProvider` creates one
`KVCacheSimple` (`MLXLMCommon/KVCache.swift:372`) per layer (`kvHeads.count` layers), or a
`RotatingKVCache` if `parameters.maxKVSize` is set. For the spike: `model.newCache(parameters: nil)`.

## 4. Greedy sample on GPU + CPU readback

**Confirmed as assumed**, with one important caveat.

```swift
// MLX/Ops+Array.swift:244
public func argMax(
    _ array: MLXArray, axis: Int, keepDims: Bool = false, stream: StreamOrDevice = .default
) -> MLXArray
```
```swift
// MLX/MLXArray.swift:331
public func item<T: HasDType>(_ type: T.Type) -> T
```

**Caveat (matters for the no-sync-readback design in Task 5):** `item(_:)`'s implementation calls
`self.eval()` internally before reading the scalar (`MLXArray.swift:331` -> `eval()` at top of the
method body). So `.item(Int.self)` is *itself* a synchronous, blocking GPU-sync point on whatever
array it's called on — the "submit next before readback" pattern only avoids the *stall on the
critical path* if the *next* forward's `asyncEval` was already issued (and thus already running on
the GPU queue) before you call `.item()` on the *current* token. Calling `.item()` does not block
on unrelated in-flight `asyncEval` work; it only forces evaluation of the array it's called on.

## 5. Async eval

**Confirmed both exist** (`MLX/Transforms+Eval.swift`):
```swift
public func eval(_ arrays: MLXArray...)
public func eval(_ arrays: some Collection<MLXArray>)
public func asyncEval(_ arrays: some Collection<MLXArray>)
public func asyncEval(_ arrays: Any...)
```
`asyncEval` calls `mlx_async_eval` (schedules on MLX's internal eval queue, returns immediately);
`eval` calls the blocking `mlx_eval`. Matches the plan's assumption exactly.

## 6. Tokenizer encode / decode

**Plan's assumption:** `encode(text:)` / `decode(tokens:)` on the huggingface/swift-transformers
`Tokenizer` directly. **Reality:** mlx-swift-lm defines its **own** `Tokenizer` protocol in
`MLXLMCommon` (not swift-transformers' type directly) that concrete tokenizers (including
swift-transformers' `Tokenizers.AutoTokenizer`, via the `MLXHuggingFace` macro) are adapted to:

```swift
// MLXLMCommon/Tokenizer.swift:6
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?
    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int]
}
extension Tokenizer {
    public func encode(text: String) -> [Int] { encode(text: text, addSpecialTokens: true) }
}
```
So the spike calls `ctx.tokenizer.encode(text: prompt)` and
`ctx.tokenizer.decode(tokenIds: [id], skipSpecialTokens: true)` — same *shape* as assumed, but the
label is `text:`/`tokenIds:`, not positional, and it's mlx-swift-lm's own protocol.

## Executable entry point gotcha (not mlx-swift, but blocks Task 2's smoke test)

The first `api-check` implementation used `DispatchSemaphore().wait()` on the top-level thread while
spawning `Task { await apiCheck(...) }` to bridge sync top-level code into async. **This deadlocks
silently** — no crash, no error, just an indefinitely stuck process with zero I/O (confirmed via
`lsof`/`ps`: no file opens, no network, RSS never grows past ~36MB, 6+ minutes elapsed). Root cause:
blocking the thread Swift's concurrency runtime needs for the task's executor. Fixed by using a
proper `@main struct SpikeCLI { static func main() async { ... } }` entry point instead (requires
renaming `main.swift` -> `SpikeCLI.swift`, since SwiftPM treats a literal `main.swift` filename as
implicit top-level-code and rejects a coexisting `@main` attribute with "cannot be used in a module
that contains top-level code"). **Task 5's `InferenceActor`/CLI wiring must use `@main` async, not
a semaphore bridge, or it will reproduce this deadlock.**

## Task 3 test-snippet bug (plan text, not mlx-swift)

The plan's `DecodeMetricsTests` snippet calls
`XCTAssertEqual(m.decodeTokensPerSecond, 10.0, accuracy: 1e-9)` directly on the `Double?` property —
this does not compile (`XCTAssertEqual(_:_:accuracy:)` requires non-optional `FloatingPoint`).
Fixed by unwrapping first: `let rate = try! XCTUnwrap(m.decodeTokensPerSecond); XCTAssertEqual(rate,
10.0, accuracy: 1e-9)`.

## Net effect on Task 5 (`MLXDecoder`)

- `any LanguageModel` and `[KVCache]` types match the plan's pseudocode as written.
- `model(ids, cache: cache)` call syntax is unchanged.
- Loading needs `import MLXHuggingFace` + `#huggingFaceTokenizerLoader()` and the free function
  `loadModel(from:using:)`, not `LLMModelFactory.shared.loadContainer(...)`.
- No fundamental API gap found for a greedy single-owner-actor decode loop.

## Smoke test result (Task 2 Step 3 `api-check`)

Ran on llmbench against the local model dir `~/perf-work/models/Qwen3-30B-A3B-Instruct-2507-4bit`
(the HF cache path in the plan's prerequisites, `~/.cache/huggingface/hub/models--mlx-community--
Qwen3-30B-A3B-Instruct-2507-4bit`, only contains an empty `refs/main` — no snapshot/blobs; the real
16GB local copy lives under `~/perf-work/models/`, a flat directory of safetensors + config, which
is exactly the shape `loadModel(from: URL, using:)` expects — no HF cache layout needed).

```
logits.shape: [1, 1, 151936]
argMax token id: 271
decoded token: (empty — token 271 decodes to whitespace/newline under skipSpecialTokens)
```

151936 matches Qwen3's vocab size; seqLen=1 because the raw prompt `"Hello"` tokenizes to a single
BPE token with this tokenizer (no chat template applied) — expected shape and a sane, non-crashing
result. This proves the load -> encode -> forward -> argMax -> item -> decode path end-to-end.
