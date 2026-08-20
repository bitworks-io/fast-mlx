# The tool call we could prove without the model

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Rapid research
integration — the flywheel; Disciplined proof over convenient claims

Tool calling looks like a feature you bolt onto a chat server: accept a `tools` array, let the
model emit a function call, hand it back in OpenAI's shape. Most of that machinery already existed
in fast-mlx before this change — the vendored MLX tokenizer stack ships a Hermes-format tool-call
parser, and Qwen3's own chat template already knows how to render tools and tool results. The work
was not inventing those parts. It was making the serving contract exact, and proving the parts fit
before a single GPU token was generated.

## Arguments are a string and an object at once

OpenAI's Chat Completions API carries a tool call's `arguments` as a JSON **string** — the client
runs `JSON.parse` on it. Qwen3's chat template, when it re-renders a prior tool call back into the
prompt, expects `arguments` as a JSON **object** and serializes it itself. The same field is a
string on the wire and an object in the template. Get that seam wrong in either direction and you
either double-encode the arguments (the model sees `"{\"query\":\"…\"}"` instead of
`{"query":"…"}`) or hand the client an object it cannot parse.

The server now serializes parsed calls to a string on the way out, and parses the client's string
back to an object on the way in. That asymmetry is the single most consequential detail in the
whole feature, and it is invisible until a second conversation turn replays a tool result.

## Proving the prompt without a model

fast-mlx runs on Apple Silicon through Metal, and the reliable way to verify a serving change is to
serve. But prompt construction — turning a multi-turn history of tool calls and tool results into
the exact token sequence Qwen3 expects — is pure tokenizer work. It needs the chat template and the
vocabulary, not the model weights and not the GPU.

So the highest-risk part, the multi-turn re-render, is verified against a real Qwen3 tokenizer on
CPU. The test decodes the rendered prompt back to text and asserts the actual structure: a `<tools>`
block carrying the function schema, the prior call rendered as `<tool_call>\n{"name": …,
"arguments": {…}}\n</tool_call>` with real JSON (not a double-encoded string), the tool result
wrapped in `<tool_response>` inside a user turn exactly as Qwen's template dictates, and — when
thinking is disabled — the empty `<think></think>` block the template pre-fills. None of that
requires a forward pass, so it runs in CI on any machine.

## Where we stopped claiming

The contract, the tool-call parsing, the multi-turn rendering, the streaming deltas, and the
`tool_choice` resolution are covered by tests that run without a model. What those tests cannot
show is that a live Qwen3, generating freely, emits a tool call this parser accepts on the first
try. That is a different claim, and it belongs to a run on real hardware, not to a unit test. The
capability is recorded as implemented and contract-verified; end-to-end behavior on a served model
is the next gate, not a settled result.

## The part that fits the flywheel

There is one detail that only a project built around quantified testing would add: tool-call
reliability is measurable, and it should be measured per build. A 4-bit quantization that answers
chat well may trigger the wrong tool, invent an argument key, or emit JSON that needs repair more
often than an 8-bit build — and today an operator choosing a quantization has no number for that.
fast-mlx now has the scoring core for one: trigger rate, name accuracy, argument-schema validity,
and JSON-repair rate over a small tool-eval corpus, as a pure function that any build can be run
through. Turning that into a per-quantization figure an operator reads before they deploy is the
work the scoring core exists to enable.

The model already knew how to call tools. The server had to learn to listen exactly, prove it could
before it claimed it had, and leave a way to measure how well each build actually does it.
