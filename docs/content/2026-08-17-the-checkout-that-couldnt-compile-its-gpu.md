# The checkout that couldn't compile its own GPU

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Disciplined proof
over convenient claims; Meeting operators where they are

A benchmark nobody can run is not evidence, and an engine nobody can start is not a product. For a
stretch, fast-mlx was the second thing: a fresh `git clone` on a current Mac would build, launch,
accept a request, and then die the instant it touched the GPU — `Failed to load the default
metallib. library not found`. The most rigorous measurement loop in the world does not matter if
the operator never gets past the first token.

## The bug was two bugs

MLX runs its kernels from a compiled Metal library, `default.metallib`. Two independent facts
conspired to make sure it never existed on a fresh machine. First, Swift's package manager has no
Metal compilation step at all — `swift build` compiles the Swift and the C++, and silently skips
the `.metal` files, so the library is simply never produced. Second, on macOS 26 Apple moved the
Metal compiler out of the base toolchain into a separate, on-demand component; even a normal Xcode
build cannot compile those kernels until roughly 700 MB of Metal Toolchain is downloaded. A machine
with Xcode installed and the command-line tools set up still could not build the one file the
runtime cannot start without.

## The fix is the one Python already uses

The Python `mlx` package does not ask every user to compile Metal; it ships the compiled
`mlx.metallib` inside the wheel. fast-mlx now does the same. The library is built once from the
pinned mlx-swift — 3.6 MB, and portable across every Apple-silicon Mac because it is GPU-independent
AIR bytecode — committed to the repository, and colocated next to the built binary at launch, where
MLX's own search path finds it first. A one-command `scripts/serve.sh` builds the server, stages the
metallib, derives memory limits from the machine's RAM, and starts serving. Nothing to install, no
toolchain to download, no Xcode build.

The proof is the run that was failing before: a fresh `swift build` on a 24 GB laptop now serves a
dense Qwen3-8B and completes a full tool-calling round-trip — the model emits a tool call, the
result is fed back, and it answers — with no manual steps in between.

## Know before you download

Fixing the start is half of meeting an operator where they are; the other half is not making them
download 60 GB to discover a model never fit. fast-mlx already carried a careful capacity model —
weight bytes, KV-cache growth per architecture and quantization, transient prefill peaks, and the
largest context a machine can actually hold. A new sizer puts a friendly face on it: point it at
your Mac, and it reports which quantized builds fit and at what context, before anything is pulled.

It is deliberately honest about what it knows. A hand-measured box reports measured headroom; an
auto-detected machine reports a synthesized wired-memory limit, and every such row is flagged as a
modeled estimate rather than a guarantee. A sizer that quietly promises a fit it never measured
would be worse than no sizer at all.

Neither of these is a performance claim. They are the unglamorous work of turning a proof harness
into something an operator can actually pick up: a checkout that starts, and a number that tells the
truth about whether the next download is worth it.
