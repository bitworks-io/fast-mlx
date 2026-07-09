import Foundation
import MLX
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers
import SpikeCore

func apiCheck(modelPath: String) async {
    do {
        let ctx = try await loadModel(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )
        let promptTokens = ctx.tokenizer.encode(text: "Hello")
        let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        let cache = ctx.model.newCache(parameters: nil)
        let logits = ctx.model(ids, cache: cache)
        eval(logits)
        print("logits.shape: \(logits.shape)")
        let last = logits[0..., -1, 0...]
        let next = argMax(last, axis: -1)
        let tokenId = next.item(Int.self)
        let decoded = ctx.tokenizer.decode(tokenIds: [tokenId], skipSpecialTokens: true)
        print("argMax token id: \(tokenId)")
        print("decoded token: \(decoded)")
    } catch {
        print("api-check FAILED: \(error)")
        exit(1)
    }
}

@main
struct SpikeCLI {
    static func main() async {
        let arguments = CommandLine.arguments
        if arguments.count >= 2, arguments[1] == "api-check" {
            var modelPath = ""
            var i = 2
            while i < arguments.count {
                if arguments[i] == "--model", i + 1 < arguments.count {
                    modelPath = arguments[i + 1]
                    i += 2
                } else {
                    i += 1
                }
            }
            guard !modelPath.isEmpty else {
                print("usage: spike-cli api-check --model <PATH>")
                exit(1)
            }
            await apiCheck(modelPath: modelPath)
        } else {
            print("spike ok: \(Spike.ok)")
        }
    }
}
