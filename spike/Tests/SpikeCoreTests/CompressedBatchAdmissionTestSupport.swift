import Foundation

@testable import HarnessCore

func makeCompressedBatchSourceSnapshot(
    modelType: String = "qwen3",
    architecture: String = "Qwen3ForCausalLM",
    layerCount: Int = 1,
    queryHeadCount: Int = 4,
    keyValueHeadCount: Int = 2,
    headDimension: Int = 128,
    maxPositionEmbeddings: Int = 64,
    checkpointManifestHash: String = "0123456789abcdef",
    checkpointContentSHA256: String = String(repeating: "d", count: 64),
    tokenizerSHA256: String = String(repeating: "c", count: 64)
) throws -> CompressedKVAttentionRuntimeSourceSnapshot {
    let hiddenSize = queryHeadCount * headDimension
    let config = Data("""
    {
      "model_type":"\(modelType)",
      "architectures":["\(architecture)"],
      "hidden_size":\(hiddenSize),
      "num_hidden_layers":\(layerCount),
      "num_attention_heads":\(queryHeadCount),
      "num_key_value_heads":\(keyValueHeadCount),
      "head_dim":\(headDimension),
      "max_position_embeddings":\(maxPositionEmbeddings),
      "vocab_size":2048,
      "torch_dtype":"float16",
      "use_sliding_window":false
    }
    """.utf8)
    return try .load(
        exactModelConfigData: config,
        checkpointManifestHash: checkpointManifestHash,
        checkpointContentSHA256: checkpointContentSHA256,
        tokenizerSHA256: tokenizerSHA256)
}

func makeCompressedBatchAdmission(
    modelType: String = "qwen3",
    architecture: String = "Qwen3ForCausalLM",
    layerCount: Int = 1,
    queryHeadCount: Int = 4,
    keyValueHeadCount: Int = 2,
    headDimension: Int = 128,
    maxPositionEmbeddings: Int = 64,
    checkpointManifestHash: String = "0123456789abcdef",
    checkpointContentSHA256: String = String(repeating: "d", count: 64),
    tokenizerSHA256: String = String(repeating: "c", count: 64)
) throws -> CompressedKVAttentionRuntimeAdmission {
    try CompressedKVAttentionRuntimeAdmission.load(
        sourceSnapshot: makeCompressedBatchSourceSnapshot(
            modelType: modelType,
            architecture: architecture,
            layerCount: layerCount,
            queryHeadCount: queryHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            maxPositionEmbeddings: maxPositionEmbeddings,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256))
}
