import { ModelType } from "./modelType.js";

export const ListTextModels = (
  tags,
  type
) => {
  if (!tags && !type) {
    return models;
  }
  return models.filter((model) => {
    let tagCheck = true;
    if (typeof model.tags === 'object' && tags && tags.length > 0) {
      tagCheck = tags.every((tag) => model.tags.includes(tag));
    }
    let typeCheck = true;
    if (type) {
      typeCheck = model.type === type;
    }
    return tagCheck && typeCheck;
  });
};

export const models = [

  {
    id: "mini-lm-v2-quant",
    title: "Quantized mini model for sentence embeddings",
    description: "",
    memEstimateMB: 100,
    type: ModelType.FeatureExtraction,
    sizeMB: 15,
    modelPaths: new Map([
      [
        "encoder",
        //"models/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx",
        "https://prifina-ai-models.s3.eu-west-1.amazonaws.com/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx"
        //"https://web-ai-models.org/text/feature-extraction/miniLM-v2/model-quant.onnx.gz",
      ],
    ]),
    outputNames: new Map([["encoder", "last_hidden_state"]]),
    tokenizerPath:
      // "models/Xenova/all-MiniLM-L6-v2/tokenizer.json",
      //"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
      "https://prifina-ai-models.s3.eu-west-1.amazonaws.com/Xenova/all-MiniLM-L6-v2/tokenizer.json",
    tokenizerParams: {
      bosTokenID: 0,
      eosTokenID: 1,
      padTokenID: 0,
    },
    tags: ["feature-extraction"],
    referenceURL:
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
  },
];