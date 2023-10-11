import { TextModel, ExtractFeature } from "./src/index.js";



async function main() {
  const modelResult = await TextModel.create("mini-lm-v2-quant", false);
  console.log(modelResult);
  const model = modelResult.model;

  const text = "Hello, world";
  const vectorData = await ExtractFeature(model, text);

  console.log(vectorData);
}

main();