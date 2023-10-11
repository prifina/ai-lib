import { models } from "./models.js";
import { ModelType } from "./modelType.js";
import { FeatureExtractionModel } from "./featureExtractionModel.js";


export class TextModel {
  static create = async (
    id,
    proxy = false  // default was true
  ) => {

    for (const modelMetadata of models) {
      if (modelMetadata.id === id) {
        switch (modelMetadata.type) {
          case ModelType.FeatureExtraction: {
            //console.log("TEXT MODEL ", modelMetadata);
            const model = new FeatureExtractionModel(modelMetadata);
            const elapsed = await model.init(proxy);
            return {
              model: model,
              elapsed: elapsed,
            };
          }
          /* case ModelType.Seq2Seq: {
            const model = new Seq2SeqModel(modelMetadata);
            const elapsed = await model.init(proxy);
            return {
              model: model,
              elapsed: elapsed,
            };
          } */
        }
      }
    }
    throw Error("there is no text model with specified id");
  };
}
