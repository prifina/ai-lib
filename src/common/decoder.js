export class Decoder {

  constructor(session, outputName, type) {
    this.session = session;
    this.outputName = outputName;
    this.type = type;
  }

  process = async (
    encoderOutput,
    decoderInput,
    decoderAttention,
    encoderAttention
  ) => {
    if (!this.session) {
      throw new Error("Session is not initialized");
    }
    const decoderFeeds = {
      input_ids: decoderInput,
      encoder_hidden_states: encoderOutput,
    };
    const inputNames = await this.session.inputNames();
    if (inputNames.includes("attention_mask")) {
      if (!decoderAttention) {
        throw new Error("Decoder attention mask is not provided");
      }
      decoderFeeds["attention_mask"] = decoderAttention;
    }
    if (inputNames.includes("encoder_attention_mask")) {
      if (!encoderAttention) {
        throw new Error("Encoder attention mask is not provided");
      }
      decoderFeeds["encoder_attention_mask"] = encoderAttention;
    }
    const output = await this.session.run(decoderFeeds);
    const result = output[this.outputName];
    return result;
  };
}