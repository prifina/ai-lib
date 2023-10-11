
import { Tokenizer } from "./tokenizer.js";

export const loadTokenizer = async (
  tokenizerPath,
  proxy = false //default was true
) => {
  /*
  if (proxy && typeof document !== "undefined") {
    const worker = new Worker(
      new URL("./tokenizer.worker.js", import.meta.url),
      {
        type: "module",
      }
    );
    const Channel = Comlink.wrap < typeof Tokenizer > (worker);
    const tokenizer: Comlink.Remote<Tokenizer> = await new Channel(
      tokenizerPath
    );
    await tokenizer.init();
    // @ts-ignore
    return tokenizer;
  }
  */
  const tokenizer = new Tokenizer(tokenizerPath);
  await tokenizer.init();
  return tokenizer;
};
