import * as ort from "onnxruntime-node";
import { Session } from "./session.js";
import { SessionParams } from "./common/index.js";

export const createSession = async (
  modelPath,
  proxy
) => {
  /*
  if (proxy && typeof document !== "undefined") {
    ort.env.wasm.proxy = true;
    const worker = new Worker(new URL("./session.worker.js", import.meta.url), {
      type: "module",
    });
    const Channel = Comlink.wrap<typeof Session>(worker);
    const session: Comlink.Remote<Session> = await new Channel(SessionParams);
    await session.init(modelPath);
    // @ts-ignore
    return session;
  } else {
    */
  //console.log(ort);
  ort.env.wasm.proxy = false;

  const session = new Session(SessionParams);
  await session.init(modelPath);
  return session;

  //}
};
