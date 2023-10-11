/*
import localforage from "localforage";
import * as ort from "onnxruntime-web";
import * as pako from "pako";
import { SessionParameters } from "../common.js";
*/

import * as ort from "onnxruntime-node";
import fs from "fs";

import { SessionParams } from "./common/index.js";
/* 

const clearCache = async () => {
  // await localforage.clear();
}; */

export class Session {
  ortSession;
  cacheSize;
  params;

  constructor(params) {
    this.params = params;
    const cacheSize = params.cacheSizeMB * 1e6;
    /*
    localforage.config({
      name: "Web-AI",
      version: 1.0,
      driver: localforage.INDEXEDDB,
      storeName: "model_storage",
    });
    */
    this.cacheSize = cacheSize;
  }

  async init(modelPath) {
    ort.env.wasm.numThreads = this.params.numThreads;
    ort.env.wasm.wasmPaths = this.params.wasmRoot;
    let modelData = new ArrayBuffer(0);
    try {
      /*
      const cachedData = await localforage.getItem(modelPath);
      if (cachedData !== null) {
        modelData = cachedData;
      } else {
        modelData = await this.fetchData(modelPath);
      } */
      modelData = await this.fetchData(modelPath);
    } catch (err) {
      console.error("unable to load the data from cache");
      console.error(err);
      modelData = await this.fetchData(modelPath);
    }
    const session = await ort.InferenceSession.create(modelData, {
      executionProviders: this.params.executionProviders,
      graphOptimizationLevel: "all",
      executionMode: "parallel",
    });
    this.ortSession = session;
  }

  async fetchData(modelPath) {
    const extension = modelPath.split(".").pop();
    let modelData = undefined;
    if (modelPath.startsWith("https://")) {

      modelData = await fetch(modelPath).then((resp) => resp.arrayBuffer());
      if (extension === "gz") {
        // modelData = pako.inflate(modelData);
      }
    } else {
      modelData = fs.readFileSync(modelPath);
    }
    /*
    if (modelData.byteLength > this.cacheSize) {
      console.warn("the model is too large to be cached");
    } else {
      await this.validateCache(modelData);
      localforage.setItem(modelPath, modelData);
    }
    */
    return modelData;
  }
  /*
    async validateCache(modelData) {
      try {
        const cacheKeys = await localforage.keys();
        let cacheSize = 0;
        const cacheItemSizes = new Map();
        for (const key of cacheKeys) {
          const data = (await localforage.getItem(key));
          cacheSize += data.byteLength;
          cacheItemSizes.set(key, data.byteLength);
        }
        let newCacheSize = cacheSize + modelData.byteLength;
        while (newCacheSize > this.cacheSize) {
          const [key, size] = cacheItemSizes.entries().next().value;
          cacheItemSizes.delete(key);
          newCacheSize -= size;
          await localforage.removeItem(key);
        }
      } catch (err) {
        console.error("unable to validate the cache");
        console.error(err);
      }
    }
  */
  async run(input) {
    if (!this.ortSession) {
      throw Error(
        "the session is not initialized. Call `init()` method first."
      );
    }
    return await this.ortSession.run(input);
  }

  inputNames() {
    if (!this.ortSession) {
      throw Error(
        "the session is not initialized. Call `init()` method first."
      );
    }
    return this.ortSession.inputNames;
  }

  outputNames() {
    if (!this.ortSession) {
      throw Error(
        "the session is not initialized. Call `init()` method first."
      );
    }
    return this.ortSession.outputNames;
  }
}
