
import { WasmTokenizer } from "@visheratin/tokenizers-node";
//import { WasmTokenizer } from "@visheratin/tokenizers-node";
//import { SessionParams } from "./common/index.js";
import fs from "fs";

export class Tokenizer {
  constructor(tokenizerPath) {
    this.tokenizerPath = tokenizerPath;
  }

  async init() {
    //console.log(init)
    //await init(SessionParams.tokenizersPath);
    let tokenizerData = null;
    if (this.tokenizerPath.startsWith("https://")) {
      const response = await fetch(this.tokenizerPath);
      tokenizerData = await response.json();
    } else {
      const response = fs.readFileSync(this.tokenizerPath);
      tokenizerData = JSON.parse(response);
      //console.log(tokenizerData);
    }
    tokenizerData["padding"] = null;
    this.instance = new WasmTokenizer(JSON.stringify(tokenizerData));
    /*
    const node_fetch_1 = __importDefault(require("node-fetch"));
    class Tokenizer {
      constructor(tokenizerPath) {
        this.tokenizerPath = tokenizerPath;
      }
      async init() {
        const response = await (0, node_fetch_1.default)(this.tokenizerPath);
        const tokenizerData = await response.json();
        tokenizerData["padding"] = null;
        this.instance = new tokenizers_node_1.WasmTokenizer(JSON.stringify(tokenizerData));
      }
      */

  }

  async decode(
    ids,
    skip_special_tokens
  ) {
    if (this.instance === undefined) {
      throw new Error("Tokenizer is not initialized");
    }
    const res = this.instance.decode(ids, skip_special_tokens);
    return res;
  }

  async encode(
    text,
    add_special_tokens
  ) {
    if (this.instance === undefined) {
      throw new Error("Tokenizer is not initialized");
    }
    return this.instance.encode(text, add_special_tokens);
  }
}
