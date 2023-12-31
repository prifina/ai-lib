import * as ort from "onnxruntime-common";

export const prepareTextTensors = async (
  inputs,
  tokenizer,
  addSpecialTokens,
  padTokenID,
  bosTokenID
) => {
  if (!tokenizer) {
    throw Error("the tokenizer is not initialized");
  }
  let maxLen = 0;
  const inputIDs = new Array(inputs.length);
  const attentionMasks = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i++) {
    const tokens = await tokenizer.encode(inputs[i], addSpecialTokens);
    let len = tokens.length;
    if (bosTokenID) {
      len++;
    }
    inputIDs[i] = new Array(len);
    let offset = 0;
    if (bosTokenID) {
      inputIDs[i][0] = bosTokenID;
      offset = 1;
    }
    for (let j = offset; j < len; j++) {
      if (bosTokenID) {
        inputIDs[i][j] = tokens[j - 1];
      } else {
        inputIDs[i][j] = tokens[j];
      }
    }
    attentionMasks[i] = new Array(len).fill(1);
    if (len > maxLen) {
      maxLen = len;
    }
  }
  for (let i = 0; i < inputs.length; i++) {
    while (inputIDs[i].length < maxLen) {
      inputIDs[i].push(padTokenID);
      attentionMasks[i].push(0);
    }
  }
  const inputIDsData = new BigInt64Array(inputs.length * maxLen);
  for (let i = 0; i < inputs.length; i++) {
    for (let j = 0; j < maxLen; j++) {
      inputIDsData[i * maxLen + j] = BigInt(inputIDs[i][j]);
    }
  }
  const attentionMasksData = new BigInt64Array(inputs.length * maxLen);
  for (let i = 0; i < inputs.length; i++) {
    for (let j = 0; j < maxLen; j++) {
      attentionMasksData[i * maxLen + j] = BigInt(attentionMasks[i][j]);
    }
  }
  const inputIDsTensor = new ort.Tensor("int64", inputIDsData, [
    inputs.length,
    maxLen,
  ]);
  const attentionMaskTensor = new ort.Tensor("int64", attentionMasksData, [
    inputs.length,
    maxLen,
  ]);
  return [inputIDsTensor, attentionMaskTensor];
};