
export const ExtractFeature = async (model, text, debug = true) => {
  const output = await model.process(typeof text === 'string' ? [text] : text);
  if (debug) {
    console.log(
      `${output.tokensNum} tokens were processed in ${output.elapsed} seconds.`
    );
  }
  return output;
}