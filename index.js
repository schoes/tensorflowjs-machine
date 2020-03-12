import * as tf from "@tensorflow/tfjs";

const featurees = tf.tensor([
  [124.5, 56.6],
  [124.5, 34.6],
  [122.5, 34.6],
  [124.5, 77.6]
]);
const label = tf.tensor([[[200], [230], [240], [345]]]);

const predictionPoint = tf.tensor([123, 35]);

featurees
    .sub(predictionPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(label, 1)
    .unstack().sort((a,b) => a.dataSync()[0] > b.dataSync()[0] ? 1:-1);
