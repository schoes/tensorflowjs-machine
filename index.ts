import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
async function doMagic() {
// Load the model.
const model = await cocoSsd.load();
// Classify the image.
// const predictions = await model.detect(img);
console.log('Predictions: ');
// console.log(predictions);
}

doMagic();
