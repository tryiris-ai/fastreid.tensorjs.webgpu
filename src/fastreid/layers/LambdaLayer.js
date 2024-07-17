// const tf = require('@tensorflow/tfjs');
import * as tf from '@tensorflow/tfjs';

class LambdaLayer extends tf.layers.Layer {
    constructor() {
      super({});
    }
  
    call(inputs) {
      return inputs;
    }
  
    computeOutputShape(inputShape) {
      return inputShape;
    }
  
    getClassName() {
      return 'LambdaLayer';
    }
  }

// module.exports = LambdaLayer;
export default LambdaLayer;