// const tf = require('@tensorflow/tfjs');
import * as tf from '@tensorflow/tfjs';

class EmbeddingHead extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.poolLayer = tf.layers.globalAveragePooling2d({
        name: config.name+'_pool'
      });
      this.bottleneck = tf.layers.batchNormalization({
        axis: -1,
        momentum: 0.1,
        epsilon: 1e-5
      });
    }

    build(inputShape) {
      this.poolLayer.build(inputShape);
      const poolOutShape = this.poolLayer.computeOutputShape(inputShape);
      this.bottleneck.build(poolOutShape);
      // this._trainableWeights = [
      //   ...this.bottleneck.trainableWeights,
      // ]
      // this._nonTrainableWeights = [
      //   ...this.bottleneck.nonTrainableWeights,
      // ]
      this.built = true;
    }

    computeOutputShape(inputShape) {
      let outputShape = this.poolLayer.computeOutputShape(inputShape);
      return outputShape;
    }

    call(inputs) {
      // TODO: do we need always the array tensor validation?
      const poolFeat = this.poolLayer.apply(inputs);
      const neckFeat = this.bottleneck.apply(poolFeat, { training: false });
      return neckFeat;
    }

    getConfig() {
      const config = super.getConfig();
      return config;
    }

    fromConfig(config) {
      return new EmbeddingHead(config);
    }
    static className = 'EmbeddingHead'; 
  }

tf.serialization.registerClass(EmbeddingHead);
// module.exports = EmbeddingHead;
export default EmbeddingHead;