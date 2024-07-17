import * as tf from '@tensorflow/tfjs';

class SimpleConv2D extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.filters = config.filters;
      this.kernelSize = config.kernelSize;
      this.strides = config.strides || [1, 1];
      this.padding = config.padding || 'valid';
      this.useBias = config.useBias !== undefined ? config.useBias : true;
    }
  
    build(inputShape) {
      this.conv2d = tf.layers.conv2d({
        filters: this.filters,
        kernelSize: this.kernelSize,
        strides: this.strides,
        padding: this.padding,
        useBias: this.useBias,
        kernelInitializer: 'glorotUniform',
        biasInitializer: 'zeros'
      });
  
      this.conv2d.build(inputShape);
      this.built = true;
    }
  
    call(inputs) {
      return this.conv2d.apply(inputs);
    }
  
    getConfig() {
      const baseConfig = super.getConfig();
      return {
        ...baseConfig,
        filters: this.filters,
        kernelSize: this.kernelSize,
        strides: this.strides,
        padding: this.padding,
        useBias: this.useBias
      };
    }
  
    getWeights() {
      return this.conv2d.getWeights();
    }
  
    setWeights(weights) {
      this.conv2d.setWeights(weights);
    }
  }

  export default SimpleConv2D;
