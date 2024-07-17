import * as tf from '@tensorflow/tfjs';

class Conv2dCustom extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.paddingSize = config.paddingSize || 0;
        this.convConfig = {
            filters: config.filters,
            kernelSize: config.kernelSize,
            strides: config.strides || [2, 2],
            padding: 'valid',
            useBias: config.useBias || false,
            name: config.name,
        };
    }

    build(inputShape) {
        // It's necessary instantiate the conv2d layer here to be able to set the kernel name
        this.conv2d = tf.layers.conv2d({
            ...this.convConfig
        });
        this.conv2d.build(inputShape);
        this.conv2d.kernel.name = `${this.conv2d.name}/kernel`;
        this.conv2d.kernel.originalName = `${this.conv2d.name}/kernel`;

        // Super important!, we're registering the conv2d weights as the layer 
        // custom weights, this way the model will be able to save and load the
        // weights correctly.
        this._trainableWeights = this.conv2d.trainableWeights;
        this._nonTrainableWeights = this.conv2d.nonTrainableWeights;
        this.built = true;
    }

    call(inputs) {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        const [batchSize, height, width, channels] = input.shape;
        const padValue = tf.zeros([batchSize, this.paddingSize, width, channels]);
        const padded1 = tf.concat([padValue, input, padValue], 1);
        const padValue2 = tf.zeros([batchSize, height + 2 * this.paddingSize, this.paddingSize, channels]);
        const padded2 = tf.concat([padValue2, padded1, padValue2], 2);
        // let value = padded2.arraySync();
        return this.conv2d.apply(padded2);
      }

    computeOutputShape(inputShape) {
        const paddedShape = [
            inputShape[0],
            inputShape[1] + 2 * this.paddingSize,
            inputShape[2] + 2 * this.paddingSize,
            inputShape[3]
        ];
        return this.conv2d.computeOutputShape(paddedShape);
    }

    getWeights() {
        return this.conv2d.getWeights();
    }

    setWeights(weights) {
        return this.conv2d.setWeights(weights);
    }

    getConfig() {
        const baseConfig = super.getConfig();
        return {
          ...baseConfig,
            paddingSize: this.paddingSize,
          filters: this.convConfig.filters,
          kernelSize: this.convConfig.kernelSize,
          strides: this.convConfig.strides,
          padding: this.convConfig.padding,
          useBias: this.convConfig.useBias,
            name: this.convConfig.name
        };
      }

    static fromConfig(cls, config) {
        return new cls({
            paddingSize: config.paddingSize,
            filters: config.filters,
            kernelSize: config.kernelSize,
            strides: config.strides,
            useBias: config.useBias,
            name: config.name
        });
    }
      static className = 'Conv2dCustom';
}

tf.serialization.registerClass(Conv2dCustom);
export default Conv2dCustom;
