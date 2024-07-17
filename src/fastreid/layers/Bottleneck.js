import * as tf from '@tensorflow/tfjs';
import Conv2dCustom from './Conv2dCustom.js';
import LambdaLayer from './LambdaLayer.js';

class Bottleneck extends tf.layers.Layer {
    static expansion = 4;
    constructor(config, ...args) {
        config.stride = config.stride || 1;
        // All bottleneck names must be unique for serialization.
        super({ ...config, name: `Bottleneck_${config.layerName}_${config.bottleneckIdx}` }, ...args);
        this.inputShape = config.inputShape;
        this.layerName = config.layerName;
        this.bottleneckIdx = config.bottleneckIdx;
        this.stride = config.stride;  
        this.planes = config.planes;
        this.inplanes = config.inplanes;
        const prefix = `${this.layerName}_${this.bottleneckIdx}`;
        if (this.inputShape === undefined){
            this.conv1 = tf.layers.conv2d({
                name: `${prefix}_conv1`,
                filters: this.planes,
                kernelSize: 1,
                useBias: false
            });    
        }else{
            
            // It seems is not necessary to set the input shape in all the bottlenecks,
            // however is a framework requirement for sequential models so we 
            // need to confirm this.
            this.conv1 = tf.layers.conv2d({
                name: `${prefix}_conv1`,
                filters: this.planes,
                kernelSize: 1,
                useBias: false,
                inputShape: this.inputShape
            });
        }

        // weights initializers to test is loading weights is working.
        this.bn1 = tf.layers.batchNormalization({
            name: `${prefix}_bn1`,
            axis: -1,
            momentum: 0.1,
            epsilon: 1e-5,
            // gammaInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1}),
            // betaInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1}),
            // movingMeanInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1}),
            // movingVarianceInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1})
        });
        const conv2dName = `conv2d_${this.layerName}_${this.bottleneckIdx}`;
        
        this.conv2 = new Conv2dCustom({
            filters: this.planes,
            kernelSize: 3,
            strides: this.stride,
            paddingSize: 1,
            useBias: false,
            name: conv2dName
        });
        this.bn2 = tf.layers.batchNormalization({
            axis: -1,
            momentum: 0.1,
            epsilon: 1e-5
        });
        this.conv3 = tf.layers.conv2d({
            filters: this.planes * Bottleneck.expansion,
            kernelSize: 1,
            useBias: false
        });
        this.bn3 = tf.layers.batchNormalization({
            axis: -1,
            momentum: 0.1,
            epsilon: 1e-5
        });
        this.relu = tf.layers.reLU();
        this.se = new LambdaLayer();
        this.downsample = config.downsample;
        this.isTraining = false;
    }

    build(inputShape){
        this.conv1.build(inputShape);
        const bn1Shape = this.conv1.computeOutputShape(inputShape);
        this.bn1.build(bn1Shape);
        const conv2Shape = this.bn1.computeOutputShape(bn1Shape);
        this.conv2.build(conv2Shape);
        const bn2Shape = this.conv2.computeOutputShape(conv2Shape);
        this.bn2.build(bn2Shape);
        this.bn2.gamma.name = `${this.layerName}_${this.bottleneckIdx}_bn2_gamma`;
        this.bn2.gamma.originalName = `${this.layerName}_${this.bottleneckIdx}_bn2_gamma`;
        this.bn2.beta.name = `${this.layerName}_${this.bottleneckIdx}_bn2_beta`;
        this.bn2.beta.originalName = `${this.layerName}_${this.bottleneckIdx}_bn2_beta`;
        this.bn2.movingMean.name = `${this.layerName}_${this.bottleneckIdx}_bn2_movingMean`;
        this.bn2.movingMean.originalName = `${this.layerName}_${this.bottleneckIdx}_bn2_movingMean`;
        this.bn2.movingVariance.name = `${this.layerName}_${this.bottleneckIdx}_bn2_movingVariance`;
        this.bn2.movingVariance.originalName = `${this.layerName}_${this.bottleneckIdx}_bn2_movingVariance`;

        this.conv3.build(bn2Shape);
        this.conv3.kernel.name = `${this.layerName}_${this.bottleneckIdx}_conv3_kernel`;
        this.conv3.kernel.originalName = `${this.layerName}_${this.bottleneckIdx}_conv3_kernel`;

        this.bn3.build(this.conv3.computeOutputShape(bn2Shape));
        this.bn3.gamma.name = `${this.layerName}_${this.bottleneckIdx}_bn3_gamma`;
        this.bn3.gamma.originalName = `${this.layerName}_${this.bottleneckIdx}_bn3_gamma`;
        this.bn3.beta.name = `${this.layerName}_${this.bottleneckIdx}_bn3_beta`;
        this.bn3.beta.originalName = `${this.layerName}_${this.bottleneckIdx}_bn3_beta`;
        this.bn3.movingMean.name = `${this.layerName}_${this.bottleneckIdx}_bn3_movingMean`;
        this.bn3.movingMean.originalName = `${this.layerName}_${this.bottleneckIdx}_bn3_movingMean`;
        this.bn3.movingVariance.name = `${this.layerName}_${this.bottleneckIdx}_bn3_movingVariance`;
        this.bn3.movingVariance.originalName = `${this.layerName}_${this.bottleneckIdx}_bn3_movingVariance`;
        if (this.downsample !== undefined){
            const dsConv = this.downsample.layers[0];
            const dsBn = this.downsample.layers[1];
            dsConv.kernel.name = `${this.layerName}_${this.bottleneckIdx}_downsample_conv_kernel`;
            dsConv.kernel.originalName = `${this.layerName}_${this.bottleneckIdx}_downsample_conv_kernel`;
            dsBn.gamma.name = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_gamma`;
            dsBn.gamma.originalName = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_gamma`;
            dsBn.beta.name = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_beta`;
            dsBn.beta.originalName = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_beta`;
            dsBn.movingMean.name = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_movingMean`;
            dsBn.movingMean.originalName = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_movingMean`;
            dsBn.movingVariance.name = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_movingVariance`;
            dsBn.movingVariance.originalName = `${this.layerName}_${this.bottleneckIdx}_downsample_bn_movingVariance`;

            this._trainableWeights = [
                ...this.conv1.trainableWeights,
                ...this.bn1.trainableWeights,
                ...this.conv2.trainableWeights,
                ...this.bn2.trainableWeights,
                ...this.conv3.trainableWeights,
                ...this.bn3.trainableWeights,
                ...this.downsample.layers[0].trainableWeights,
                ...this.downsample.layers[1].trainableWeights
                
            ]
            this._nonTrainableWeights = [
                ...this.conv1.nonTrainableWeights,
                ...this.bn1.nonTrainableWeights,
                ...this.conv2.nonTrainableWeights,
                ...this.bn2.nonTrainableWeights,
                ...this.conv3.nonTrainableWeights,
                ...this.bn3.nonTrainableWeights,
                ...this.downsample.layers[0].nonTrainableWeights,
                ...this.downsample.layers[1].nonTrainableWeights
            ]
        }else{
            // Register the weights of the conv1 and bn1 layers
            this._trainableWeights = [
                ...this.conv1.trainableWeights,
                ...this.bn1.trainableWeights,
                ...this.conv2.trainableWeights,
                ...this.bn2.trainableWeights,
                ...this.conv3.trainableWeights,
                ...this.bn3.trainableWeights
            ]
            this._nonTrainableWeights = [
                ...this.conv1.nonTrainableWeights,
                ...this.bn1.nonTrainableWeights,
                ...this.conv2.nonTrainableWeights,
                ...this.bn2.nonTrainableWeights,
                ...this.conv3.nonTrainableWeights,
                ...this.bn3.nonTrainableWeights
            ]
        }

        this.built = true;
    }

    computeOutputShape(inputShape){
        let residualShape = inputShape;
        let shape = this.conv1.computeOutputShape(inputShape);
        shape = this.conv2.computeOutputShape(shape);
        shape = this.conv3.computeOutputShape(shape);
        shape = this.se.computeOutputShape(shape);
        if (this.downsample !== undefined){
            for (const layer of this.downsample.layers) {
                residualShape = layer.computeOutputShape(residualShape);
            }
        }
        if (!tf.util.arraysEqual(shape, residualShape)) {
            throw new Error('the residual shape is not equal to the output shape');
        }
        return shape;
    }

    call(x){
        let residual = Array.isArray(x) ? x[0] : x;
        let out = this.conv1.apply(x);
        
        // Inference mode
        out = this.bn1.apply(out, { training: this.isTraining }); 
        out = this.relu.apply(out);
        out = this.conv2.call(out);
        out = this.bn2.apply(out, { training: this.isTraining });
        out = this.relu.apply(out);
        out = this.conv3.apply(out);
        out = this.bn3.apply(out, { training: this.isTraining });
        out = this.se.call(out);
        if (this.downsample !== undefined) {
          residual = this.downsample.call(x)[0];
        }
        out = tf.add(out, residual);
        out = this.relu.apply(out);

        return out;
    }

    migrateBottleneckWeights(){
        let file_name = `./kerasWeights/stage2_${this.layerName}_weights.json`;
        let weightsDict = this.loadWeights(file_name);
        this.migrateWeightsConvBnBlock(weightsDict, this.conv1, this.bn1, '1');
        // this.migrateWeightsConvBlock(weightsDict, this.conv2, '2');
        this.migrateWeightsConvBnBlock(weightsDict, this.conv2, this.bn2, '2');
        this.migrateWeightsConvBnBlock(weightsDict, this.conv3, this.bn3, '3');
        // // let values = this.conv1.getWeights()[0].arraySync();
        // // console.log("conv1 weights: ", values);
        this.migrateDownsample(weightsDict);
    }

    migrateWeightsConvBlock(weightsDict, conv, cbIdx){
        let bottleneckKey = `bottleneck${this.bottleneckIdx}`;
        let convKey = `backbone.${this.layerName}.${this.bottleneckIdx}.conv${cbIdx}`;
        let convWeights = weightsDict[this.layerName][bottleneckKey][convKey].map((w) => tf.tensor(w));
        conv.setWeights(convWeights);
    }

    migrateWeightsConvBnBlock(weightsDict, conv, bn, cbIdx){
        // Format of the keys to access the weights:
        // dict[layer1][bottleneck0][backbone.layer1.0.conv1]
        let bottleneckKey = `bottleneck${this.bottleneckIdx}`;
        let convKey = `backbone.${this.layerName}.${this.bottleneckIdx}.conv${cbIdx}`;
        let bnKey = `backbone.${this.layerName}.${this.bottleneckIdx}.bn${cbIdx}`;
        let convWeights = weightsDict[this.layerName][bottleneckKey][convKey].map((w) => tf.tensor(w));
        let bnWeights = weightsDict[this.layerName][bottleneckKey][bnKey].map((w) => tf.tensor(w));
        conv.setWeights(convWeights);
        bn.setWeights(bnWeights);
    }

    migrateDownsample(weightsDict){
        if (this.downsample !== undefined){
            let bottleneckKey = `bottleneck${this.bottleneckIdx}`;
            let downsampleConvKey = `backbone.${this.layerName}.${this.bottleneckIdx}.downsample.conv`;
            let downsampleBnKey = `backbone.${this.layerName}.${this.bottleneckIdx}.downsample.bn`;
            let downsampleConvWeights = weightsDict[this.layerName][bottleneckKey][downsampleConvKey].map((w) => tf.tensor(w));
            let downsampleBnWeights = weightsDict[this.layerName][bottleneckKey][downsampleBnKey].map((w) => tf.tensor(w));
            // conv layer
            this.downsample.layers[0].setWeights(downsampleConvWeights);
            // bn layer
            this.downsample.layers[1].setWeights(downsampleBnWeights);
        }
    }

    // loadWeights(jsonFilePath) {
    //     try {
    //         const data = fs.readFileSync(jsonFilePath, 'utf8');
    //         return JSON.parse(data);
    //     } catch (err) {
    //         console.error('Error reading file:', err);
    //         throw err;
    //     }
    // }

    getConfig() {
        const baseConfig = super.getConfig();
        const downsampleConfig = this.downsample !== undefined ? this.downsample.getConfig() : undefined;
        return {
            ...baseConfig,
            inputShape: this.inputShape,
            layerName: this.layerName,
            bottleneckIdx: this.bottleneckIdx,
            stride: this.stride,
            planes: this.planes,
            inplanes: this.inplanes,
            downsampleConfig: downsampleConfig
        };
    }

    static fromConfig(cls, config) {
        const {
            inputShape,
            layerName,
            bottleneckIdx,
            stride,
            planes,
            inplanes,
            downsampleConfig,
            ...otherConfig
        } = config;

        let downsample = undefined;
        // if (stride !== 1 || inplanes !== planes * Bottleneck.expansion) {
        if (downsampleConfig !== null) {
            const downsampleLayers = [

              // In tensorflowjs the sequence layers need to know the input shape
              // When using this layer as the first layer in a model, provide the 
              // keyword argument inputShape (Array of integers, does not include 
              //the sample axis), e.g. inputShape=[128, 128, 3] for 128x128 RGB 
              // pictures in dataFormat='channelsLast'.
              tf.layers.conv2d({
                filters: planes * Bottleneck.expansion,
                kernelSize: 1,
                strides: stride,
                useBias: false,
                inputShape: inputShape
              }),
              tf.layers.batchNormalization({
                axis: -1,
                momentum: 0.1,
                epsilon: 1e-5,
                gammaInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1}),
                betaInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1}),
                movingMeanInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1}),
                movingVarianceInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1})
              })
            ];
            downsample = tf.sequential({ 
              layers: downsampleLayers
            });
          }

        const layer = new cls({
            layerName,
            bottleneckIdx,
            inputShape,
            planes,
            inplanes,
            stride,
            downsample,
            ...otherConfig
        });

        return layer;
    }
    
    static className = 'Bottleneck';
}

tf.serialization.registerClass(Bottleneck);
export default Bottleneck;