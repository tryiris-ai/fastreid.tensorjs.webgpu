import * as tf from '@tensorflow/tfjs';
import Conv2dCustom from './layers/Conv2dCustom.js';
import Bottleneck from './layers/Bottleneck.js';
import EmbeddingHead from './layers/EmbeddingHead.js';

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';  
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

class ResNet50Bot {
    constructor(config) {
        this.inputShape = config.inputShape || [256, 128, 3];
        this.inplanes = 64;

        // if not build mode return, this is useful for loading the model from
        // bin files.
        if (!config.buildModel) {
            return;
        }
        this.conv1 = new Conv2dCustom({
            filters: 64,
            kernelSize: 7,
            paddingSize: 3,
            strides: [2, 2],
            useBias: false,
            name: 'custom_conv2d'
        })

        this.bn1 = tf.layers.batchNormalization({
            axis: -1,
            momentum: 0.1,
            epsilon: 1e-5
        });
        this.relu = tf.layers.activation({activation: 'relu', name: 'relu'});
        this.maxpool = tf.layers.maxPooling2d({
            poolSize: [3, 3], 
            strides: [2, 2], 
            padding: 'same'
        });
        

        const numBlocksPerStage = [3, 4, 6, 3];
        this.layer1 = this._makeLayer(
            'layer1', 
            64, 
            numBlocksPerStage[0], 
            // Need to pass the downsample input shape without the batch size.
            // TODO: Change this to a more dynamic way to get the input shape with
            // build method.
            [64, 32, 64],
            1
        );
        this.layer2 = this._makeLayer(
            'layer2', 
            128, 
            numBlocksPerStage[1], 
            [64, 32, 256],
            2
        );
        this.layer3 = this._makeLayer(
            'layer3',
            256,
            numBlocksPerStage[2],
            [32, 16, 512],
            2
        );
        this.layer4 = this._makeLayer(
            'layer4',
            512,
            numBlocksPerStage[3],
            [16, 8, 1024],
            1,
        );

        // this.avgpool = tf.layers.globalAveragePooling2d({name: 'avgpool'});
        this.heads = new EmbeddingHead({
            name: 'embedding_head',
        });

        // Build the model
        this.buildModel();
    }

    buildModel() {
        const input = tf.input({shape: this.inputShape});
        let x = input;
        
        // Can't use call even if it's a custom layer because the apply in high
        // level connects the layers during the model building, internally apply
        // calls the call method of the layer when is necessary. Using call could
        // break the symbolic model building.
        x = this.conv1.apply(x);
        x = this.bn1.apply(x);
        x = this.relu.apply(x);
        x = this.maxpool.apply(x);

        for (let i = 0; i < this.layer1.layers.length; i++) {
            x = this.layer1.layers[i].apply(x);
          }
          for (let i = 0; i < this.layer2.layers.length; i++) {
            x = this.layer2.layers[i].apply(x);
          }
          for (let i = 0; i < this.layer3.layers.length; i++) {
            x = this.layer3.layers[i].apply(x);
          }
          for (let i = 0; i < this.layer4.layers.length; i++) {
            x = this.layer4.layers[i].apply(x);
          }
        // const output = this.avgpool.apply(x);
        const output = this.heads.apply(x);
        this.model = tf.model({inputs: input, outputs: output});
    }

    _makeLayer(layerName, planes, blocks, dsInputShape, stride = 1) {
      let downsample = null;
      const expansion = Bottleneck.expansion;
      if (stride !== 1 || this.inplanes !== planes * expansion) {
        const downsampleLayers = [
          // In tensorflowjs the sequence layers need to know the input shape
          // When using this layer as the first layer in a model, provide the 
          // keyword argument inputShape (Array of integers, does not include 
          //the sample axis), e.g. inputShape=[128, 128, 3] for 128x128 RGB 
          // pictures in dataFormat='channelsLast'.
          tf.layers.conv2d({
            filters: planes * expansion,
            kernelSize: 1,
            strides: stride,
            useBias: false,
            inputShape: dsInputShape
          }),
          tf.layers.batchNormalization({
            axis: -1,
            momentum: 0.1,
            epsilon: 1e-5
          })
        ];
        downsample = tf.sequential({ 
          layers: downsampleLayers
        });
      }
  
      const layersList = [];
      layersList.push(
        new Bottleneck({
          layerName: layerName,
          bottleneckIdx: 0,
          inputShape: dsInputShape,
          planes,
          inplanes: this.inplanes,
          stride,
          downsample,
        })
      );
      this.inplanes = planes * expansion;
      for (let i = 1; i < blocks; i++) {
        layersList.push(
          new Bottleneck({
            layerName: layerName,
            bottleneckIdx: i,
            planes,
            inplanes: this.inplanes,
          })
        );
      }
      
      // This instruction call the build method of each layer automatically
      return tf.sequential({ 
        layers: layersList
       });
    }

    async loadStg2Layer1Weights(){
        // this.layer1.migrateBottleneckWeights();
        for (let i = 0; i < this.layer1.layers.length; i++) {
          this.layer1.layers[i].migrateBottleneckWeights();
        }
        for (let i = 0; i < this.layer2.layers.length; i++) {
          this.layer2.layers[i].migrateBottleneckWeights();
        }
        for (let i = 0; i < this.layer3.layers.length; i++) {
          this.layer3.layers[i].migrateBottleneckWeights();
        }
        for (let i = 0; i < this.layer4.layers.length; i++) {
          this.layer4.layers[i].migrateBottleneckWeights();
        }
    }

    predict(inputs) {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        return this.model.predict(input);
    }

    summary() {
        return this.model.summary();
    }

    async loadWeights(jsonFilePath) {
        try {
          const data = await fs.readFile(jsonFilePath, 'utf8');
          return JSON.parse(data);
        } catch (err) {
          console.error('Error reading file:', err);
          throw err;
        }
      }

    async loadStg1Weights() {
        const layerName = 'conv1';
        const weightsDict = await this.loadWeights('./kerasWeights/stage1_weights.json');
        const layerWeights = weightsDict[layerName].map(w => tf.tensor(w));
        this.conv1.setWeights(layerWeights);
      }

      async loadStg1WeightsBn(){
        const layerName = 'batch_normalization';
        // This layer has 4 weights: gamma, beta, moving mean, moving variance
        const weightsDict = await this.loadWeights('./kerasWeights/stage1_bn_weights.json');
        const layerWeights = weightsDict[layerName].map(w => tf.tensor(w));
        this.bn1.setWeights(layerWeights);
      }

    static async load(baseUrl = 'http://localhost:8080/models/resnet50') {
      const modelUrl = `${baseUrl}/model.json`;
      const model = await tf.loadLayersModel(modelUrl);
    
      const resnet = new ResNet50Bot({
        inputShape: model.inputs[0].shape.slice(1),
        buildModel: false
      });
      resnet.model = model;
      return resnet;
    }

    static async loadAndPreprocessImage(imageUrl) {
      const pixelMean = tf.tensor([123.675, 116.28, 103.53]).reshape([1, 1, 1, 3]);
      const pixelStd = tf.tensor([58.395, 57.12, 57.375]).reshape([1, 1, 1, 3]);
      const img = await tf.browser.fromPixels(await loadImage(imageUrl));
      let tensor = tf.image.resizeBilinear(img, [256, 128]);
      tensor = tensor.expandDims(0);
      tensor = tensor.toFloat();
      tensor = tensor.sub(pixelMean).div(pixelStd);
    
      return tensor;
    }

    static postProcessing(feat){
      const EPSILON = 1e-7; 
      const normalizedFeat =  tf.tidy(() => {
        const norm = feat.norm('euclidean', 1, true);
        return feat.div(norm.add(EPSILON));
    });

    return normalizedFeat;
}

    async loadStg3Weights() {
      const weightsDict = await this.loadWeights('stage3_embedding_head_weights.json');
      const layerWeights = weightsDict["embedding_head"]["batch_normalization"].map(w => tf.tensor(w));
      this.heads.bottleneck.setWeights(layerWeights);
    }

    async save(path) {
        return this.model.save(path);
    }

    getConfig() {
      return super.getConfig();
    }

    static fromConfig(cls, config) {
      return new cls(config);
    }
}

export default ResNet50Bot;