import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import ResNet50Bot from './fastreid/ResNet50Bot';

async function getGPUInfo() {
  if ('gpu' in navigator) {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const info = await adapter.requestAdapterInfo();
      console.log('GPU Vendor:', info.vendor);
      console.log('GPU Architecture:', info.architecture);
      return info;
    }
  }
  return null;
}

async function initializeWebGPU() {
    try {
        const ok = await tf.setBackend('webgpu');
        // await tf.setBackend('cpu');
        console.log('$WEBGPU', 'Backend set to webgpu:', ok);
        if (!ok) {
            console.error('$WEBGPU', 'Failed to set backend to webgpu');
            await tf.setBackend('webgl');
          }

          // Waiting for TF to be ready
          console.log('$WEBGPU', 'Waiting for TF to be ready...');
          await tf.ready();
        const isWebGPU = tf.getBackend() === 'webgpu';
        const outputElement = document.getElementById('output');
        if (outputElement) {
            outputElement.innerHTML = isWebGPU
                ? 'WebGPU backend is initialized and being used.'
                : 'WebGPU backend is not available. Using fallback.';
        }

        console.log('Current backend:', tf.getBackend());
        console.log('Is WebGPU:', isWebGPU);
        getGPUInfo().then(info => {
          if (info) {
            console.log(`Using ${info.vendor} ${info.vendor} ${info.device} ${info.architecture}`);
          } else {
            console.log('WebGPU not available');
          }
        });

        const image1 = await ResNet50Bot.loadAndPreprocessImage('http://localhost:8080/testImages/1.jpg');
        const image2 = await ResNet50Bot.loadAndPreprocessImage('http://localhost:8080/testImages/2.jpg');
        console.log('Image loaded');

        // const inputShape = [256, 128, 3];
        // const resnet2 = new ResNet50Bot({
        //     inputShape: inputShape,
        //     buildModel: false
        //   });
        const resnet = await ResNet50Bot.load('http://localhost:8080/models/resnet50');
        console.log('Model loaded');

        // measure inference time
        console.time('Inference time');
        let x1 = resnet.predict(image1) as tf.Tensor;
        console.timeEnd('Inference time');
        let val1 = x1.array();
        console.log('Result:', val1);

        let feat1 = ResNet50Bot.postProcessing(x1);
        const result1 = tf.tidy(() => {
            const squeezedFeat = feat1.squeeze();
            const normFeat = squeezedFeat.norm();
            return {
                squeezedFeat: squeezedFeat,
                norm: normFeat
            };
          });
        // const sqeezedFeatArray1 = await result1.squeezedFeat.array();
        // const normValue1 = await result1.norm.data();

        // predict image 2
        console.time('Inference time im2');
        let x2 = resnet.predict(image2) as tf.Tensor;
        console.timeEnd('Inference time im2');
        let feat2 = ResNet50Bot.postProcessing(x2);
        const result2 = tf.tidy(() => {
            const squeezedFeat = feat2.squeeze();
            const normFeat = squeezedFeat.norm();
            return {
                squeezedFeat: squeezedFeat,
                norm: normFeat
            };
          });

          // Calculate cosine similarity
        const dotProduct = tf.dot(result1.squeezedFeat, result2.squeezedFeat);
        const normProduct = tf.mul(result1.norm, result2.norm);
        const similarity = dotProduct.div(normProduct);
        const similarityValue = await similarity.data();
        console.log('Similarity:', similarityValue);

        // For loop to inferencing the same image infinite times just to test 
        // the performance of the WebGPU backend
        // for (let i = 0; i < 50; i++) {
        //     console.time('Inference time im4');
        //     let x4 = resnet.predict(image1) as tf.Tensor;
        //     console.timeEnd('Inference time im4');
        //     x4.dispose();
        //     // Sleep for 50ms
        //     await new Promise((resolve) => setTimeout(resolve
        //     , 50));
        // }

        // Free up the memory
        x1.dispose();
        x2.dispose();
        feat1.dispose();
        feat2.dispose();
        result1.squeezedFeat.dispose();
        result1.norm.dispose();
        result2.squeezedFeat.dispose();
        result2.norm.dispose();
        dotProduct.dispose();
        normProduct.dispose();
        similarity.dispose();
        
        
    } catch (error) {
        console.error('Error initializing WebGPU:', error);
    }
}

initializeWebGPU();