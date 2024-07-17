import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
// import Conv2dCustom  from './Conv2dCustom';
import ResNet50Bot from './fastreid/ResNet50Bot';




// const pixelMean = tf.tensor([123.675, 116.28, 103.53]).reshape([1, 1, 1, 3]);
// const pixelStd = tf.tensor([58.395, 57.12, 57.375]).reshape([1, 1, 1, 3]);

// async function loadAndPreprocessImage(imagePath: string, pixelMean: tf.Tensor4D, pixelStd: tf.Tensor4D): Promise<tf.Tensor4D> {
//     // if (!fs.existsSync(imagePath)) {
//     //     console.error('The image does not exist in the specified path:', imagePath);
//     //     return;
//     // }
    
//     const image = await Jimp.read(imagePath);
//     image.resize(128, 256, Jimp.RESIZE_BICUBIC);
    
//     // Convertir la imagen a un array de píxeles
//     const imageData = new Float32Array(128 * 256 * 3);
//     image.scan(0, 0, image.bitmap.width, image.bitmap.height, function(x, y, idx) {
//         imageData[idx / 4 * 3] = this.bitmap.data[idx];
//         imageData[idx / 4 * 3 + 1] = this.bitmap.data[idx + 1];
//         imageData[idx / 4 * 3 + 2] = this.bitmap.data[idx + 2];
//     });
  
//     // Crear un tensor a partir del array de píxeles
//     let tensor = tf.tensor4d(imageData, [1, 256, 128, 3]);
  
//     // Aplicar la normalización
//     return tensor.sub(pixelMean).div(pixelStd);
//   }


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
        
          console.log('$WEBGPU', 'Done... Loading model...');
        
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
            console.log(`Usando WebGPU con ${info.vendor} - ${info.architecture}`);
          } else {
            console.log('WebGPU no disponible o información no accesible');
          }
        });
        

        // Simple tensor operation to confirm everything is working
        const a = tf.tensor1d([1, 2, 3]);
        const b = tf.tensor1d([4, 5, 6]);
        const c = a.add(b);
        
        // const pixelMean: tf.Tensor4D = tf.tensor([123.675, 116.28, 103.53]).reshape([1, 1, 1, 3]);
        // const pixelStd: tf.Tensor4D = tf.tensor([58.395, 57.12, 57.375]).reshape([1, 1, 1, 3]);
        // const image1 = await ResNet50Bot.loadAndPreprocessImage('./testImages/1.jpg');
        const image1 = await ResNet50Bot.loadAndPreprocessImage('http://localhost:8080/testImages/1.jpg');
        const image2 = await ResNet50Bot.loadAndPreprocessImage('http://localhost:8080/testImages/4.jpg');
        console.log('Image loaded');

        // const inputShape = [256, 128, 3];
        // const resnet2 = new ResNet50Bot({
        //     inputShape: inputShape,
        //     buildModel: false
        //   });
        const resnet = await ResNet50Bot.load();
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
        const sqeezedFeatArray1 = await result1.squeezedFeat.array();
        const normValue1 = await result1.norm.data();

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

        // console.time('Inference time im3');
        // const result = await tf.profile(() => resnet.predict(image1));
        // console.log(JSON.stringify(result, null, 2));
        // console.timeEnd('Inference time im3');

          // Calculate cosine similarity
        const dotProduct = tf.dot(result1.squeezedFeat, result2.squeezedFeat);
        const normProduct = tf.mul(result1.norm, result2.norm);
        const similarity = dotProduct.div(normProduct);
        const similarityValue = await similarity.data();
        console.log('Similarity:', similarityValue);

        // For loop to inferencing the same image infinite times just to test 
        // the performance of the WebGPU backend
        let x4: tf.Tensor;
        // for (let i = 0; i < 100; i++) {
        //     console.time('Inference time im4');
        //     x4 = resnet.predict(image1) as tf.Tensor;
        //     console.timeEnd('Inference time im4');
        //     // x4.dispose();
        //     // Sleep for 50ms
        //     await new Promise((resolve) => setTimeout(resolve
        //     , 100));
        //     console.log('woke up');
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




        // print the name of the layer conv1
        // console.log('Layer name:', conv1.name);
        
        console.log('Tensor operation result:', await c.array());
    } catch (error) {
        console.error('Error initializing WebGPU:', error);
    }
}

initializeWebGPU();