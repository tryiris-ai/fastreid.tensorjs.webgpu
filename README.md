# FastReID for tensorflow js

Javascript project that loads a fastreid model using tensorflow js with webgpu support. The model wrote in this repo is the ResNet50 custom architecture based on Bag of tricks: [BoT(R50)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

## Installation
This project can be installed and tested through npm and webpack config:
```bash
$ npm install
$ npm start
```
The main entry points are in ./src/index.html and ./src/index.ts. The repo is ready to load the binary model and test images from an http server. You can run this other server cloning this [model server repo](https://github.com/tryiris-ai/model.tensorjs.server)