# Show and Tell 
This repo serves as Umich's FA23 EECS 442's final project code submission. In this project, we reimplemented the Google NIC model for Image Captioning from [this paper](https://arxiv.org/pdf/1411.4555.pdf), built a training/testing pipeline, and experimented with the model. 
## Some qualitative results 
![flickr1](https://github.com/SoulflareRC/Show-And-Tell-EECS442/assets/107384280/b9b9a27b-c41d-469b-b95a-8ae40fdd512e)
![shinkai1](https://github.com/SoulflareRC/Show-And-Tell-EECS442/assets/107384280/3150feda-b481-4911-9d4b-e4bfeded591a)
![pokemon1](https://github.com/SoulflareRC/Show-And-Tell-EECS442/assets/107384280/7d3a8ac9-5c8f-4d05-9933-134e9147847a)

## Report 
The [final report](/showcase/report.pdf) of this project is included in this repo. The report is written in CVPR format and has 4 pages excluding citations.
## Pretrained models 
The pretrained models are provided [here](https://drive.google.com/drive/folders/17B5zF7-IrEjcRfwX8M9bANsWTwrFZLyd?usp=sharing). Only the **Final Model** (mentioned in the report) trained on different datasets are provided. 
## Installation 
We provide an interactive demo built with [Gradio](https://github.com/gradio-app/gradio). To use the demo, follow the installation instructions. 

Clone this repo 
```
https://github.com/SoulflareRC/Show-And-Tell-EECS442.git 
```
Install requirements 
```
pip install requirements.txt
```
Download the pretrained models, create a `models` folder in the root directory of the project, and put the pretrained models in the `models` folder. 
Launch the UI and go to `http://127.0.0.1:7860/`. It might take longer to predict for the first image as the models need to be loaded into the GPU device. 
```
python ui.py
```
![image](https://github.com/SoulflareRC/Show-And-Tell-EECS442/assets/107384280/c9e8927e-f21b-4cc7-8212-6947ae6c32fe)
