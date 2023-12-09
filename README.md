# Show and Tell 
This repo serves as Umich's FA23 EECS 442's final project code submission. In this project, we reimplemented the Google NIC model for Image Captioning from [this paper](https://arxiv.org/pdf/1411.4555.pdf), built a training/testing pipeline, and experimented with the model. 
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
Launch the UI and go to `http://127.0.0.1:7860/`
```
python ui.py
```
