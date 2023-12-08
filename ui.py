import logging

import gradio as gr
EXTRACTORS = [
    'clip',
    # 'inceptionv3',
    # 'googlenet',
    # 'resnet101',
    # 'vgg19',
    # 'vit',
]
MODELS = [
    'flickr8k',
    'shinkai',
    'pokemon',
]
INFER_MODES = [
    "beam_search",
    "greedy_search",
]
import argparse

from utils import inference, inference2, get_feature_extractor
from model import *
from transformers import AutoTokenizer
import torch
from PIL import Image
from argparse import ArgumentParser



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = None
logging.basicConfig()
logging.root.setLevel(logging.INFO)
class UIWrapper:
    def __init__(self):
        self.model_name = MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = NIC_BN(vocab_size=tokenizer.vocab_size, embed_dim=1024, hidden_size=512, feature_dim=512)
        model_path = f"models/{MODELS[0]}.pth"
        state = torch.load(model_path)
        state_dict = state['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.extractor =  get_feature_extractor("clip").to(device)
        logging.info("UI setup completed!")
    def infer(self,model_name, extractor_name, mode, img_path):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if model_name != self.model_name:
            logging.info(f"Switching model to {model_name}")
            model_path = f"models/{model_name}.pth"
            state = torch.load(model_path)
            state_dict = state['state_dict']
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_name = model_name
        img = Image.open(img_path)
        captions = inference2([img],self.model,tokenizer,extractor_name,self.extractor,mode=mode)
        # captions = inference([img], model, tokenizer, extractor_name="clip", mode=mode)
        print(captions)
        return captions[0]
    def interface(self):
        model_selection = gr.Dropdown(choices=MODELS, value=MODELS[0], label="Model", interactive=True)
        extractor_selection = gr.Dropdown(choices=EXTRACTORS, value=EXTRACTORS[0], label="Feature Extractor",
                                          interactive=True)
        infer_mode_selection = gr.Dropdown(choices=INFER_MODES, value=INFER_MODES[0], label="Inference Mode",
                                           interactive=True)
        image_upload = gr.Image(interactive=True, label="Input Image", type="filepath")
        image_upload_btn = gr.Button(value="Predict!", variant="primary", interactive=True)
        result_captions = gr.Textbox(interactive=False, label="Result")

        with gr.Blocks() as demo:
            with gr.Row():
                model_selection.render()
                extractor_selection.render()
                infer_mode_selection.render()
            with gr.Row():
                image_upload.render()
            with gr.Row():
                image_upload_btn.render()
            with gr.Row():
                result_captions.render()
            image_upload_btn.click(self.infer,
                                   inputs=[model_selection, extractor_selection, infer_mode_selection, image_upload],
                                   outputs=[result_captions])
        return demo

if __name__ == "__main__":
    demo = UIWrapper().interface()
    demo.launch()