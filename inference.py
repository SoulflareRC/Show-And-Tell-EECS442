import argparse

from utils import inference
from model import *
from transformers import AutoTokenizer
import torch
from PIL import Image
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def infer(model_path,img_path,mode="beam_search"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = NIC_BN(vocab_size=tokenizer.vocab_size, embed_dim=1024, hidden_size=512, feature_dim=512)
    # model_path = "log/NICdropout_bn_clip_512_emb_1024_hidden_512_Adam_0.0001/NIC_epoch_98_dropout_bn_clip_512_emb_1024_hidden_512_Adam_0.0001.pth"
    state = torch.load(model_path)
    state_dict = state['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    # for i in range(6):
    #     eval_vis(flickr_ds,model)
    # img_path = r"D:\pycharmWorkspace\Show-And-Tell-442\test_img.jpg"
    img = Image.open(img_path)
    captions = inference([img], model, tokenizer, extractor_name="clip",mode=mode)
    print(captions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",type=str,help="trained model path",required=True)
    parser.add_argument("-i","--image",type=str,help="image path",required=True)
    parser.add_argument("-c", "--mode", type=str, help="search mode, can be either beam_search or greedy_search", required=False, choices=["greedy_search","beam_search"])

    args = parser.parse_args()
    infer(args.model,args.image,args.mode)
