from model import *
from dataset_hg import Flickr8kDataset2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
# print("What")
from pathlib import Path
import json

from utils import train_model, evaluate_model, save_checkpoint, restore_checkpoint,restore_checkpoint_from_file, eval, eval2
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    device = torch.device(device)
    data_dir = "./data/flickr8k"

    '''Hyperparameters'''
    batch_size = 256
    emb_dim = 768
    hidden_size = 512
    feature_extractor = "inceptionv3"
    feature_dim = 2048
    lr = 1e-2
    log_dir = Path("log")
    epochs = 500
    log_period = 10
    save_period = 20
    patience = 5 # if validation loss doesn't decrease after 5 epochs, save model and stop training
    name = "NIC"
    config_name = f"emb_{emb_dim}_hidden_{hidden_size}_SGD_{lr}"
    # config_name = "test"

    '''Dataset'''
    test_ds = Flickr8kDataset2(data_dir,split="test",feature_extractor=feature_extractor)
    test_loader=DataLoader(test_ds,batch_size=batch_size)
    vocab_size = test_ds.tokenizer.vocab_size

    '''Setup'''
    model = NIC_BN(vocab_size=vocab_size,embed_dim=emb_dim,hidden_size=hidden_size,feature_dim=feature_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    '''Training'''
    start_epoch = 0
    stats = [ ]
    '''Resume training'''
    state = restore_checkpoint_from_file("log/NICdropout_bn_emb_768_hidden_512_Adam_0.0001/NIC_epoch_84_dropout_bn_emb_768_hidden_512_Adam_0.0001.pth")
    state_dict, stats = state['state_dict'], state['stats']
    model.load_state_dict(state_dict)
    start_epoch = stats[-1]['epoch']

    '''Testing'''

    model = model.to(torch.device("cpu"))
    model = model.eval()
    scores = eval2(test_ds,test_ds.tokenizer,model,mode="beam_search")
    logging.info(json.dumps(scores,indent=4))
    logging.info(json.dumps(stats[-1],indent=4))
    metadata = {"metrics":scores}

    # save_checkpoint(model,log_dir,stats,name,config_name,metadata)



