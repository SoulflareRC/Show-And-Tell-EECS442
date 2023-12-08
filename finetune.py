from model import *
from dataset_hg import Flickr8kDataset2, ShinkaiDataset, PokemonDataset
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

from utils import train_model, evaluate_model, save_checkpoint, restore_checkpoint,restore_checkpoint_from_file, eval, eval2, finetune_eval, finetune_vis
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    device = torch.device(device)
    data_dir = "./data/flickr8k"

    '''Hyperparameters'''
    batch_size = 256
    emb_dim = 1024
    hidden_size = 512
    feature_dim = 512
    feature_extractor = "clip"
    lr = 1e-4
    # lr = 1e-2
    log_dir = Path("log")
    epochs = 500
    log_period = 10
    save_period = 20
    patience = 5 # if validation loss doesn't decrease after 5 epochs, save model and stop training
    name = "NIC"

    config_name = f"shinkai_patience5_dropout_bn_{feature_extractor}_{feature_dim}_emb_{emb_dim}_hidden_{hidden_size}_Adam_{lr}"

    # config_name = "test"

    '''Dataset'''
    train_ds = ShinkaiDataset(split="train",feature_extractor=feature_extractor)
    test_ds = ShinkaiDataset(split="test",feature_extractor=feature_extractor)
    val_ds = ShinkaiDataset(split="valid",feature_extractor=feature_extractor)

    # train_ds = PokemonDataset(split="train", feature_extractor=feature_extractor)
    # test_ds = PokemonDataset(split="test", feature_extractor=feature_extractor)
    # val_ds = PokemonDataset(split="valid", feature_extractor=feature_extractor)

    train_loader,test_loader,val_loader = DataLoader(train_ds,batch_size=batch_size), DataLoader(test_ds,batch_size=batch_size), DataLoader(val_ds,batch_size=batch_size)
    vocab_size = train_ds.tokenizer.vocab_size

    '''Setup'''
    model = NIC_BN(vocab_size=vocab_size,embed_dim=emb_dim,hidden_size=hidden_size,feature_dim=feature_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    '''Load base model'''
    state = restore_checkpoint_from_file("log/NICdropout_bn_clip_512_emb_1024_hidden_512_Adam_0.0001/NIC_epoch_98_dropout_bn_clip_512_emb_1024_hidden_512_Adam_0.0001.pth")
    state_dict = state['state_dict']
    model.load_state_dict(state_dict)
    '''Training'''
    start_epoch = 0
    stats = [ ]
    '''Resume training'''
    # state = restore_checkpoint(log_dir,name,config_name)
    # state_dict, stats = state['state_dict'], state['stats']
    # model.load_state_dict(state_dict)
    # start_epoch = stats[-1]['epoch']

    global_min_val_loss = 1e9
    patience_cnt = 0
    model = model.to(device)
    criterion = criterion.to(device)

    for i in tqdm(range(start_epoch,epochs)):
        total_loss, total_acc, total_cnt = 0,0,0
        for batch in tqdm(train_loader,position=0,leave=True):
            loss, acc = train_model(model,criterion,optimizer,batch)
            total_loss += loss
            total_acc += acc
            total_cnt += 1
        train_avg_loss = total_loss / total_cnt
        train_avg_acc = total_acc / total_cnt
        logging.info(f"[TRAINING] Avg loss:{train_avg_loss} Avg acc:{train_avg_acc}")
        val_avg_loss, val_avg_acc = evaluate_model(model,criterion,val_loader)
        logging.info(f"[VALIDATION] Avg loss:{val_avg_loss} Avg acc:{val_avg_acc}")
        step = i+1
        '''early stopping'''
        if val_avg_loss < global_min_val_loss:
            global_min_val_loss = val_avg_loss
            patience_cnt = 0
            logging.info(f"Patience Counter set to:({patience_cnt}/{patience}) Global min val loss set to:{global_min_val_loss}")
        else:
            patience_cnt += 1
            logging.info(f"Patience Counter:({patience_cnt}/{patience})")
        '''logging'''
        if step % log_period == 0 or patience_cnt == patience:
            stat = {
                "epoch": step,
                "train_loss": train_avg_loss,
                "train_acc": train_avg_acc,
                "val_loss": val_avg_loss,
                "val_acc": val_avg_acc,
            }
            stats.append(stat)
            logging.info(json.dumps(stat,indent=4))
            if patience_cnt >= patience:
                logging.info("Patience exceeded, early stopping...")
                break
        if step % save_period == 0 :
            save_checkpoint(model=model,log_dir=log_dir,stats=stats,name=name,config_name=config_name)

    '''Testing'''

    model = model.to(torch.device("cpu"))
    model = model.eval()
    scores = finetune_eval(test_ds,train_ds.tokenizer,model)
    logging.info(json.dumps(scores,indent=4))
    for i in range(6):
        finetune_vis(test_ds,model)
    metadata = {"metrics":scores}
    #
    #
    save_checkpoint(model,log_dir,stats,name,config_name,metadata)



