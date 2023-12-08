import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import collections
import typing
from PIL import Image
from transformers import BertTokenizer, AutoTokenizer, CLIPModel, CLIPProcessor
from queue import PriorityQueue
from dataset_hg import Flickr8kDataset, Flickr8kDataset2, ShinkaiDataset, PokemonDataset
from model import NIC, NIC_BN
from tqdm import tqdm
from torchtext.data import metrics
import matplotlib.pyplot as plt
from nltk.translate.meteor_score import meteor_score
import json
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import numpy as np
import operator
from time import time
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import random
import nltk
from textwrap import wrap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
logging.basicConfig()
logging.root.setLevel(logging.INFO)
def train_model(model:NIC,criterion,optimizer,batch):
    '''do one training step, return loss, accuracy'''
    input1,input2, target = batch
    input1 = input1.to(device)
    input2 = input2.to(device)
    target = target.to(device)
    target = target.view(-1)
    optimizer.zero_grad()
    output = model(input1,input2)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    pred = output.argmax(1)
    acc = (pred == target).sum() / pred.size(0)
    return loss.item(), acc.item()

def evaluate_model(model:NIC,criterion,loader:DataLoader):
    '''
    do an entire evaluation loop for the val/test dataset,
    return avg loss, avg acc
    '''
    logging.info("Evaluating model...")
    total_loss, total_acc, total_cnt = 0,0,0
    with torch.no_grad():
        for batch in tqdm(loader,position=0,leave=True):
            input1, input2, target = batch
            input1 = input1.to(device)
            input2 = input2.to(device)
            target = target.to(device)
            target = target.view(-1)
            output = model(input1, input2)
            loss = criterion(output, target)
            pred = output.argmax(1)
            acc = (pred == target).sum() / pred.size(0)
            total_loss += loss.item()
            total_acc += acc.item()
            total_cnt += 1
    avg_loss = total_loss / total_cnt
    avg_acc = total_acc / total_cnt
    return avg_loss, avg_acc

def save_checkpoint(model:NIC, log_dir:Path, stats:list, name="NIC",config_name=None,metadata:dict={}):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "name":name,
        "config_name":config_name,
        "state_dict": model.state_dict(),
        "stats": stats,
    }
    state.update(metadata)
    config = [name,"epoch",stats[-1]["epoch"]]
    if config_name:
        config += [config_name]
    config = [str(x) for x in config]
    fname = "_".join(config)+".pth"
    save_dir = log_dir.joinpath(name+(config_name if config_name else ""))
    model_path = save_dir.joinpath(fname)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    torch.save(state, model_path)
    logging.info(f"Model saved to {model_path}")

def restore_checkpoint_from_file(model_path:Path):
    model_path = Path(model_path)
    state = torch.load(model_path)
    state_dict, stats = state["state_dict"], state['stats']
    logging.info(f"Model loaded from {model_path} and epoch {stats[-1]['epoch']}")
    return state
def restore_checkpoint(log_dir:Path,name="NIC",config_name=None,epoch=None):
    '''
    Load checkpoint with model name and config name
    epoch: if specified then find the model of that epoch, if not specified then load the latest one
    '''
    save_dir = log_dir.joinpath(name+(config_name if config_name else ""))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    if epoch:
        config = [name, "epoch", epoch]
        if config_name:
            config += [config_name]
        config = [str(x) for x in config]
        fname = "_".join(config) + ".pth"
        model_path = save_dir.joinpath(fname)
    else:
        path_list = save_dir.glob("*.pth")
        # print(path_list)
        model_path = max(path_list,key=lambda p:p.stat().st_ctime)
    # print(model_path)
    state = torch.load(model_path)
    state_dict, stats = state["state_dict"], state['stats']
    logging.info(f"Model loaded from {model_path} and epoch {stats[-1]['epoch']}")
    return state

def get_img_transforms():
    return T.Compose([
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
def _get_image_features( imgs, extractor: models.Inception3, extractor_name="inceptionv3"):
    '''get image features for a batch'''
    # Load all the images in a batch
    img_size = 224
    if extractor_name == "inceptionv3":
        img_size = 299
    elif extractor_name == "clip":
        return extractor(imgs).squeeze()
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((img_size, img_size)),  # inceptionv3 needs this
    ])
    images = [trans(img) for img in imgs]
    images = torch.stack(images, dim=0).to(device)
    # print(images.shape)
    res = extractor(images)
    return res.squeeze()
def extract_features(img:Image,transforms=None):
    '''
    :param img: A PIL Image
    :return: a feature vector (tensor)
    '''
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Resize(299), # inceptionv3 needs this
    ])
    img = to_tensor(img)
    if transforms:
        img = transforms(img)
    img = img.unsqueeze(0).to(device)
    extractor = get_feature_extractor().to(device)
    return extractor(img).squeeze().detach().cpu()

def caption_to_vector(caption:str):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(caption,padding=True)
    return tokens["input_ids"]
def vector_to_caption(vector:torch.Tensor|list):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.decode(vector)
def greedy_search(model:NIC,tokenizer:BertTokenizer,img_ft:torch.Tensor,max_len=45):
    # '''1. get the image's feature vector first'''
    # trans = get_img_transforms()
    # img_ft = extract_features(image,trans)
    # '''2. get hidden from image and then start with only [CLS] '''

    img_ft = img_ft.unsqueeze(0)
    # print(img_ft.shape)
    # h = encoder(img_ft) # starting hidden state
    h = model.encode(img_ft)
    # print(h.shape)
    '''3. Get word one by one'''
    word = torch.tensor(tokenizer.cls_token_id).view(1,1) # start with [CLS]
    res = torch.zeros((1,max_len),dtype=torch.int)

    for i in range(max_len):
        # print(word.shape)
        # next_word, h = decoder(word,h)
        next_word,h = model.decode(word,h)
        # print(next_word.shape)
        '''
        Greedily choose the word that maximizes p  
        '''
        next_pred = next_word.argmax(dim=-1)
        pred = next_pred.squeeze()
        # print(pred)
        # decoded_word = tokenizer.decode(pred)
        # print(decoded_word)
        res[0,i] = pred.item()
        if pred.item() == tokenizer.sep_token_id:
            # print(f"{tokenizer.sep_token} encountered")
            # print(res)
            break
        word = next_pred
    decoded = tokenizer.decode(res.squeeze(), skip_special_tokens=True)
    # print(decoded)
    return decoded

def beam_search(model:NIC,tokenizer:BertTokenizer,img_ft:torch.Tensor,max_len=45,beam_width=3,num_captions=1):
    '''
    iteratively consider k best sentences up to time t as candidates to generate
    sentences of size t+1, keep only resulting best k of them
    this is a bit similar to BFS,
    1. starting with [CLS] token and img hidden state
    2. generate prob score of <vocab size>
    3. get k highest prob score, push into pq
    '''
    # model = model.to(device)
    # img_ft = img_ft.to(device)
    # just pass in the feature
    img_ft = img_ft.unsqueeze(0)
    h = model.encode(img_ft) # starting hidden state
    # print(h.shape)
    '''3. Get word one by one'''
    word = torch.tensor(tokenizer.cls_token_id).view(1,1) #.to(device) # start with [CLS]
    '''4. Start beam search'''
    cur_word = word # 1x1
    cur_seq = [word.squeeze().item()] # a list
    cur_item = [-0.0,cur_word,cur_seq,h] # score, current word, current seq, hidden state to use for the next step
    '''5. Maintain a pq, keeps top k by normalized log prob score'''
    pq = PriorityQueue()
    completed_seqs = []
    pq.put( (-0.0,cur_item) )
    for _ in range(max_len):
        if pq.empty():
            break
        score, item = pq.get()
        score, word, seq, h = item
        if word.item() == tokenizer.sep_token_id:
            completed_seqs.append(item)
            if len(completed_seqs) >= num_captions:
                break
            continue # skip this item if reached sep
        word = word.view(1,1)
        outputs,new_h = model.decode(word,h)
        sub_candidates, sub_candidates_ids = torch.topk(outputs, k=beam_width, dim=-1)
        # print(sub_candidates)
        # print(sub_candidates_ids)
        sub_candidates = sub_candidates.squeeze()
        sub_candidates_ids = sub_candidates_ids.squeeze()
        for j in range(beam_width):
            new_word = sub_candidates_ids[j]
            new_seq = seq+[new_word]
            log_score = torch.log(sub_candidates[j])
            new_score = score + log_score
            norm_score = new_score / (len(new_seq)) # sort by normalized score, otherwise longer captions will never be selected
            new_item = [new_score,new_word,new_seq,new_h]
            pq.put((-norm_score,new_item)) # negate score since pq is asc by default
    if len(completed_seqs) < num_captions: # handle case when failed to find sep
        while not pq.empty() and len(completed_seqs)<num_captions:
            _, item = pq.get()
            completed_seqs.append(item)
    completed_seqs = sorted(completed_seqs,key=lambda x:x[0]/len(x[2]),reverse=True)
    completed_captions = []
    for item in completed_seqs:
        score, word, seq, h = item
        decoded = tokenizer.decode(seq,skip_special_tokens=True)
        # print(decoded,score.item()/len(seq))
        completed_captions.append(decoded)
    # completed_captions = [tokenizer.decode(seq) for score, word, seq, h in completed_seqs]
    return completed_captions

def beam_search2(model:NIC,tokenizer:BertTokenizer,img_ft:torch.Tensor,max_len=45,beam_width=3):
    '''
    this is top 1 implementation is from
    https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00346/96473/Best-First-Beam-Search
    '''
    # '''1. get the image's feature vector first'''
    # trans = get_img_transforms()
    # img_ft = extract_features(image,trans)
    # '''2. get hidden from image and then start with only [CLS] '''
    img_ft = img_ft.unsqueeze(0)
    h = model.encode(img_ft) # starting hidden state
    # print(h.shape)
    '''3. Get word one by one'''
    word = torch.tensor(tokenizer.cls_token_id).view(1,1) # start with [CLS]
    '''4. Start beam search'''
    cur_word = word # 1x1
    cur_seq = [word.squeeze().item()] # a list
    cur_item = [-0.0,cur_word,cur_seq,h] # score, current word, current seq, hidden state to use for the next step
    '''5. Maintain a pq, keeps top k by normalized log prob score'''
    pq = PriorityQueue()
    completed_seqs = []
    pq.put( (-0.0,cur_item) )
    for _ in range(max_len):
        if pq.empty():
            break
        score, item = pq.get()
        score, word, seq, h = item
        if word.item() == tokenizer.sep_token_id:
            # completed_seqs.append(item)
            # if len(completed_seqs) >= num_captions:
            #     break
            norm_score = score  / len(seq)
            pq.put((-norm_score,item))
            continue # skip this item if reached sep
        word = word.view(1,1)
        outputs,new_h = model.decode(word,h)
        sub_candidates, sub_candidates_ids = torch.topk(outputs, k=beam_width, dim=-1)
        # print(sub_candidates)
        # print(sub_candidates_ids)
        sub_candidates = sub_candidates.squeeze()
        sub_candidates_ids = sub_candidates_ids.squeeze()
        for j in range(beam_width):
            new_word = sub_candidates_ids[j]
            new_seq = seq+[new_word]
            log_score = torch.log(sub_candidates[j])
            new_score = score + log_score
            norm_score = new_score  / (len(new_seq)) # sort by normalized score, otherwise longer captions will never be selected
            new_item = [new_score,new_word,new_seq,new_h]
            pq.put((-norm_score,new_item)) # negate score since pq is asc by default
    caption = pq.get()
    score, item  = caption
    score, word, seq, h = item
    decoded = tokenizer.decode(seq,skip_special_tokens=True)
    # print(decoded)
    # return [decoded]
    return decoded
def eval2(dataset:Flickr8kDataset2, tokenizer:BertTokenizer, model:NIC, bleu_n=4,mode="beam_search", **kwargs):
    '''
    The difference here is that this uses the fact that flickr8k dataset has img-txt 5 in a row
    '''
    # ds_test = dataset.data
    eval_metrics = {
        "bleu":None,
        "meteor": None,
    }
    total_bleu_scores = [0] * bleu_n
    total_meteor_score = 0
    total_cnt = 0

    # items = []
    # extractor = get_feature_extractor()
    # extractor = extractor.to(device)

    ds_img_fts, ds_txt_fts = dataset.img_data, dataset.txt_data
    for i in tqdm(range(len(ds_img_fts)//5)):
        img_ft, ref_txt_ft = ds_img_fts[5*i], ds_txt_fts[5*i:5*i+5]
        captions = [tokenizer.decode(txt_ft,skip_special_tokens=True) for txt_ft in ref_txt_ft]
        ref_txt_ft = [tokenizer.tokenize(x) for x in captions]
        # print(captions)
        # print(ref_txt_ft)
        ref_txt_ft = [ref_txt_ft]
        if mode == "beam_search":
            pred_captions = beam_search(model,tokenizer,img_ft,beam_width=10,num_captions=1)
            pred_caption = pred_captions[0]
        elif mode == "greedy_search":
            pred_caption = greedy_search(model,tokenizer,img_ft)
        elif mode == "beam_search2":
            pred_caption = beam_search2(model,tokenizer,img_ft,beam_width=10)

        cand_txt_ft = [tokenizer.tokenize(pred_caption)]
        '''calculate meteor score'''
        m_captions = [word_tokenize(c) for c in captions]
        pred_caption = word_tokenize(pred_caption)
        # print(m_captions)
        # print(pred_caption)
        m_score = meteor_score(m_captions,pred_caption)
        # print(f"Ref:{m_captions} Cand:{pred_caption} Meteor Score:{m_score}")
        total_meteor_score += m_score
        # print(m_score)
        '''calculate bleu score'''
        for i in range(bleu_n):
            n = i+1
            bleu_score = metrics.bleu_score(cand_txt_ft, ref_txt_ft, max_n=n, weights=[1 / n] * n)
            total_bleu_scores[i] += bleu_score
            # print(i,bleu_score)
        total_cnt+=1
        # break

    total_meteor_score = total_meteor_score/total_cnt
    total_bleu_scores = [x / total_cnt for x in total_bleu_scores]
    eval_metrics['bleu']=total_bleu_scores
    eval_metrics['meteor']=total_meteor_score

    return eval_metrics
def finetune_eval(dataset:ShinkaiDataset, tokenizer:BertTokenizer, model:NIC, bleu_n=4,mode="beam_search", **kwargs):
    '''
    The difference here is that this uses the fact that flickr8k dataset has img-txt 5 in a row
    '''
    # ds_test = dataset.data
    eval_metrics = {
        "bleu":None,
        "meteor": None,
    }
    total_bleu_scores = [0] * bleu_n
    total_meteor_score = 0
    total_cnt = 0

    ds_img_fts, ds_txt_fts = dataset.img_data, dataset.txt_data
    for i in tqdm(range(len(ds_img_fts))):
        img_ft, ref_txt_ft = ds_img_fts[i], [ds_txt_fts[i] ]
        captions = [tokenizer.decode(txt_ft,skip_special_tokens=True) for txt_ft in ref_txt_ft]
        ref_txt_ft = [tokenizer.tokenize(x) for x in captions]
        # print(captions)
        # print(ref_txt_ft)
        ref_txt_ft = [ref_txt_ft]
        if mode == "beam_search":
            pred_captions = beam_search(model,tokenizer,img_ft,beam_width=10,num_captions=1)
            pred_caption = pred_captions[0]
        elif mode == "greedy_search":
            pred_caption = greedy_search(model,tokenizer,img_ft)
        elif mode == "beam_search2":
            pred_caption = beam_search2(model,tokenizer,img_ft,beam_width=10)

        cand_txt_ft = [tokenizer.tokenize(pred_caption)]
        '''calculate meteor score'''
        m_captions = [word_tokenize(c) for c in captions]
        pred_caption = word_tokenize(pred_caption)
        # print(m_captions)
        # print(pred_caption)
        m_score = meteor_score(m_captions,pred_caption)
        print(f"Ref:{m_captions} Cand:{pred_caption} Meteor Score:{m_score}")
        total_meteor_score += m_score
        # print(m_score)
        '''calculate bleu score'''
        for i in range(bleu_n):
            n = i+1
            bleu_score = metrics.bleu_score(cand_txt_ft, ref_txt_ft, max_n=n, weights=[1 / n] * n)
            total_bleu_scores[i] += bleu_score
            # print(i,bleu_score)
        total_cnt+=1
        # break

    total_meteor_score = total_meteor_score/total_cnt
    total_bleu_scores = [x / total_cnt for x in total_bleu_scores]
    eval_metrics['bleu']=total_bleu_scores
    eval_metrics['meteor']=total_meteor_score

    return eval_metrics
def finetune_vis(dataset:ShinkaiDataset, model:NIC):
    items = dataset.split_data
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax = ax.ravel()
    sample_size = 4
    samples = random.sample(range(len(dataset)), sample_size)
    for i,n in tqdm(enumerate(samples)):
        # n = random.randint(0, len(items) - 1)
        print(n)
        item = items[n]
        img, txt = item['image'], item['text']
        # # print(img_path,txt)
        # img = Image.open(img_path)
        img_ft, txt_ft, _ = dataset[n]
        # print(img_ft.shape)
        pred_txt = beam_search2(model, dataset.tokenizer, img_ft, beam_width=10)
        print("txt ft:",dataset.tokenizer.decode(txt_ft,skip_special_tokens=True))
        # print(txt, pred_txt)
        caption = f"Predicted:{pred_txt} \nCorrect:{txt}"
        print(caption)
        # txt = "\n".join(wrap(,40)
        # "\n".join(wrap(caption, 40))
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(caption, size=16, wrap=True)
    plt.show()
def eval_vis(dataset:Flickr8kDataset2,model:NIC):
    model.eval()

    data_pairs = dataset.data_dict[dataset.split]
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    ax = ax.ravel()
    # fig.tight_layout()
    sample_size = 4
    samples = random.sample(range(len(data_pairs)), sample_size)
    for i, n in enumerate(samples):
        # n = samples[i]
        print(n)
        img_path, txt = data_pairs[n]
        # print(img_path,txt)
        img = Image.open(img_path)
        img_ft, txt_ft, _ = dataset[n]
        # print(img_ft.shape)
        pred_txt = beam_search2(model, dataset.tokenizer, img_ft)
        print(txt, pred_txt)
        caption = f"Predicted: {pred_txt} \nCorrect: {txt}"
        # txt = "\n".join(wrap(,40)
        # "\n".join(wrap(caption, 40))
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(caption, size=16, wrap=True)
    plt.show()
def eval(dataset:Flickr8kDataset, tokenizer:BertTokenizer, model:NIC, bleu_n=4, **kwargs):
    '''
    This is too slow, deprecated.

    METEOR Score:
    1. alignment
    map each unigram in candidates to at most 1 reference, pick the alignment with most connections
    use 3 modules for alignment: exact, porter stemmed, WordNet synonymy
    2. calculate precision, recall, f1-score based on alignment
    3. penalty
    matched chunks of aligned consecutive unigrams in reference and candidate, the longer the chunk is, the less the chunks is
    (1-penalty) * f1-score


    Function to evaluate the performance of model on dataset using bleu score
    The loader will give (img_ft, txt_ft) for each item, and txt_ft
    can be recovered into caption
    https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b#:~:text=Finally%2C%20to%20calculate%20the%20Bleu,Average%20of%20the%20Precision%20Scores.&text=Bleu%20Score%20can%20be%20computed,%2C%20we%20use%20N%20%3D%204.
    To calculate the bleu score:
    1. calculate precision n-gram
        how many target n-grams appeared in predicted n-grams?
        use n = [1,4]
    2.  calculate geometric average precision GAP(N)
        GAP(N) = exp(sum(n=[1,N],wn*log(pn)))
        wn is weight n, usually just use uniform weight
    3.  brevity penalty * GAP(N)
        this is to prevent situation where predicted text is too short which will produce very high precision
        if c = pred len, r = target len
        brevity penalty = 1 (if c>r, pred len > target len)
                        = e^(1-r/c) (if c<=r, pred len <= target len)
                        ( this means a ratio < 1 will be multiplied if pred len <= target len)
    implementation:
    1. take an image, get feature vector
    2. generate caption for image
    3. get all the ref captions for the image
    4. compute bleu score

    '''
    ds_test = dataset.data
    eval_metrics = {
        "bleu":None,
        "meteor": None,
    }
    total_bleu_scores = [0] * bleu_n
    total_meteor_score = 0
    total_cnt = 0
    items = []
    extractor = get_feature_extractor()
    extractor = extractor.to(device)

    for item in tqdm(ds_test):
        captions = item["captions"]
        img_path = item['image_path']
        img = Image.open(img_path)
        trans = get_img_transforms()
        to_tensor = T.Compose([
            T.ToTensor(),
            T.Resize(299),  # inceptionv3 needs this
        ])
        img = trans(to_tensor(img))
        img = img.unsqueeze(0).to(device)
        extractor = extractor.to(device)
        img_ft = extractor(img).squeeze().detach().cpu()
        items.append([img_ft,captions])
        # break

    for item in tqdm(items):
        img_ft, captions = item
        ref_txt_ft = tokenizer(captions)['input_ids']
        ref_txt_ft = [tokenizer.convert_ids_to_tokens(x) for x in ref_txt_ft]
        # print(ref_txt_ft)
        ref_txt_ft = [ref_txt_ft]
        pred_captions = beam_search(model,tokenizer,img_ft,beam_width=20,num_captions=1)
        pred_caption = pred_captions[0]
        cand_txt_ft = [tokenizer.convert_ids_to_tokens(tokenizer(pred_caption)['input_ids'])]
        '''calculate meteor score'''
        m_captions = [word_tokenize(c) for c in captions]
        pred_caption = word_tokenize(pred_caption)
        # print(m_captions)
        m_score = meteor_score(m_captions,pred_caption)
        # print(f"Ref:{m_captions} Cand:{pred_caption} Meteor Score:{m_score}")
        total_meteor_score += m_score
        # print(m_score)
        '''calculate bleu score'''
        for i in range(bleu_n):
            n = i+1
            total_bleu_scores[i] += metrics.bleu_score(cand_txt_ft, ref_txt_ft, max_n=n, weights=[1 / n] * n)

        total_cnt+=1
        # break

    total_meteor_score = total_meteor_score/total_cnt
    total_bleu_scores = [x / total_cnt for x in total_bleu_scores]
    eval_metrics['bleu']=total_bleu_scores
    eval_metrics['meteor']=total_meteor_score

    return eval_metrics
def get_feature_extractor(extractor_name="inceptionv3") -> models.Inception3:
        if extractor_name == "inceptionv3":
            extractor = models.inception_v3(weights=models.Inception_V3_Weights)
            extractor.eval()
            extractor.dropout = nn.Identity()  # disable the dropout layer as well
            extractor.fc = nn.Identity()  # disable the fc layer
            # output 1 x 2048
        elif extractor_name == "resnet101":
            extractor = models.resnet101(weights=models.ResNet101_Weights)
            extractor.fc = nn.Identity()
            extractor.eval()
            extractor: models.ResNet
            # output 1 x 2048
        elif extractor_name == "vgg19":
            extractor = models.vgg19_bn(weights=models.VGG19_BN_Weights)
            cls_modules = [*extractor.classifier]
            extractor.classifier = nn.Sequential(*(cls_modules[:-1]))
            extractor.eval()
            extractor: models.VGG
            # output 1 x 4096
        elif extractor_name == "vit":
            class ViTExtractor:
                def __init__(self):
                    self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights)
                    self.preprocessing = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()

                def __call__(self, img):
                    img = self.preprocessing(img)

                    feats = self.vit._process_input(img)

                    # Expand the class token to the full batch
                    batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
                    feats = torch.cat([batch_class_token, feats], dim=1)

                    feats = self.vit.encoder(feats)

                    # We're only interested in the representation of the classifier token that we appended at position 0
                    feats = feats[:, 0]
                    return feats

                def to(self, device):
                    self.vit = self.vit.to(device)
                    return self

            extractor = ViTExtractor()
            # output 1 x 768
        elif extractor_name == "clip":
            class CLIPExtractor:
                def __init__(self):
                    self.clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

                def __call__(self, images):
                    # Use the processor to encode the batch of images
                    inputs = self.clip_processor(text=None, images=images, return_tensors="pt", padding=True)
                    pixel_val = inputs['pixel_values'].to(device)
                    image_features = self.clip_model.get_image_features(pixel_values=pixel_val)
                    image_features = image_features.detach().cpu()
                    return image_features

                def to(self, device):
                    self.clip_model = self.clip_model.to(device)
                    return self

            extractor = CLIPExtractor()
        elif extractor_name == "googlenet":
            extractor = models.googlenet(models.GoogLeNet_Weights)
            extractor.fc = nn.Identity()
            extractor.eval()
            # output 1 x 1024
        return extractor
def inference(imgs:list,model:NIC,tokenizer:BertTokenizer,extractor_name="inceptionv3",batch_size=16,mode="beam_search"):
    logging.info(f"Inferencing in {mode} mode...")
    extractor = get_feature_extractor(extractor_name)
    image_batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
    img_features = []
    for batch in tqdm(image_batches):
        batch_features = _get_image_features(batch,extractor, extractor_name).detach().cpu()
        # print(batch_features.shape)
        img_features.append(batch_features)
    all_features = torch.cat(img_features, dim=0)
    all_features = all_features.view(len(imgs),-1)
    captions = []
    # print(all_features.shape)
    for i in range(len(imgs)):
        img_ft = all_features[i]
        # print(img_ft.shape)
        if mode == "beam_search" or mode is None:
            caption = beam_search2(model,tokenizer,img_ft,beam_width=10)
        elif mode == "greedy_search":
            caption = greedy_search(model,tokenizer,img_ft)

        captions.append(caption)
    return captions
def inference2(imgs:list,model:NIC,tokenizer:BertTokenizer,extractor_name="clip",extractor=get_feature_extractor("clip"),batch_size=16,mode="beam_search"):
    '''separate extractor out to prevent redundant loads'''
    logging.info(f"Inferencing in {mode} mode...")

    image_batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
    img_features = []
    for batch in tqdm(image_batches):
        batch_features = _get_image_features(batch,extractor, extractor_name).detach().cpu()
        # print(batch_features.shape)
        img_features.append(batch_features)
    all_features = torch.cat(img_features, dim=0)
    all_features = all_features.view(len(imgs),-1)
    captions = []
    # print(all_features.shape)
    for i in range(len(imgs)):
        img_ft = all_features[i]
        # print(img_ft.shape)
        if mode == "beam_search" or mode is None:
            caption = beam_search2(model,tokenizer,img_ft,beam_width=10)
        elif mode == "greedy_search":
            caption = greedy_search(model,tokenizer,img_ft)

        captions.append(caption)
    return captions

if __name__ == "__main__":
    # scores = [0,0,0,0]
    # scores[:3] = [1/3]*3
    # print(scores)
    #
    # feature_extractor = "clip"
    # data_dir = "./data/flickr8k"
    # ds = Flickr8kDataset2(data_dir=data_dir,split="train",feature_extractor=feature_extractor)
    # sample = next(iter(ds))
    # _, input, target = sample
    # print("Input:",ds.tokenizer.decode(input))
    # print("Target:",ds.tokenizer.decode(target))

    # train_ds = PokemonDataset(split="train", feature_extractor=feature_extractor)
    # test_ds = PokemonDataset(split="test", feature_extractor=feature_extractor)
    # val_ds = PokemonDataset(split="valid", feature_extractor=feature_extractor)
    #
    # train_data = train_ds.split_data
    # for i in range(10):
    #     item = train_data[i]
    #     # print(train_data[i])
    #     print(item['text'])
    #     txt_ft = train_ds.txt_data[i]
    #     print(train_ds.tokenizer.decode(txt_ft,skip_special_tokens=True))

    # from transformers import BertTokenizer, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #

    #
    # # encoder = NICEncoder(vocab_size=tokenizer.vocab_size,embed_dim=1024,hidden_size=512)
    # # encoder.load_state_dict(state_dict)
    # # decoder = NICDecoder(vocab_size=tokenizer.vocab_size,embed_dim=1024,hidden_size=512)
    # # decoder.load_state_dict(state_dict)
    #
    #
    data_dir = "./data/flickr8k"
    # flickr_ds = Flickr8kDataset2(data_dir, split="test",feature_extractor="clip")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = NIC_BN(vocab_size=tokenizer.vocab_size,embed_dim=1024,hidden_size=512,feature_dim=512)
    model_path = "log/NICpokemon_dropout_bn_clip_512_emb_1024_hidden_512_Adam_0.0001/NIC_epoch_114_pokemon_dropout_bn_clip_512_emb_1024_hidden_512_Adam_0.0001.pth"
    state = torch.load(model_path)
    state_dict = state['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    test_ds = PokemonDataset(split="test", feature_extractor="clip")
    for i in range(6):
        finetune_vis(test_ds,model)

    # # for i in range(6):
    # #     eval_vis(flickr_ds,model)
    # img_path = r"D:\pycharmWorkspace\Show-And-Tell-442\test_img.jpg"
    # img = Image.open(img_path)
    # captions = inference([img],model,tokenizer,extractor_name="clip")
    # print(captions)

    # model.eval()
    #
    # data_pairs = flickr_ds.data_dict[flickr_ds.split]
    # fig,ax = plt.subplots(2,2,figsize=(20,20))
    #
    # ax = ax.ravel()
    # # fig.tight_layout()
    # sample_size = 4
    # samples = random.sample(range(len(data_pairs)),sample_size)
    # for i,n in enumerate(samples):
    #     # n = samples[i]
    #     print(n)
    #     img_path,txt = data_pairs[n]
    #     # print(img_path,txt)
    #     img = Image.open(img_path)
    #     img_ft, txt_ft, _ = flickr_ds[n]
    #     # print(img_ft.shape)
    #     pred_txt = beam_search2(model,flickr_ds.tokenizer,img_ft)
    #     print(txt,pred_txt)
    #     caption = f"Predicted: {pred_txt} \nCorrect: {txt}"
    #     # txt = "\n".join(wrap(,40)
    #     # "\n".join(wrap(caption, 40))
    #     ax[i].imshow(img)
    #     ax[i].axis("off")
    #     ax[i].set_title(caption,size = 16, wrap = True)
    # plt.show()


    # scores = eval2(flickr_ds,tokenizer,model)
    # print(scores)

    # ds = flickr_ds.data
    # batch1 = next(iter(ds))
    # captions = batch1["captions"]
    # img_path = batch1['image_path']
    # trans = get_img_transforms()
    # img_ft = extract_features(img, trans).unsqueeze(0)
    # print(img_ft)
    # captions = captions[:1]
    # captions_ft = tokenizer(captions,padding='max_length',max_length = 45,truncation=True)['input_ids']
    # captions_ft = torch.tensor(captions_ft).view(len(captions),-1)
    # print(captions_ft.shape)
    # h_img = encoder(img_ft)
    # outputs, h_txt = decoder(captions_ft,h_img)
    # print(outputs.shape,h_txt[0].shape)
    # h_img = h_img[0]
    # h_txt = h_txt[0]
    # sim = torch.cosine_similarity(h_img,h_txt)
    # print(sim)
    #
    # batch2 = next(iter(ds))
    # captions = batch2["captions"]
    # img_path = batch2['image_path']

    # img_path = r"D:\pycharmWorkspace\Show-And-Tell-442\data\flickr8k\Flicker8k_Dataset\2083778090_3aecaa11cc.jpg"
    # img = Image.open(img_path)
    # trans = get_img_transforms()
    # img_ft = extract_features(img, trans)
    # caption = greedy_search(model,tokenizer,img_ft)
    # print(caption)
    # captions = beam_search(model,tokenizer,img_ft,beam_width=10)
    # print(captions)
    # captions = beam_search2(model, tokenizer, img_ft, beam_width=10)
    # print(captions)

    #
    # # start = time()
    # captions = beam_search(encoder,decoder,tokenizer,img_ft,beam_width=20)
    # # end = time()
    # # print(f"Beam search took {end-start}")
    # print(f"Beam Search :{captions}")
    #
    # captions = beam_search2(encoder,decoder,tokenizer,img_ft,beam_width=20)
    # print(f"Beam Search 2:{captions}")

    # data_dir = "./data/flickr8k"
    # flickr_ds = Flickr8kDataset(data_dir,split="test")
    # res = eval(flickr_ds, tokenizer, model)
    # print(json.dumps(res,indent=4))




    # print(flickr_ds.data)
    # ds_test = flickr_ds.data
    # sample = next(iter(ds_test))
    # # print(sample)
    # captions = sample['captions']
    # img_path = sample['image_path']
    # print(captions)
    # img = Image.open(img_path)
    # img.show()
