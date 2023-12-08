import torch
import torch.nn as nn
import torch.functional as F
from transformers import BertTokenizer, AutoTokenizer, BertModel
from torchtext.vocab import GloVe
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
class NIC(nn.Module):
    def __init__(self,vocab_size,embed_dim=512,hidden_size=512,feature_dim=2048):
        super().__init__()
        logging.info(f"Creating NIC model with vocab size {vocab_size} embed dim {embed_dim} hidden size {hidden_size}")
        self.vocab_size = vocab_size
        '''
        image feature vec: 2048 x 1 
        word vec: vocab size x 1 
        1. take in 2 vectors, transform them to the same length embedding
           use 512 here according to the paper 
        
        '''
        self.word_emb = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.img_emb = nn.Linear(in_features=feature_dim,out_features=embed_dim)
        '''
        2. pass the img vector as a hidden state (x^-1) 
        '''
        self.lstm = nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,batch_first=True)
        '''
        3. transform passed txt vector to vocab_size 
        '''
        self.fc = nn.Linear(in_features=hidden_size,out_features=vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout()

    def forward(self,img_ft,txt_ft):
        '''
        1. Get img embedding and txt embedding
        '''
        img_emb = self.img_emb(img_ft)
        # img_emb = self.bn(img_emb)
        # img_emb = self.dropout(img_emb)
        txt_emb = self.word_emb(txt_ft)
        # txt_emb = self.dropout(txt_emb)
        '''
        2. pass image through lstm to get initial hidden states 
        '''
        img_emb = img_emb.unsqueeze(1)
        out_img, h_img = self.lstm(img_emb)
        # print("Text emb:",txt_emb.shape)
        # print("h_img:",h_img[0].shape,h_img[1].shape)
        # print(txt_emb.shape,h_img.shape)
        out_txt, h_txt = self.lstm(txt_emb,h_img) # lstm pred next token for all tokens given
        '''
        3. transform predicted embedding back to one-hot encodings 
        '''
        output = self.fc(out_txt)
        output = output.view(-1,self.vocab_size)
        return output
    def encode(self,img_ft):
        img_emb = self.img_emb(img_ft)
        # img_emb = self.bn(img_emb)
        img_emb = img_emb.unsqueeze(1)
        out_img, h_img = self.lstm(img_emb)
        return h_img
    def decode(self,txt_ft,h_img):
        txt_emb = self.word_emb(txt_ft)
        out_txt, h_txt = self.lstm(txt_emb, h_img)  # lstm pred next token for all tokens given
        output = self.fc(out_txt)
        output = self.softmax(output)
        return output, h_txt

class NIC_BN(NIC):
    def __init__(self,vocab_size,embed_dim=512,hidden_size=512,feature_dim=2048):
        super().__init__(vocab_size,embed_dim,hidden_size,feature_dim)
        self.bn = nn.BatchNorm1d(num_features=embed_dim)
    def forward(self,img_ft,txt_ft):
        '''
        1. Get img embedding and txt embedding
        '''
        img_emb = self.img_emb(img_ft)
        img_emb = self.bn(img_emb)
        img_emb = self.dropout(img_emb)
        txt_emb = self.word_emb(txt_ft)
        txt_emb = self.dropout(txt_emb)
        '''
        2. pass image through lstm to get initial hidden states 
        '''
        img_emb = img_emb.unsqueeze(1)
        out_img, h_img = self.lstm(img_emb)
        # print(txt_emb.shape,h_img[0].shape)
        out_txt, h_txt = self.lstm(txt_emb,h_img) # lstm pred next token for all tokens given
        '''
        3. transform predicted embedding back to one-hot encodings 
        '''
        output = self.fc(out_txt)
        output = output.view(-1,self.vocab_size)
        return output
    def encode(self,img_ft):
        img_emb = self.img_emb(img_ft)
        img_emb = self.bn(img_emb)
        img_emb = img_emb.unsqueeze(1)
        out_img, h_img = self.lstm(img_emb)
        return h_img
    def decode(self,txt_ft,h_img):
        txt_emb = self.word_emb(txt_ft)
        out_txt, h_txt = self.lstm(txt_emb, h_img)  # lstm pred next token for all tokens given
        output = self.fc(out_txt)
        output = self.softmax(output)
        return output, h_txt

class NIC_BERTEmb(NIC_BN):
    '''Same as NIC_BN but initializes word embeddings with bert embedding'''
    def __init__(self,vocab_size,embed_dim=512,hidden_size=512,feature_dim=2048):
        super().__init__(vocab_size,embed_dim,hidden_size,feature_dim)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model: BertModel = BertModel.from_pretrained("bert-base-uncased")
        embed_mat = model.embeddings.word_embeddings.weight
        self.word_emb.weight = embed_mat





# class NICEncoder(NIC):
#     def __init__(self,vocab_size,embed_dim=512,hidden_size=512):
#         super().__init__(vocab_size,embed_dim,hidden_size)
#     def forward(self,img_ft):
#         '''Get the hidden state from img feature'''
#         img_emb = self.img_emb(img_ft)
#         img_emb = self.bn(img_emb)
#         img_emb = img_emb.unsqueeze(1)
#         out_img, h_img = self.lstm(img_emb)
#         return h_img
# class NICDecoder(NIC):
#     def __init__(self,vocab_size,embed_dim=512,hidden_size=512):
#         super().__init__(vocab_size,embed_dim,hidden_size)
#     def forward(self,txt_ft,h_img):
#         txt_emb = self.word_emb(txt_ft)
#         # print(txt_emb.shape,h_img.shape)
#         out_txt, h_txt = self.lstm(txt_emb, h_img)  # lstm pred next token for all tokens given
#         '''
#         3. transform predicted embedding back to one-hot encodings
#         '''
#         output = self.fc(out_txt)
#         output = self.softmax(output)
#         return output,h_txt


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer:AutoTokenizer
# tokenizer:BertTokenizer
# print(len(tokenizer))
#
# print(tokenizer.all_special_tokens)
# print(tokenizer.sep_token,tokenizer.sep_token_id)
# print(tokenizer.cls_token,tokenizer.cls_token_id)