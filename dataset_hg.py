import typing

import PIL.Image
import datasets
from datasets import DatasetDict
import torch
import torchvision.models
from torch.utils.data import Dataset
from pathlib import Path

from transformers import BertTokenizer, AutoTokenizer
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import collections
import typing
from PIL import Image
from transformers import BertTokenizer, AutoTokenizer, CLIPModel, CLIPProcessor
from tqdm import  tqdm
import pickle
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
device = torch.device(device)

class Flickr8kDataset(Dataset):
    def __init__(self,data_dir:Path,split="train",feature_extractor="inceptionv3"):
        '''
        :param data_dir: the folder that contains 2 zip files
        :param split: split of dataset
        :param feature_extractor: feature extractor of the dataset, options are:

        '''
        ds = datasets.load_dataset("atasoglu/flickr8k-dataset",data_dir=data_dir)
        splits = list(ds.keys())
        self.reorg_dir = Path("./reorg")
        self.pkl_dir = Path("./pkl")
        self.splits = splits
        self.img_features = {}
        self.txt_features = {}
        self.data_dict = {}

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.split = split
        self.data = ds

        '''
        reorg data into img - caption pairs and write to folder  
        '''
        for spl in splits:
            data_pairs = self.reorg_data(spl,self.reorg_dir)
            self.data_dict[spl] = data_pairs

        '''
        preprocess all splits data into pickle
        '''
        self.preprocess_data(self.pkl_dir)
        '''
        get data of the split finally
        '''
        self.img_data = self.img_features[split]
        self.txt_data = self.txt_features[split]
        self.split_data = self.data[self.split]

    def reorg_data(self,split:str="train",output_dir=None,overwrite=False):
        '''
        Reorganize the dataset. Each image-caption pair will be 1 item.
        :return:
        '''
        if output_dir and not overwrite:
            output_txt = output_dir.joinpath(split + ".txt")
            if output_txt.exists():
                '''Just read existing txt file if there exists one'''
                logging.info(f"Loading reorganized data for {split} split from {output_txt}")
                data_pairs = []
                with open(output_txt,'r') as f:
                    for line in f.readlines():
                        data_pair = line.strip().split("\t")
                        data_pairs.append(data_pair)
                return data_pairs

        ds = self.data.get(split)
        data_pairs = []
        for record in tqdm(ds):
            img_path = record['image_path']
            captions = record['captions']
            for caption in captions:
                data_pairs.append([img_path,caption])
        if output_dir:
            '''Save to text file if output path specified'''
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            output_txt = output_dir.joinpath(split+".txt")
            with open(output_txt,"w") as f:
                for pair in data_pairs:
                    line = "\t".join(pair)+"\n"
                    f.write(line)
        return data_pairs

    def preprocess_data(self,output_dir:Path=None,overwrite=False):
        '''
        Preprocess the raw dataset, generate
        1. feature vectors using img - caption pairs
        2. caption vectors using img - caption pairs & vocab
        if output_dir is specified, store both as pickle files
        :return:
        '''
        ft_extractor = self.get_feature_extractor().to(device)
        transforms = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            output_img = output_dir.joinpath("img_features.pkl")
            output_txt = output_dir.joinpath("txt_features.pkl")
            if output_img.exists() and output_txt.exists():
                logging.info(f"Loading pickled txt and img features from {output_txt} and {output_img}")
                self.img_features = pickle.load(open(output_img,'rb'))
                self.txt_features = pickle.load(open(output_txt,'rb'))
                return

        for split in tqdm(self.splits):

            self.img_features[split] = []
            self.txt_features[split] = []

            data_pairs = self.data_dict[split]
            for pair in tqdm(data_pairs):
                img_path, caption = pair
                img = Image.open(img_path)
                img_ft = self.extract_features(img,extractor=ft_extractor,transforms=transforms).detach().cpu()
                self.img_features[split].append(img_ft)

                txt_vec = self.caption_to_vector(caption)
                #
                self.txt_features[split].append(txt_vec)
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            pickle.dump(self.img_features,open(output_dir.joinpath("img_features.pkl"),'wb'))
            pickle.dump(self.txt_features, open(output_dir.joinpath("txt_features.pkl"), 'wb'))
        del ft_extractor # delete this to free VRAM


    def get_feature_extractor(self,extractor="inceptionv3") -> models.Inception3:
        if extractor == "inceptionv3":
            extractor = models.inception_v3(weights=models.Inception_V3_Weights)
            extractor.eval()
            extractor.dropout = nn.Identity()  # disable the dropout layer as well
            extractor.fc = nn.Identity()  # disable the fc layer

        return extractor

    def extract_features(self,img: Image,extractor:models.Inception3, transforms=None):
        '''
        :param img: A PIL Image
        :return: a feature vector (tensor)
        '''
        to_tensor = T.Compose([
            T.ToTensor(),
            T.Resize(299),  # inceptionv3 needs this
        ])
        img = to_tensor(img)
        if transforms:
            img = transforms(img)
        img = img.unsqueeze(0).to(device)
        res = extractor(img)
        return res.squeeze()

    def caption_to_vector(self,caption:str):
        tokenizer = self.tokenizer
        '''longest text was length 42'''
        tokens = tokenizer(caption,padding='max_length',max_length = 45,truncation=True)
        return tokens["input_ids"]
    def vector_to_caption(self, vector: torch.Tensor | list):
        tokenizer = self.tokenizer
        return tokenizer.decode(vector)
    def __len__(self):
        return len(self.img_data)
    def __getitem__(self, idx):
        caption_vec = self.txt_data[idx]
        caption_len = caption_vec.index(self.tokenizer.sep_token_id) - 1  # find caption vector length by end token position
        caption_vec = torch.tensor(caption_vec)
        input_caption_vec = torch.zeros_like(caption_vec)
        target_caption_vec = torch.zeros_like(caption_vec)

        input_caption_vec[:caption_len+1] = caption_vec[:caption_len+1] # exclude [SEP]
        target_caption_vec[:caption_len+1] = caption_vec[1:caption_len+2] # exclude [CLS]
        # return torch.tensor(self.img_data[idx]), torch.tensor(self.txt_data[idx])
        return self.img_data[idx], input_caption_vec, target_caption_vec
    

class Flickr8kDataset2(Dataset):
    def __init__(self,data_dir:Path,split="train",feature_extractor="inceptionv3"):
        '''
        :param data_dir: the folder that contains 2 zip files
        :param split: split of dataset
        :param feature_extractor: feature extractor of the dataset, options are:

        '''
        ds = datasets.load_dataset("atasoglu/flickr8k-dataset",data_dir=data_dir)
        splits = list(ds.keys())
        self.reorg_dir = Path("./reorg")
        self.pkl_dir = Path(f"./pkl")
        self.splits = splits
        self.img_features = {}
        self.txt_features = {}
        self.data_dict = {}

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.split = split
        self.data = ds

        '''
        reorg data into img - caption pairs and write to folder  
        '''
        for spl in splits:
            data_pairs = self.reorg_data(spl,self.reorg_dir)
            self.data_dict[spl] = data_pairs

        '''
        preprocess all splits data into pickle
        '''
        # self.preprocess_data(self.pkl_dir)
        self.preprocess_txt(output_dir=self.pkl_dir)
        self.preprocess_img(output_dir=self.pkl_dir,extractor=feature_extractor)

        '''
        get data of the split finally
        '''
        self.img_data = self.img_features[split]
        self.txt_data = self.txt_features[split]
        self.data = self.data[self.split]

    def reorg_data(self,split:str="train",output_dir=None,overwrite=False):
        '''
        Reorganize the dataset. Each image-caption pair will be 1 item.
        :return:
        '''
        if output_dir and not overwrite:
            output_txt = output_dir.joinpath(split + ".txt")
            if output_txt.exists():
                '''Just read existing txt file if there exists one'''
                logging.info(f"Loading reorganized data for {split} split from {output_txt}")
                data_pairs = []
                with open(output_txt,'r') as f:
                    for line in f.readlines():
                        data_pair = line.strip().split("\t")
                        data_pairs.append(data_pair)
                return data_pairs

        ds = self.data.get(split)
        data_pairs = []
        for record in tqdm(ds):
            img_path = record['image_path']
            captions = record['captions']
            for caption in captions:
                data_pairs.append([img_path,caption])
        if output_dir:
            '''Save to text file if output path specified'''
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            output_txt = output_dir.joinpath(split+".txt")
            with open(output_txt,"w") as f:
                for pair in data_pairs:
                    line = "\t".join(pair)+"\n"
                    f.write(line)
        return data_pairs

    def preprocess_txt(self,output_dir:Path=None,overwrite=False):
        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            output_txt = output_dir.joinpath("txt_features.pkl")
            if output_txt.exists():
                logging.info(f"Loading pickled txt features from {output_txt} ")
                self.txt_features = pickle.load(open(output_txt, 'rb'))
                return
        logging.info("Extracting text features...")
        for split in tqdm(self.splits):

            self.txt_features[split] = []

            data_pairs = self.data_dict[split]
            for pair in tqdm(data_pairs):
                img_path, caption = pair
                txt_vec = self.caption_to_vector(caption)
                self.txt_features[split].append(txt_vec)
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            pickle.dump(self.txt_features, open(output_dir.joinpath("txt_features.pkl"), 'wb'))


    def preprocess_img(self,output_dir:Path=None,overwrite=False,extractor="inceptionv3",batch_size=64):
        ft_extractor = self.get_feature_extractor(extractor).to(device)

        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            save_dir = output_dir.joinpath(extractor)

            output_img = save_dir.joinpath("img_features.pkl")
            if output_img.exists():
                logging.info(f"Loading pickled txt and img features from {output_img}")
                self.img_features = pickle.load(open(output_img, 'rb'))
                return
        print(ft_extractor)
        for split in tqdm(self.splits):
            self.img_features[split] = []
            data_pairs = self.data_dict[split]
            img_paths = [img_path for (img_path,caption) in data_pairs] # each
            # print(img_paths[:30])
            image_batches = [img_paths[i:i + batch_size] for i in range(0, len(img_paths), batch_size)]
            img_features = []
            for batch in tqdm(image_batches):
                batch_features = self._get_image_features(batch,ft_extractor,extractor).detach().cpu()
                # print(batch_features.shape)
                img_features.append(batch_features)
            all_features = torch.cat(img_features, dim=0)
            self.img_features[split] = all_features

            # for pair in tqdm(data_pairs):
            #     img_path, caption = pair
            #     img = Image.open(img_path)
            #     img_ft = self.extract_features(img, extractor=ft_extractor,extractor_name=extractor).detach().cpu()
            #     self.img_features[split].append(img_ft)

        if output_dir:
            output_dir = Path(output_dir)
            save_dir = output_dir.joinpath(extractor)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            pickle.dump(self.img_features, open(save_dir.joinpath("img_features.pkl"), 'wb'))
        del ft_extractor  # delete this to free VRAM

    def preprocess_data(self,output_dir:Path=None,overwrite=False):
        '''
        Preprocess the raw dataset, generate
        1. feature vectors using img - caption pairs
        2. caption vectors using img - caption pairs & vocab
        if output_dir is specified, store both as pickle files
        :return:
        '''
        ft_extractor = self.get_feature_extractor().to(device)
        transforms = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            output_img = output_dir.joinpath("img_features.pkl")
            output_txt = output_dir.joinpath("txt_features.pkl")
            if output_img.exists() and output_txt.exists():
                logging.info(f"Loading pickled txt and img features from {output_txt} and {output_img}")
                self.img_features = pickle.load(open(output_img,'rb'))
                self.txt_features = pickle.load(open(output_txt,'rb'))
                return

        for split in tqdm(self.splits):

            self.img_features[split] = []
            self.txt_features[split] = []

            data_pairs = self.data_dict[split]
            for pair in tqdm(data_pairs):
                img_path, caption = pair
                img = Image.open(img_path)
                img_ft = self.extract_features(img,extractor=ft_extractor,transforms=transforms).detach().cpu()
                self.img_features[split].append(img_ft)

                txt_vec = self.caption_to_vector(caption)
                #
                self.txt_features[split].append(txt_vec)
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            pickle.dump(self.img_features,open(output_dir.joinpath("img_features.pkl"),'wb'))
            pickle.dump(self.txt_features, open(output_dir.joinpath("txt_features.pkl"), 'wb'))
        del ft_extractor # delete this to free VRAM


    def get_feature_extractor(self,extractor_name="inceptionv3") -> models.Inception3:
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
                def to(self,device):
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
                def to(self,device):
                    self.clip_model = self.clip_model.to(device)
                    return self
            extractor = CLIPExtractor()
        elif extractor_name == "googlenet":
            extractor = models.googlenet(models.GoogLeNet_Weights)
            extractor.fc = nn.Identity()
            extractor.eval()
            # output 1 x 1024
        return extractor
    def _get_image_features(self,image_paths,extractor:models.Inception3,extractor_name="inceptionv3"):
        '''get image features for a batch'''
        # Load all the images in a batch
        img_size = 224
        if extractor_name == "inceptionv3":
            img_size = 299
        elif extractor_name == "clip":
            images = [Image.open(image_path) for image_path in image_paths]
            return extractor(images).squeeze()
        trans = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((img_size,img_size)),  # inceptionv3 needs this
        ])
        images = [trans(Image.open(image_path)) for image_path in image_paths]
        images = torch.stack(images,dim=0).to(device)
        # print(images.shape)
        res = extractor(images)
        return res.squeeze()
    def extract_features(self,img: Image,extractor:models.Inception3,extractor_name="inceptionv3",  transforms=None):
        '''
        :param img: A PIL Image
        :return: a feature vector (tensor)
        '''
        img_size = 224
        if extractor_name == "inceptionv3":
            img_size = 299
        trans = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(img_size),  # inceptionv3 needs this
        ])
        img = trans(img)
        if transforms:
            img = transforms(img)
        img = img.unsqueeze(0).to(device)
        res = extractor(img)
        return res.squeeze()

    def caption_to_vector(self,caption:str):
        tokenizer = self.tokenizer
        '''longest text was length 42'''
        tokens = tokenizer(caption,padding='max_length',max_length = 45,truncation=True)
        return tokens["input_ids"]
    def vector_to_caption(self, vector: torch.Tensor | list):
        tokenizer = self.tokenizer
        return tokenizer.decode(vector)
    def __len__(self):
        return len(self.img_data)
    def __getitem__(self, idx):
        caption_vec = self.txt_data[idx]
        caption_len = caption_vec.index(self.tokenizer.sep_token_id) - 1  # find caption vector length by end token position
        caption_vec = torch.tensor(caption_vec)
        input_caption_vec = torch.zeros_like(caption_vec)
        target_caption_vec = torch.zeros_like(caption_vec)

        input_caption_vec[:caption_len+1] = caption_vec[:caption_len+1] # exclude [SEP]
        target_caption_vec[:caption_len+1] = caption_vec[1:caption_len+2] # exclude [CLS]
        # return torch.tensor(self.img_data[idx]), torch.tensor(self.txt_data[idx])
        return self.img_data[idx], input_caption_vec, target_caption_vec



class ShinkaiDataset(Dataset):
    def __init__(self, split="train", feature_extractor="inceptionv3"):
        '''
        :param data_dir: the folder that contains 2 zip files
        :param split: split of dataset
        :param feature_extractor: feature extractor of the dataset, options are:

        '''
        ds = self.prepare_dataset()
        splits = list(ds.keys())
        self.master_dir = Path("./shinkai")
        self.reorg_dir = self.master_dir.joinpath("reorg")
        self.pkl_dir = self.master_dir.joinpath("pkl")
        self.splits = splits
        self.feature_extractor = feature_extractor

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.split = split
        self.data = ds
        self.img_features = {}
        self.txt_features = {}
        self.caption_max_len = self.get_caption_max_len(self.data)
        '''
        no reorg, this dataset only has PIL image and captions, one to one 
        '''
        # for spl in splits:
        #     data_pairs = self.reorg_data(spl, self.reorg_dir)
        #     self.data_dict[spl] = data_pairs

        '''
        preprocess all splits data into pickle
        '''
        self.preprocess_txt(output_dir=self.pkl_dir,max_len=self.caption_max_len)
        self.preprocess_img(output_dir=self.pkl_dir, extractor=feature_extractor)

        '''
        get data of the split finally
        '''
        self.img_data = self.img_features[split]
        self.txt_data = self.txt_features[split]
        self.split_data = self.data[self.split]
    def prepare_dataset(self):
        shinkai_ds = datasets.load_dataset("Fung804/makoto-shinkai-picture")
        train_test = shinkai_ds["train"].train_test_split(test_size=0.1,shuffle=False,seed=0)
        test_valid = train_test["test"].train_test_split(test_size=0.5,shuffle=False,seed=0)
        shinkai_ds_split = DatasetDict({
            "train": train_test["train"],
            "valid": test_valid["train"],
            "test": test_valid["test"],
        })
        print(shinkai_ds_split)
        return  shinkai_ds_split

    def get_caption_max_len(self,data:DatasetDict):
        train_split = data['train']
        max_len = 0
        for item in train_split:
            caption = item['text']
            input_ids = self.tokenizer(caption)["input_ids"]
            max_len = max(max_len,len(input_ids))
        return max_len

    def preprocess_txt(self, output_dir: Path = None, overwrite=False, max_len = 45):
        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            output_txt = output_dir.joinpath("txt_features.pkl")
            if output_txt.exists():
                logging.info(f"Loading pickled txt features from {output_txt} ")
                self.txt_features = pickle.load(open(output_txt, 'rb'))
                return
        logging.info("Extracting text features...")
        for split in tqdm(self.splits):
            ds_split = self.data[split]
            self.txt_features[split] = []
            for idx,item in tqdm(enumerate(ds_split)):
                caption = item['text']
                txt_vec = self.caption_to_vector(caption,max_len=max_len)
                self.txt_features[split].append(txt_vec)
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            pickle.dump(self.txt_features, open(output_dir.joinpath("txt_features.pkl"), 'wb'))

    def preprocess_img(self, output_dir: Path = None, overwrite=False, extractor="inceptionv3", batch_size=64):
        ft_extractor = self.get_feature_extractor(extractor).to(device)

        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            save_dir = output_dir.joinpath(extractor)

            output_img = save_dir.joinpath("img_features.pkl")
            if output_img.exists():
                logging.info(f"Loading pickled txt and img features from {output_img}")
                self.img_features = pickle.load(open(output_img, 'rb'))
                return
        print(ft_extractor)
        for split in tqdm(self.splits):
            self.img_features[split] = []
            ds_split = self.data[split]
            imgs = [ item['image'] for item in ds_split]  # each
            # print(img_paths[:30])
            image_batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
            img_features = []
            for batch in tqdm(image_batches):
                batch_features = self._get_image_features(batch, ft_extractor, extractor).detach().cpu()
                # print(batch_features.shape)
                img_features.append(batch_features)
            all_features = torch.cat(img_features, dim=0)
            self.img_features[split] = all_features


        if output_dir:
            output_dir = Path(output_dir)
            save_dir = output_dir.joinpath(extractor)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            pickle.dump(self.img_features, open(save_dir.joinpath("img_features.pkl"), 'wb'))
        del ft_extractor  # delete this to free VRAM


    def get_feature_extractor(self, extractor_name="inceptionv3") -> models.Inception3:
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

    def _get_image_features(self, imgs, extractor: models.Inception3, extractor_name="inceptionv3"):
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

    def extract_features(self, img: Image, extractor: models.Inception3, extractor_name="inceptionv3", transforms=None):
        '''
        :param img: A PIL Image
        :return: a feature vector (tensor)
        '''
        img_size = 224
        if extractor_name == "inceptionv3":
            img_size = 299
        trans = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(img_size),  # inceptionv3 needs this
        ])
        img = trans(img)
        if transforms:
            img = transforms(img)
        img = img.unsqueeze(0).to(device)
        res = extractor(img)
        return res.squeeze()

    def caption_to_vector(self, caption: str, max_len = 45):
        tokenizer = self.tokenizer
        '''longest text was length 42'''
        tokens = tokenizer(caption, padding='max_length', max_length=max_len, truncation=True)
        return tokens["input_ids"]

    def vector_to_caption(self, vector: torch.Tensor | list):
        tokenizer = self.tokenizer
        return tokenizer.decode(vector)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        caption_vec = self.txt_data[idx]
        caption_len = caption_vec.index(
            self.tokenizer.sep_token_id) - 1  # find caption vector length by end token position
        caption_vec = torch.tensor(caption_vec)
        input_caption_vec = torch.zeros_like(caption_vec)
        target_caption_vec = torch.zeros_like(caption_vec)

        input_caption_vec[:caption_len + 1] = caption_vec[:caption_len + 1]  # exclude [SEP]
        target_caption_vec[:caption_len + 1] = caption_vec[1:caption_len + 2]  # exclude [CLS]
        # return torch.tensor(self.img_data[idx]), torch.tensor(self.txt_data[idx])
        return self.img_data[idx], input_caption_vec, target_caption_vec

class PokemonDataset(Dataset):
    def __init__(self, split="train", feature_extractor="inceptionv3"):
        '''
        :param data_dir: the folder that contains 2 zip files
        :param split: split of dataset
        :param feature_extractor: feature extractor of the dataset, options are:

        '''
        ds = self.prepare_dataset()
        splits = list(ds.keys())
        self.master_dir = Path("./pokemon")
        self.reorg_dir = self.master_dir.joinpath("reorg")
        self.pkl_dir = self.master_dir.joinpath("pkl")
        self.splits = splits
        self.feature_extractor = feature_extractor

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.split = split
        self.data = ds
        self.img_features = {}
        self.txt_features = {}
        self.caption_max_len = self.get_caption_max_len(self.data)
        '''
        no reorg, this dataset only has PIL image and captions, one to one 
        '''
        # for spl in splits:
        #     data_pairs = self.reorg_data(spl, self.reorg_dir)
        #     self.data_dict[spl] = data_pairs

        '''
        preprocess all splits data into pickle
        '''
        self.preprocess_txt(output_dir=self.pkl_dir,max_len=self.caption_max_len)
        self.preprocess_img(output_dir=self.pkl_dir, extractor=feature_extractor)

        '''
        get data of the split finally
        '''
        self.img_data = self.img_features[split]
        self.txt_data = self.txt_features[split]
        self.split_data = self.data[self.split]
    def prepare_dataset(self):
        shinkai_ds = datasets.load_dataset("lambdalabs/pokemon-blip-captions")
        train_test = shinkai_ds["train"].train_test_split(test_size=0.1,shuffle=False,seed=0)
        test_valid = train_test["test"].train_test_split(test_size=0.5,shuffle=False,seed=0)
        shinkai_ds_split = DatasetDict({
            "train": train_test["train"],
            "valid": test_valid["train"],
            "test": test_valid["test"],
        })
        print(shinkai_ds_split)
        return  shinkai_ds_split

    def get_caption_max_len(self,data:DatasetDict):
        train_split = data['train']
        max_len = 0
        for item in train_split:
            caption = item['text']
            input_ids = self.tokenizer(caption)["input_ids"]
            max_len = max(max_len,len(input_ids))
        return max_len

    def preprocess_txt(self, output_dir: Path = None, overwrite=False, max_len = 45):
        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            output_txt = output_dir.joinpath("txt_features.pkl")
            if output_txt.exists():
                logging.info(f"Loading pickled txt features from {output_txt} ")
                self.txt_features = pickle.load(open(output_txt, 'rb'))
                return
        logging.info("Extracting text features...")
        for split in tqdm(self.splits):
            ds_split = self.data[split]
            self.txt_features[split] = []
            for idx,item in tqdm(enumerate(ds_split)):
                caption = item['text']
                txt_vec = self.caption_to_vector(caption,max_len=max_len)
                self.txt_features[split].append(txt_vec)
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            pickle.dump(self.txt_features, open(output_dir.joinpath("txt_features.pkl"), 'wb'))

    def preprocess_img(self, output_dir: Path = None, overwrite=False, extractor="inceptionv3", batch_size=64):
        ft_extractor = self.get_feature_extractor(extractor).to(device)

        if output_dir and not overwrite:
            '''Just read from existing file if exists any '''
            output_dir = Path(output_dir)
            save_dir = output_dir.joinpath(extractor)

            output_img = save_dir.joinpath("img_features.pkl")
            if output_img.exists():
                logging.info(f"Loading pickled txt and img features from {output_img}")
                self.img_features = pickle.load(open(output_img, 'rb'))
                return
        print(ft_extractor)
        for split in tqdm(self.splits):
            self.img_features[split] = []
            ds_split = self.data[split]
            imgs = [ item['image'] for item in ds_split]  # each
            # print(img_paths[:30])
            image_batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
            img_features = []
            for batch in tqdm(image_batches):
                batch_features = self._get_image_features(batch, ft_extractor, extractor).detach().cpu()
                # print(batch_features.shape)
                img_features.append(batch_features)
            all_features = torch.cat(img_features, dim=0)
            self.img_features[split] = all_features


        if output_dir:
            output_dir = Path(output_dir)
            save_dir = output_dir.joinpath(extractor)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            pickle.dump(self.img_features, open(save_dir.joinpath("img_features.pkl"), 'wb'))
        del ft_extractor  # delete this to free VRAM


    def get_feature_extractor(self, extractor_name="inceptionv3") -> models.Inception3:
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

    def _get_image_features(self, imgs, extractor: models.Inception3, extractor_name="inceptionv3"):
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

    def extract_features(self, img: Image, extractor: models.Inception3, extractor_name="inceptionv3", transforms=None):
        '''
        :param img: A PIL Image
        :return: a feature vector (tensor)
        '''
        img_size = 224
        if extractor_name == "inceptionv3":
            img_size = 299
        trans = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(img_size),  # inceptionv3 needs this
        ])
        img = trans(img)
        if transforms:
            img = transforms(img)
        img = img.unsqueeze(0).to(device)
        res = extractor(img)
        return res.squeeze()

    def caption_to_vector(self, caption: str, max_len = 45):
        tokenizer = self.tokenizer
        '''longest text was length 42'''
        tokens = tokenizer(caption, padding='max_length', max_length=max_len, truncation=True)
        return tokens["input_ids"]

    def vector_to_caption(self, vector: torch.Tensor | list):
        tokenizer = self.tokenizer
        return tokenizer.decode(vector)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        caption_vec = self.txt_data[idx]
        caption_len = caption_vec.index(
            self.tokenizer.sep_token_id) - 1  # find caption vector length by end token position
        caption_vec = torch.tensor(caption_vec)
        input_caption_vec = torch.zeros_like(caption_vec)
        target_caption_vec = torch.zeros_like(caption_vec)

        input_caption_vec[:caption_len + 1] = caption_vec[:caption_len + 1]  # exclude [SEP]
        target_caption_vec[:caption_len + 1] = caption_vec[1:caption_len + 2]  # exclude [CLS]
        # return torch.tensor(self.img_data[idx]), torch.tensor(self.txt_data[idx])
        return self.img_data[idx], input_caption_vec, target_caption_vec


if __name__ == "__main__":
    max_len = 0
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # shinkai_ds = datasets.load_dataset("parquet",data_files={"train":"data/shinkai/shinkai.parquet"})
    hg_repo = "ayoubkirouane/One-Piece-anime-captions"
    one_piece_ds = datasets.load_dataset(hg_repo)
    # print(shinkai_ds)
    # print(shinkai_ds["train"][0])
    img, text = one_piece_ds["train"][0]
    train_test = one_piece_ds["train"].train_test_split(test_size=0.1)
    test_valid = train_test["test"].train_test_split(test_size=0.5)
    onepiece_ds_split = DatasetDict({
        "train":train_test["train"],
        "valid":test_valid["train"],
        "test":test_valid["test"],
    })
    print(onepiece_ds_split['train'][0])
    exit(0)
    shinkai_ds = datasets.load_dataset("Fung804/makoto-shinkai-picture")
    print(shinkai_ds)
    # print(shinkai_ds["train"][0])
    img, text = shinkai_ds["train"][0]
    train_test = shinkai_ds["train"].train_test_split(test_size=0.1)
    test_valid = train_test["test"].train_test_split(test_size=0.5)
    shinkai_ds_split = DatasetDict({
        "train":train_test["train"],
        "valid":test_valid["train"],
        "test":test_valid["test"],
    })
    # print(shinkai_ds_split)
    # for item in shinkai_ds_split["train"]:
    #     print(item)
    #     txt = item['text']
    #     input_ids = tokenizer(txt)['input_ids']
    #     print(txt)
    #     print(input_ids)
    #     max_len = max(max_len,len(input_ids))
    # print(max_len)
    print(shinkai_ds_split['train'][0])

    # pokemon_ds = datasets.load_dataset("lambdalabs/pokemon-blip-captions")
    # print(pokemon_ds)
    # # img, txt = pokemon_ds['train']
    # # print(img,txt)
    #
    # for item in pokemon_ds['train']:
    #     print(item)
    # train_test = pokemon_ds["train"].train_test_split(test_size=0.1)
    # test_valid = train_test["test"].train_test_split(test_size=0.5)
    # pokemon_ds_split = DatasetDict({
    #     "train":train_test["train"],
    #     "valid":test_valid["train"],
    #     "test":test_valid["test"],
    # })
    # print(pokemon_ds_split)


    # data_dir = "./data/flickr8k"
    # flickr_ds = Flickr8kDataset(data_dir)
    # # flickr_ds.preprocess_data(flickr_ds.pkl_dir,True)
    # batch = next(iter(flickr_ds) )
    #
    # for input1, input2 ,target in flickr_ds:
    #     print(input2.shape, target.shape)
    #     print(input2)
    #     print(target)
    #     print(flickr_ds.tokenizer.decode(input2))
    # exit(0)

    # print("Loading text vectors")
    # txt_ft = flickr_ds.pkl_dir.joinpath("txt_features.pkl")
    # txt_ft = pickle.load(open(txt_ft,'rb'))
    # print(txt_ft.keys())
    # vecs = txt_ft['train']
    # vec = vecs[0]
    # captions = flickr_ds.data_dict['train']
    # print(captions[0])
    # img_path, caption = captions[0]
    # decoded = flickr_ds.tokenizer.decode(vec,skip_special_tokens=True)
    # print(vec)
    # print(caption)
    # print(decoded)
    # print(len(flickr_ds))
    # vec = flickr_ds.caption_to_vector(caption)
    # print(len(vec))
    # print(flickr_ds.tokenizer.decode(vec,skip_special_tokens=True))
    #
    # # example = next(iter(flickr_ds))
    # # print(example)
    # # img_ft, txt_ft = example
    # # print(img_ft.shape,txt_ft.shape)
    # # max_len = 0
    # # for img_ft,txt_ft in flickr_ds:
    # #     # print(txt_ft.shape)
    # #     max_len = max(txt_ft.shape[0],max_len)
    # #     print(max_len)
    # exit(0)
    # # flickr_ds.preprocess_data()
    # # pairs = flickr_ds.reorg_data(output_dir="./test")[:10]
    # # print(pairs)
    # pairs = flickr_ds.data_dict['train'][:10]
    # pair = pairs[0]
    # img, caption = pair
    # img = PIL.Image.open(img)
    # ft = utils.extract_features(img)
    # print(ft)
    # caption = "I have a new GPU!"
    # # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # # tokens = tokenizer.tokenize(caption,add_special_tokens=False)
    # # # tokens = utils.tokenize_caption(caption)
    # # print(tokens)
    # # encoded = tokenizer.encode(tokens)
    # # print(encoded)
    # # decoded = tokenizer.decode(encoded)
    # # print(decoded)
    # print(utils.caption_to_vector(caption))
    # captions = [
    #     "I have a new GPU!",
    #     "Save to text file if output path specified",
    # ]
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # res = utils.caption_to_vector(captions)
    # res = torch.tensor(res)
    # print(res)
    # for sent in res:
    #     print(tokenizer.decode(sent))
    # # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    # # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # # tokens = tokenizer(captions)
    # # print(tokens)
    # # # img.show()
    # # with open("./test/train.txt",'r') as f:
    # #     for line in f.readlines()[:10]:
    # #         line = line.split("\t")
    # #         print(line)