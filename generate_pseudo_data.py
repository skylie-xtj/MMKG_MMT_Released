# -*- coding:utf-8 -*-
import os
import os.path
import pickle
import random
import re
import string
import subprocess
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
import torch.cuda
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from PIL import Image
from spacy.matcher import Matcher, PhraseMatcher
from torch import nn, optim
from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm

random.seed(2022)


def remove_unsed_img():
    parent_dir = "data/wiki_image"
    filenames = os.listdir(parent_dir)
    # print([i.split(".jpg")[0] for i in list(set(filenames)-set(unused))])
    for i in tqdm(filenames):
        i = i.split(".jpg")[0]
        try:
            img = Image.open(f"{parent_dir}/{i}.jpg").convert(
                "RGB"
            )
            continue
        except:# Open img error
            try:
                os.remove(f"{parent_dir}/{i}.jpg")
            except:
                continue

def dir_img_embed(lang="fr", data_type="ikea"):
    for x in ["train", "test", "val"]:
        with open(
            f"IKEA-Dataset-master/IKEA/data.en.{lang}/image.index/{x}_WMT_2017_images.txt",
            "r",
            encoding="utf-8",
        ) as f:
            img_ids = f.readlines()
            img_ids = [sen.replace("\n", "") for sen in img_ids]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform1 = transforms.Compose(
            [  # integrate multiple image transformations
                # transforms.Resize(256),  # zoom
                # transforms.CenterCrop(224),  # ClipCenter
                # transforms.ToTensor()]  # Convert to Tensor
                # transforms.ToPILImage(),
                transforms.CenterCrop(512),
                transforms.Resize(448),
                transforms.ToTensor(),
            ]
        )
        data_list = []
        for i in tqdm(img_ids):
            try:
                if os.path.exists(
                    f"IKEA-Dataset-master/IKEA/image/image.en.{lang}/{x}.1/{i}"
                ):
                    path = f"IKEA-Dataset-master/IKEA/image/image.en.{lang}/{x}.1/{i}"
                else:
                    path = f"IKEA-Dataset-master/IKEA/image/image.en.{lang}/{x}.2/{i}"
                img = Image.open(path).convert("RGB") 
                # img = img.flatten()
                img1 = transform1(img)  # Perform transform1 operations on the image
                img1 = img1.reshape(1, 3, 448, 448)
                img1 = img1.to(device)
                resnet50_feature_extractor = models.resnet50(
                    pretrained=True
                )  # Import the pre-trained model of ResNet50
                resnet50_feature_extractor.fc = nn.Linear(2048, 2048)  # Redefine the last layer
                torch.nn.init.eye(
                    resnet50_feature_extractor.fc.weight
                )  # Initialize the two-dimensional tensor as the identity matrix
                resnet50_feature_extractor = resnet50_feature_extractor.to(device)
                for param in resnet50_feature_extractor.parameters():
                    param.requires_grad = False
                with torch.no_grad():
                    y = resnet50_feature_extractor(img1)
                data_list.append(
                    y.cpu().detach().numpy().reshape(-1)
                )  # Put the vectors extracted from each picture in the list one by one
            except:
                continue
        z = torch.Tensor(data_list)
        np.save(
            f"MMKG_MMT/feature_extractor/{x}-res50-avgpool-{data_type}_{lang}.npy", z
        ) 

def generate_img_embed(data_type, lang):
    data_list = []  # Define an empty list for storing data
    # path='data/wiki_en_de_image/'
    path = f"MMKG_MMT/datasets/wiki_{data_type}_en_{lang}_image/"  # Here is the path for storing pictures
    img_id = {}
    unused = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if data_type == "multi30k":
        img_npy = np.load(
            f"MMKG_MMT/feature_extractor/train-resnet50-avgpool-{data_type}_{lang}.npy"
        )
    elif data_type == "ikea":
        img_npy = np.load(
            f"MMKG_MMT/feature_extractor/train-res50-avgpool-{data_type}_{lang}.npy"
        )
    for i in img_npy:
        data_list.append(i)
    i = len(data_list)
    transform1 = transforms.Compose(
        [  
            # transforms.Resize(256),  # zoom
            # transforms.CenterCrop(224),  # ClipCenter
            # transforms.ToTensor()]  # Convert to Tensor
            # transforms.ToPILImage(),
            transforms.CenterCrop(512),
            transforms.Resize(448),
            transforms.ToTensor(),
        ]
    )
    for filename in tqdm(os.listdir(path)):  # Initialize the two-dimensional tensor as the identity matrix
        try:
            # print(filename)
            img_id[filename.split(".jpg")[0]] = i
            img = Image.open(path + filename).convert("RGB")  # error:wheelset
            # img = img.flatten()
            img1 = transform1(img)  
            img1 = img1.reshape(1, 3, 448, 448)
            img1 = img1.to(device)
            resnet50_feature_extractor = models.resnet50(
                pretrained=True
            )  # 导入ResNet50的预训练模型
            resnet50_feature_extractor.fc = nn.Linear(2048, 2048)  # Redefine the last layer
            torch.nn.init.eye(resnet50_feature_extractor.fc.weight)  # Initialize the two-dimensional tensor as the identity matrix
            resnet50_feature_extractor = resnet50_feature_extractor.to(device)
            for param in resnet50_feature_extractor.parameters():
                param.requires_grad = False
            with torch.no_grad():
                y = resnet50_feature_extractor(img1)
            data_list.append(
                y.cpu().detach().numpy().reshape(-1)
            )  # Put the vectors extracted from each picture in the list one by one
            i += 1
        except:
            unused.append(filename)
            continue
    z = torch.Tensor(data_list)
    np.save(
        f"MMKG_MMT/feature_extractor/wiki-res50-avgpool-{data_type}_{lang}.npy", z
    )  # 查看unused
    with open(
        f"MMKG_MMT/feature_extractor/wiki_img_id_{data_type}_{lang}.pkl", "wb"
    ) as f:
        pickle.dump(img_id, f, -1)

def generate_pseudo_img_embed(data_type, lang):
    embeding_weights = np.load(
        f"MMKG_MMT/feature_extractor/wiki-res50-avgpool-{data_type}_{lang}.npy"
    )
    if data_type == "multi30k":
        test_embed1 = np.load(
            "MMKG_MMT/feature_extractor/test_2016_flickr-resnet50-avgpool.npy"
        )
        test_embed2 = np.load(
            "MMKG_MMT/feature_extractor/test_2017_flickr-resnet50-avgpool.npy"
        )
        test_embed3 = np.load(
            "MMKG_MMT/feature_extractor/test_2017_coco-resnet50-avgpool.npy"
        )
        test_embed4 = np.load(
            "MMKG_MMT/feature_extractor/test_2018_flickr-resnet50-avgpool.npy"
        )
        val_embed = np.load("MMKG_MMT/feature_extractor/val-resnet50-avgpool.npy")
        test_embed = np.concatenate(
            (test_embed1, test_embed2, test_embed3, test_embed4, val_embed), axis=0
        )
    elif data_type == "ikea":
        test_embed = np.load(
            f"MMKG_MMT/feature_extractor/test-res50-avgpool-{data_type}_{lang}.npy"
        )
        val_embed = np.load(
            f"MMKG_MMT/feature_extractor/val-res50-avgpool-{data_type}_{lang}.npy"
        )
        test_embed = np.concatenate((test_embed, val_embed), axis=0)
    embeddings_matrix = np.zeros((len(embeding_weights) + 1, 2048))  
    embeddings_matrix[1:] = embeding_weights
    img_features = []
    with open(f"MMKG_MMT/data/data_process_{data_type}/new_img_id.txt", "r", encoding="utf-8") as f:
        new_img_id = f.readlines()
        new_img_id = [sen.replace("\n", "") for sen in new_img_id]
    for i in tqdm(new_img_id):
        tmp_img = []
        for j in i.split():
            tmp_img.append(embeddings_matrix[int(j)])
        tmp_img = np.array(tmp_img)
        tmp_img = np.mean(tmp_img, 0)
        img_features.append(tmp_img.flatten())
    img_features = np.array(img_features)
    img_features = np.concatenate((img_features, test_embed), axis=0)
    z = torch.FloatTensor(img_features)
    np.save(
        f"MMKG_MMT/feature_extractor/train-wiki-res50-avgpool-{data_type}_{lang}.npy", z
    )  # 1894
    data_size = len(z)  # 58166
    if data_type == "multi30k":
        with open(f"MMKG_MMT/data/data_process_{data_type}/train.vision.en", "w", encoding="utf-8") as f:
            for i in range(1, data_size - 1014 - 1071 - 461 - 2000 + 1):
                f.write(str(i) + "\n")
        with open(
            f"MMKG_MMT/data/data_process_{data_type}/test.2016.vision.en", "w", encoding="utf-8"
        ) as f:
            for i in range(
                data_size - 1014 - 1071 - 461 - 2000 + 1,
                data_size - 1014 - 1071 - 461 - 1000 + 1,
            ):
                f.write(str(i) + "\n")
        with open(
            f"MMKG_MMT/data/data_process_{data_type}/test.2017.vision.en", "w", encoding="utf-8"
        ) as f:
            for i in range(
                data_size - 1014 - 1071 - 461 - 1000 + 1,
                data_size - 1014 - 1071 - 461 + 1,
            ):
                f.write(str(i) + "\n")
        with open(
            f"MMKG_MMT/data/data_process_{data_type}/test.coco.vision.en", "w", encoding="utf-8"
        ) as f:
            for i in range(
                data_size - 1014 - 1071 - 461 + 1, data_size - 1014 - 1071 + 1
            ):
                f.write(str(i) + "\n")
        with open(
            f"MMKG_MMT/data/data_process_{data_type}/test.2018.vision.en", "w", encoding="utf-8"
        ) as f:
            for i in range(data_size - 1014 - 1071 + 1, data_size - 1014 + 1):
                f.write(str(i) + "\n")
        with open(f"MMKG_MMT/data/data_process_{data_type}/valid.vision.en", "w", encoding="utf-8") as f:
            for i in range(data_size - 1014 + 1, data_size + 1):
                f.write(str(i) + "\n")
    elif data_type == "ikea":
        with open(f"MMKG_MMT/data/data_process_{data_type}/train.vision.en", "w", encoding="utf-8") as f:
            for i in range(1, data_size - len(test_embed) + 1):
                f.write(str(i) + "\n")
        with open(f"MMKG_MMT/data/data_process_{data_type}/test.vision.en", "w", encoding="utf-8") as f:
            for i in range(
                data_size - len(test_embed) + 1, data_size - len(val_embed) + 1
            ):
                f.write(str(i) + "\n")
        with open(f"MMKG_MMT/data/data_process_{data_type}/valid.vision.en", "w", encoding="utf-8") as f:
            for i in range(data_size - len(val_embed) + 1, data_size + 1):
                f.write(str(i) + "\n")

def generate_gate_img_embed():
    train_embed = np.load(
        "Revisit-MMT-master-copy/feature_extractor/resnet50-avgpool.npy"
    )
    test_embed = np.load("feature_extractor/test_2018_flickr-resnet50-avgpool.npy")
    all_embed = np.concatenate((train_embed, test_embed), axis=0)
    z = torch.FloatTensor(all_embed)
    np.save(
        "Revisit-MMT-master-copy/feature_extractor/train-all-res50-avgpool.npy", z
    )  # 1894

def generate_vision_id():
    parent_dir = "Revisit-MMT-master-copy/data/multi30k-en-de/"
    with open("test.2018.vision.en", "w", encoding="utf-8") as f:
        for i in range(32475, 33546):
            f.write(str(i) + "\n")

def read_tok(data_type, lang1, lang2):  # return train list[sen1, sen2, ...]
    data_dir = {
        "multi30k": f"MMKG_MMT/datasets/multi30k-dataset/data/task1/tok/train.lc.norm.tok.{lang1}",
        "ikea": f"MMKG_MMT/datasets/IKEA-Dataset-master/IKEA/data.en.{lang2}/data.norm.tok.lc/train.norm.tok.lc.{lang1}",
    }
    with open(data_dir[data_type], "r", encoding="utf-8") as f:
        sen_en = f.readlines()
        sen_en = [sen.replace("\n", "") for sen in sen_en]
    return sen_en

def clean_dict(data_type, lang):  # Output clean src dict word: data/src_dict_en.txt
    sen_en = read_tok(data_type, "en", lang)
    vocab = set()
    for i in sen_en:
        words = i.split()
        for word in words:
            vocab.add(word)
    with open(
        f"MMKG_MMT/datasets/stopwords/stopwords-{lang}.txt", "r", encoding="utf-8"
    ) as f:
        stop_words = f.readlines()
        stop_words = [l.replace("\n", "").replace("'", "") for l in stop_words]
    clean_vocab = set()  # 9533
    for word in vocab:
        word = word.strip().lower()
        if word not in stop_words and word not in string.punctuation and len(word) >= 3:
            clean_vocab.add(word)
    with open(
        f"MMKG_MMT/data/data_process_{data_type}/src_dict_{data_type}_{lang}.txt", "w", encoding="utf-8"
    ) as f:
        for word in clean_vocab:
            f.write(word + "\n")

def preprocess_fastalign_data(data_type, lang):  
    data_dir = {
        "multi30k": f"MMKG_MMT/datasets/multi30k-dataset/data/task1/tok/train.lc.norm.tok.",
        "ikea": f"MMKG_MMT/datasets/IKEA-Dataset-master/IKEA/data.en.{lang}/data.norm.tok.lc/train.norm.tok.lc.",
    }
    with open(f"{data_dir[data_type]}en", "r", encoding="utf-8") as f:
        sen_en = f.readlines()
        sen_en = [sen.replace("\n", "") for sen in sen_en]
    with open(f"{data_dir[data_type]}{lang}", "r", encoding="utf-8") as f:
        sen_de = f.readlines()
        sen_de = [sen.replace("\n", "") for sen in sen_de]
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.en.{lang}", "w", encoding="utf-8") as f:
        for i in range(len(sen_en)):
            f.write(sen_en[i] + " ||| " + sen_de[i] + "\n")# concat two files using |||

def aug_sen_multi30k_jianjinshi(data_type, lang):
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.en", "r", encoding="utf-8") as f:
        sen_en = f.readlines()
        sen_en = [sen.replace("\n", "") for sen in sen_en]
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.de", "r", encoding="utf-8") as f:
        sen_de = f.readlines()
        sen_de = [sen.replace("\n", "") for sen in sen_de]
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/wiki_ent_match_{data_type}_{lang}.pkl", "rb"
    ) as f:
        match_dict = pickle.load(f)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/ent_{data_type}_en_{lang}.pkl", "rb"
    ) as f:
        ent_en_de = pickle.load(f)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/forward.align.{data_type}.{lang}",
        "r",
        encoding="utf-8",
    ) as f:
        align_id = f.readlines()
        align_id = [sen.replace("\n", "") for sen in align_id]
    alter_ent = []
    alter_en = []
    pStemmer = PorterStemmer()
    match_set = set(match_dict.keys())
    with open(
        f"MMKG_MMT/feature_extractor/wiki_img_id_{data_type}_{lang}.pkl", "rb"
    ) as f:
        img_id_dict = pickle.load(f)
    with open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.en", "w", encoding="utf-8"
    ) as f1, open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.{lang}", "w", encoding="utf-8"
    ) as f2, open(
        f"MMKG_MMT/data/data_process_{data_type}/new_img_id.txt", "w", encoding="utf-8"
    ) as f3:
        for j in tqdm(range(len(sen_en))):
            align = align_id[j].split()
            align_word = {}
            for a in align:
                i1, i2 = a.split("-")
                align_word[int(i1)] = int(i2)
            sen1 = " " + sen_en[j] + " "
            sen2 = " " + sen_de[j] + " "
            flag = False
            f1.write(sen1.strip() + "\n")
            f2.write(sen2.strip() + "\n")
            f3.write(str(j + 1) + "\n")
            alter_match = {}
            change_num = 0
            
            split_en = sen_en[j].split()
            split_de = sen_de[j].split()
            for n1 in range(len(sen_en[j].split())):
                noun_stem = pStemmer.stem(split_en[n1])
                img_id = [j + 1]
                for n in range(5):
                    if noun_stem in match_set and n1 in align_word.keys():
                        i = random.randint(
                            0, len(match_dict[noun_stem]) - 1
                        )  # int(math.floor(random.random() * len(match_dict[noun_stem])))
                        if (
                            match_dict[noun_stem][i][0] not in ent_en_de.keys()
                            or noun_stem == match_dict[noun_stem][i][0]
                            or len(match_dict[noun_stem][i][0].split()) > 3
                        ):
                            continue
                        alter_match[noun_stem] = match_dict[noun_stem][i][0]
                        split_en[n1] = alter_match[noun_stem]
                        alter_ent.append(split_en[n1])
                        flag = True
                        n2 = align_word[n1]
                        split_de[n2] = ent_en_de[
                            alter_match[noun_stem]
                        ]  # ent_en_de[noun].lower()
                        try:
                            img_id.append(img_id_dict[split_en[n1]])
                        except:
                            continue
                        if flag:
                            tmp_en = " ".join(split_en).lower() + "\n"
                            if tmp_en in alter_en:
                                continue
                            f1.write(tmp_en)
                            alter_en.append(tmp_en)
                            f2.write(" ".join(split_de).lower() + "\n")
                            f3.write(" ".join([str(i) for i in img_id]) + "\n")
                # change_num += 1
                # if change_num == 5: continue
    alter_ent = list(set(alter_ent))
    alter_ent = sorted(alter_ent, key=lambda i: len(i), reverse=True)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/wiki_ent_alter_{data_type}_{lang}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for token in tqdm(alter_ent):
            f.write(token + "\n")

def aug_sen_multi30k(data_type, lang):
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.alter.en", "r", encoding="utf-8") as f:
        sen_en = f.readlines()
        sen_en = [sen.replace("\n", "") for sen in sen_en]
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.alter.de", "r", encoding="utf-8") as f:
        sen_de = f.readlines()
        sen_de = [sen.replace("\n", "") for sen in sen_de]
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/wiki_ent_match_{data_type}_{lang}.pkl", "rb"
    ) as f:
        match_dict = pickle.load(f)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/ent_{data_type}_en_{lang}.pkl", "rb"
    ) as f:
        ent_en_de = pickle.load(f)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/forward.align.{data_type}.{lang}",
        "r",
        encoding="utf-8",
    ) as f:
        align_id = f.readlines()
        align_id = [sen.replace("\n", "") for sen in align_id]
    alter_ent = []
    alter_en = []
    pStemmer = PorterStemmer()
    match_set = set(match_dict.keys())
    with open(
        f"MMKG_MMT/feature_extractor/wiki_img_id_{data_type}_{lang}.pkl", "rb"
    ) as f:
        img_id_dict = pickle.load(f)
    if not os.path.exists(f"MMKG_MMT/data/data_process_{data_type}"):
        os.makedirs(f"MMKG_MMT/data/data_process_{data_type}")
    with open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.en", "w", encoding="utf-8"
    ) as f1, open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.{lang}", "w", encoding="utf-8"
    ) as f2, open(
        f"MMKG_MMT/data/data_process_{data_type}/new_img_id.txt", "w", encoding="utf-8"
    ) as f3:
        for j in tqdm(range(len(sen_en))):
            align = align_id[j].split()
            align_word = {}
            for a in align:
                i1, i2 = a.split("-")
                align_word[int(i1)] = int(i2)
            sen1 = " " + sen_en[j] + " "
            sen2 = " " + sen_de[j] + " "
            f1.write(sen1.strip() + "\n")
            f2.write(sen2.strip() + "\n")
            f3.write(str(j + 1) + "\n")
            alter_match = {}
           
            split_en = sen_en[j].split()
            split_de = sen_de[j].split()
            flag = False
            for n in range(1):
                for n1 in range(len(sen_en[j].split())):
                    img_id = [j + 1]
                    noun_stem = pStemmer.stem(split_en[n1])
                    if noun_stem in match_set and n1 in align_word.keys():
                        # i = 0
                        i = random.randint(
                            0, len(match_dict[noun_stem]) - 1
                        )  # int(math.floor(random.random() * len(match_dict[noun_stem])))
                        while match_dict[noun_stem][i][0] not in ent_en_de.keys() or noun_stem == match_dict[noun_stem][i][0] or \
                            len(match_dict[noun_stem][i][0].split()) > 3:
                            i = random.randint(
                                0, len(match_dict[noun_stem]) - 1
                            )
                            # i += 1
                        alter_match[noun_stem] = match_dict[noun_stem][i][0]
                        split_en[n1] = alter_match[noun_stem]
                        alter_ent.append(split_en[n1])
                        flag = True
                        n2 = align_word[n1]
                        split_de[n2] = ent_en_de[alter_match[noun_stem]]  # ent_en_de[noun].lower()
                        try:
                            img_id.append(img_id_dict[split_en[n1]])
                        except:
                            continue                    
                if flag:
                    tmp_en = " ".join(split_en).lower() + "\n"
                    if tmp_en in alter_en:
                        continue
                    f1.write(tmp_en)
                    alter_en.append(tmp_en)
                    f2.write(" ".join(split_de).lower() + "\n")
                    f3.write(" ".join([str(i) for i in img_id]) + "\n")
    alter_ent = list(set(alter_ent))
    alter_ent = sorted(alter_ent, key=lambda i: len(i), reverse=True)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/wiki_ent_alter_{data_type}_{lang}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for token in tqdm(alter_ent):
            f.write(token + "\n")

def simi_spacy_mmkb_src_wiki(
    data_type, lang, thresh=0.5
):  # Calculate the similarity between mmkb and src_word based on the subwords and match them: data/mmkb/wiki_ent_match.pkl wiki_ent_match_new.pkl
    pStemmer = PorterStemmer()
    src_sen = read_tok(data_type, "en", "en")  # 7953
    with open(
        f"MMKG_MMT/feature_extractor/wiki_img_id_{data_type}_{lang}.pkl", "rb"
    ) as f:
        img_id_dict = pickle.load(f)
    with open(
        "MMKG_MMT/datasets/stopwords/stopwords-en.txt",
        "r",
        encoding="utf-8",
    ) as f:
        stop_words = f.readlines()
        stop_words = [l.replace("\n", "").replace("'", "") for l in stop_words] + list(
            string.punctuation
        )
    src_ent = set()
    for i in src_sen:
        for j in i.split():
            if j not in stop_words and len(j) >= 3:
                src_ent.add(j)
    nlp = spacy.load("en_core_web_lg")
    count = 0
    match_dict = {}
    src_nlp = {}
    mmkb_nlp = {}
    pat = "\d+"
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/ent_{data_type}_en_{lang}.pkl", "rb"
    ) as f:
        mmkb_dict = pickle.load(f)
        tmp = set(mmkb_dict.keys())
        for i in tmp:
            if len(i.split()) > 1 or re.search(pat, i) is not None or i in stop_words or i not in img_id_dict.keys():
                mmkb_dict.pop(i)
    # spacy
    for token in tqdm(src_ent):  # 7953
        if token not in stop_words:
            src_nlp[token] = nlp(pStemmer.stem(token))
    for token in tqdm(list(mmkb_dict.keys())):  # 194189, 37152
        mmkb_nlp[token] = nlp(pStemmer.stem(token))
    # with open('data/mmkb/ent_en_de_nlp.pkl', 'wb') as f:# wiki_ent_match_new0
    #     pickle.dump(mmkb_nlp, f, -1)
    # with open('data/mmkb/src_nlp.pkl', 'wb') as f:# wiki_ent_match_new0
    #     pickle.dump(src_nlp, f, -1)
    # with open('data/mmkb/src_nlp.pkl', 'rb') as f:
    #     src_nlp = pickle.load(f)
    # with open('data/mmkb/ent_en_de_nlp.pkl', 'rb') as f:
    #     mmkb_nlp = pickle.load(f)
    for token1 in tqdm(src_nlp.keys()):
        if not src_nlp[token1].has_vector:
            # print(token1)
            count += 1
            continue
        simi = {}
        for token2 in mmkb_nlp.keys():
            simi[token2] = src_nlp[token1].similarity(mmkb_nlp[token2])
        # if pStemmer.stem(token1) in tmp_match_dict.keys():
        #     for j in tmp_match_dict[pStemmer.stem(token1)]:
        #         try:
        #             if len(j.split()) > 3 or pStemmer.stem(j) == pStemmer.stem(token1): continue
        #             simi[j] = src_nlp[token1].similarity(mmkb_nlp[j]) + 0.2
        #         except: continue
        simi = sorted(simi.items(), key=lambda item: item[1], reverse=True)
        i = 0
        try:
            if simi[i][1] < thresh:
                continue
            while i < len(simi):
                while pStemmer.stem(token1) == pStemmer.stem(simi[i][0].lower()):
                    i += 1
                if i >= len(simi):
                    break
                if simi[i][1] < thresh or simi[i][0] not in img_id_dict.keys():
                    break
                if token1 not in match_dict.keys():
                    match_dict[token1] = [simi[i]]
                else:
                    match_dict[token1].append(simi[i])
                i += 1
        except:
            continue
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/wiki_ent_match_{data_type}_{lang}.pkl", "wb"
    ) as f:  # wiki_ent_match_new0
        pickle.dump(match_dict, f, -1)

def gen_ratio_sen(data_type, lang, ratio):  # low-resource-multi30k
    sen_en = read_tok(data_type, "en", lang)
    sen_de = read_tok(data_type, lang, lang)
    num = int(len(sen_en) * ratio)
    sen_en = sen_en[:num]
    sen_de = sen_de[:num]
    with open(f"MMKG_MMT/data/data_process_{data_type}_{ratio}/new_img_id.txt", "w", encoding="utf-8") as f3, \
    open(
        f"MMKG_MMT/data/data_process_{data_type}_{ratio}/train.alter.en", "w", encoding="utf-8"
    ) as f1, open(
        f"MMKG_MMT/data/data_process_{data_type}_{ratio}/train.alter.{lang}", "w", encoding="utf-8"
    ) as f2:
        for j in tqdm(range(len(sen_en))):
            f1.write(sen_en[j] + "\n")
            f2.write(sen_de[j] + "\n")
            f3.write(str(j + 1) + "\n")

def aug_sen_mixgen(data_type, lang, ratio):  # mixgen
    sen_en = read_tok("en")
    sen_de = read_tok(lang)
    num = int(len(sen_en) * ratio)
    sen_en = sen_en[:num]
    sen_de = sen_de[:num]
    with open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.en", "w", encoding="utf-8"
    ) as f1, open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.{lang}", "w", encoding="utf-8"
    ) as f2, open(
        f"MMKG_MMT/data/data_process_{data_type}/new_img_id.txt", "w", encoding="utf-8"
    ) as f3:
        for j in tqdm(range(len(sen_en))):
            f1.write(sen_en[j] + "\n")
            f2.write(sen_de[j] + "\n")
            f3.write(str(j + 1) + "\n")
            tmp = [j]
            for n in range(4):
                i = random.randint(0, len(sen_en) - 1)
                while i in tmp:
                    i = random.randint(0, len(sen_en) - 1)
                tmp.append(i)
                f1.write(sen_en[j] + " " + sen_en[i] + "\n")
                f2.write(sen_de[j] + " " + sen_de[i] + "\n")
                f3.write(str(j + 1) + " " + str(i + 1) + "\n")

def aug_sen_ikea(data_type, lang):  # ikea
    multi = 5
    sen_en = read_tok(data_type, "en", lang)
    sen_de = read_tok(data_type, lang, lang)
    # num = int(len(sen_en))
    # sen_en = sen_en[:num]
    # sen_de = sen_de[:num]
    ent_match_type = f"wiki_ent_match_{data_type}_{lang}"
    with open(f"MMKG_MMT/datasets/mmkg-dataset/{ent_match_type}.pkl", "rb") as f:
        match_dict = pickle.load(f)
    # with open(f'data/ent_en_de.pkl', 'rb') as f:
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/ent_{data_type}_en_{lang}.pkl", "rb"
    ) as f:
        ent_en_de = pickle.load(f)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/forward.align.{data_type}.{lang}",
        "r",
        encoding="utf-8",
    ) as f:
        align_id = f.readlines()
        align_id = [sen.replace("\n", "") for sen in align_id]
    alter_ent = []
    # filenames = os.listdir("data/wiki_en_de_image")
    # filenames = [l.replace("\n","").replace('.jpg','') for l in filenames]153125
    pStemmer = PorterStemmer()
    match_set = set(match_dict.keys())
    # nlp = spacy.load("en_core_web_lg")
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/ent_img_{data_type}_{lang}.pkl", "rb"
    ) as f:  # item_img_
        img_id_dict = pickle.load(f)
    if not os.path.exists(f"MMKG_MMT/data/data_process_{data_type}"):
        os.makedirs(f"MMKG_MMT/data/data_process_{data_type}")
    with open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.en", "w", encoding="utf-8"
    ) as f1, open(
        f"MMKG_MMT/data/data_process_{data_type}/train.alter.{lang}", "w", encoding="utf-8"
    ) as f2, open(
        f"MMKG_MMT/data/data_process_{data_type}/new_img_id.txt", "w", encoding="utf-8"
    ) as f3:
        alter_sen = []
        for j in tqdm(range(len(sen_en))):
            align = align_id[j].split()
            align_word = {}
            for a in align:
                i1, i2 = a.split("-")
                align_word[int(i1)] = int(i2)
            sen1 = " " + sen_en[j] + " "
            sen2 = " " + sen_de[j] + " "
            flag = False
            f1.write(sen1.strip() + "\n")
            f2.write(sen2.strip() + "\n")
            f3.write(str(j + 1) + "\n")
            alter_match = {}
            alter_en = []
            alter_de = []
            alter_img = []
            for n1 in range(len(sen_en[j].split())):
                split_en = sen_en[j].split()
                split_de = sen_de[j].split()
                img_id = [j + 1]
                noun_stem = pStemmer.stem(split_en[n1])
                if noun_stem in match_set and n1 in align_word.keys():
                    i = random.randint(
                        0, len(match_dict[noun_stem]) - 1
                    )  # int(math.floor(random.random() * len(match_dict[noun_stem])))
                    if (
                        match_dict[noun_stem][i][0] not in ent_en_de.keys()
                        or noun_stem == match_dict[noun_stem][i][0]
                        or len(match_dict[noun_stem][i][0].split()) > 3
                    ):
                        continue
                    alter_match[noun_stem] = match_dict[noun_stem][i][0]
                    split_en[n1] = alter_match[noun_stem]
                    alter_ent.append(split_en[n1])
                    flag = True
                    # noun2 = ' '.join(set(align[noun])&set(split_de))
                    n2 = align_word[n1]
                    split_de[n2] = ent_en_de[
                        alter_match[noun_stem]
                    ]  # ent_en_de[noun].lower()
                    try:
                        img_id.append(img_id_dict[split_en[n1]])
                    except:
                        continue
                    if flag:
                        tmp_en = " ".join(split_en).lower() + "\n"
                        if tmp_en in alter_en:
                            continue
                        tmp_de = " ".join(split_de).lower() + "\n"
                        if tmp_de in alter_de:
                            continue
                    alter_en.append(tmp_en)
                    alter_de.append(tmp_de)
                    alter_img.append(" ".join([str(i) for i in img_id]) + "\n")
            if len(alter_en) == 0:
                continue
            for x in range(multi):
                i = random.randint(
                    0, len(alter_en) - 1
                )  # int(math.floor(random.random() * len(match_dict[noun_stem])))
                if alter_en[i] in alter_sen:
                    continue
                f1.write(alter_en[i])
                alter_sen.append(alter_en[i])
                f2.write(alter_de[i])
                f3.write(alter_img[i])
    alter_ent = list(set(alter_ent))
    alter_ent = sorted(alter_ent, key=lambda i: len(i), reverse=True)
    with open(
        f"MMKG_MMT/datasets/mmkg-dataset/wiki_ent_alter_{data_type}_{lang}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for token in tqdm(alter_ent):
            f.write(token + "\n")

def analysis_test_words_multi30k(lang):
    test_words = set()
    train_words = set()
    aug_train_words = set()
    test_dir = [
        "test_2016_flickr.lc.norm.tok.de",
        "test_2017_flickr.lc.norm.tok.de",
        "test_2017_mscoco.lc.norm.tok.de",
        "test_2018_flickr.lc.norm.tok.de",
    ]
    for dir in test_dir:
        with open(
            f"MMKG_MMT/datasets/multi30k-dataset/data/task1/tok/{dir}",
            "r",
            encoding="utf-8",
        ) as f:
            test_de = f.readlines()
            for i in test_de:
                for j in i.replace("\n", "").split():
                    test_words.add(j)
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.{lang}", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            for j in i.replace("\n", "").split():
                train_words.add(j)
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.alter.{lang}", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            for j in i.replace("\n", "").split():
                aug_train_words.add(j)
    print(len(train_words & test_words) / len(test_words))
    print(len(aug_train_words & test_words) / len(test_words))

def analysis_test_words_ikea(lang):
    test_words = set()
    train_words = set()
    aug_train_words = set()
    with open(
        f"MMKG_MMT/datasets/IKEA-Dataset-master/IKEA/data.en.{lang}/data.norm.tok.lc/test.norm.tok.lc.{lang}",
        "r",
        encoding="utf-8",
    ) as f:
        test_de = f.readlines()
        for i in test_de:
            for j in i.replace("\n", "").split():
                test_words.add(j)
    with open(
        f"MMKG_MMT/datasets/IKEA-Dataset-master/IKEA/data.en.{lang}/data.norm.tok.lc/train.norm.tok.lc.{lang}",
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.readlines()
        for i in lines:
            for j in i.replace("\n", "").split():
                train_words.add(j)
    with open(f"MMKG_MMT/data/data_process_{data_type}/train.alter.{lang}", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            for j in i.replace("\n", "").split():
                aug_train_words.add(j)
    print(len(train_words & test_words) / len(test_words))
    print(len(aug_train_words & test_words) / len(test_words))


if __name__ == "__main__":
    lang = "de" 
    data_type = "multi30k"
    # align
    preprocess_fastalign_data(data_type, lang)# Combine src data and tar data together
    command = "MMKG_MMT/utils/fast_align-master/build/fast_align -i MMKG_MMT/data/data_process/train.en.{} -d -o -v > MMKG_MMT/data/data_process/forward.align.{}.{}".format(lang, data_type, lang)
    subprocess.run(command, shell=True)
    
    # compute simi
    simi_spacy_mmkb_src_wiki(data_type, lang, 0.5)
    
    # other
    analysis_test_words_multi30k(lang)
    analysis_test_words_ikea(lang)
    
    # org img
    dir_img_embed()
    generate_img_embed(data_type, lang)

    # aug sen
    if data_type == "ikea":
        aug_sen_ikea("ikea", lang)# ikea
    if data_type == "multi30k":
        aug_sen_multi30k("multi30k", lang)  # multi30k
    # aug img    
    generate_pseudo_img_embed(data_type, lang)
    
    # if low-resource
    ratios = [0.1, 0.25, 0.5, 1]
    for ratio in ratios:
        gen_ratio_sen(data_type, lang, ratio=ratio)  

