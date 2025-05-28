import os
import pickle
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve

from nltk.stem import PorterStemmer
from PIL import Image
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

endpoint_url = "https://query.wikidata.org/sparql"

def get_results(endpoint_url, query):
    # SPARQL爬取结果
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_all_class(data_type = 'multi30k', lang = 'de'):
    with open("MMKG_MMT/datasets/multi30k-dataset/data/task1/raw/train.de", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [re.sub(r'[.,„(){}\d]','',i) for i in lines]
    words = set()
    for i in lines:
        for j in i.split():
            words.add(j)
    all_class = {}
    for item in tqdm(words):
        try:
            query = f"""SELECT DISTINCT ?item ?class
            WHERE 
            {{ 
            ?item ?label "{item}"@de. 
            ?item wdt:P279 ?class.
            }}"""
            results = get_results(endpoint_url, query)
            for result in results["results"]["bindings"]:
                i = result['class']['value'].split('/')[-1]
                if i in all_class.keys():
                    all_class[i] += 1
                else:
                    all_class[i] = 1
        except:
            continue
    with open(f"MMKG_MMT/datasets/mmkg-dataset/all_class_{data_type}_{lang}.pkl", "wb") as myprofile:  
        pickle.dump(all_class, myprofile)
  
def get_entity_label(data_type = 'multi30k', lang = 'de'):
    with open(f"MMKG_MMT/datasets/mmkg-dataset/all_class_{data_type}_{lang}.pkl", "rb") as f:  
        all_class = pickle.load(f)
    item_img = {}
    en_de = {}
    for i in tqdm(sorted(all_class.items(), key=lambda item:item[1],reverse=True)[:150]):
        try:
            class_id = i[0]# DISTINCT
            query = """
            SELECT DISTINCT ?item ?item_en ?item_de ?image
            WHERE {{
                ?item wdt:P31 wd:{}. # member state of the European Union
                OPTIONAL {{?item wdt:P18 ?image.}}.
                OPTIONAL {{?item rdfs:label ?item_en filter (lang(?item_en) = "en")}}.
                OPTIONAL {{?item rdfs:label ?item_de filter (lang(?item_de) = "de")}}.
            }}
            """.format(class_id)
            results = get_results(endpoint_url, query)
            if len(results["results"]["bindings"]) > 0:
                for result in tqdm(results["results"]["bindings"]):
                    try:
                        ent_en = result['item_en']['value']
                        if ent_en in item_img.keys():
                            continue
                        try:
                            # urlretrieve(result['image']['value'], f'wiki_en_de_image/{ent_en}.jpg')
                            # img = Image.open(f'wiki_en_de_image/{ent_en}.jpg').convert('RGB')  # 打开图片# error:wheelset
                            item_img[ent_en] = result['image']['value']
                            en_de[ent_en] = result['item_de']['value']
                        except:
                            continue
                    except:
                        continue
        except:
            continue
    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_img_{data_type}_{lang}.pkl", "wb") as f: # 存放item-img url的对应关系
        pickle.dump(item_img, f)
    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_{data_type}_en_{lang}.pkl", "wb") as f:# 存放entity en-de的翻译对
        pickle.dump(en_de, f)
    
def get_wiki_image(data_type = 'multi30k', lang = 'de'):
    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_img_{data_type}_{lang}.pkl", "rb") as f:  
        item_img = pickle.load(f)
    with open(f"wiki_ent_alter_{data_type}_{lang}.txt", "r", encoding="utf-8") as f:
        wiki_ent_alter = f.readlines()
        wiki_ent_alter = [l.replace("\n","").replace('\'','') for l in wiki_ent_alter]
    # wiki_ent_alter = ['blue triangle', 'blue', 'brown triangle', 'Caucasus', 'convolution', 'cornflower blue', 'Diagonal', 'dispersion']
    if not os.path.exists(f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}'):
        os.mkdir(f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}')
    for i in tqdm(wiki_ent_alter):# 1137
        try:
            urlretrieve(item_img[i], f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}/{i}.jpg')
            img = Image.open(f'wiki_new_image/{i}.jpg').convert('RGB')  
        except:
            # if error, then download again
            query = """SELECT DISTINCT ?itemLabel ?imageLabel
            WHERE 
            {{
            {{?item ?label "{}"@en. 
            ?item wdt:P18 ?image.}}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}""".format(i)
            results = get_results(endpoint_url, query)
            for result in results["results"]["bindings"]:
                try:
                    urlretrieve(result['imageLabel']['value'], f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}/{i}.jpg')
                    img = Image.open(f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}/{i}.jpg').convert('RGB')  # 打开图片# error:wheelset
                    break
                except:
                    if os.path.exists(f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}/{i}.jpg'):
                        os.remove(f'MMKG_MMT/datasets/mmkg-dataset/wiki_images_{data_type}_{lang}/{i}.jpg')
                    break

if __name__ == "__main__":
    data_type = 'multi30k', lang = 'de'
    get_all_class(data_type, lang)# Obtain the categories of all words in the source sentence
    get_entity_label(data_type, lang)# Obtain the img list of all entity words under the category and the corresponding translation pairs
    get_wiki_image(data_type, lang)# Obtain the img corresponding to the entity word