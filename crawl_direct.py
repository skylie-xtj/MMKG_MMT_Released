from SPARQLWrapper import SPARQLWrapper, JSON
import os
import pickle
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve
import time
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm
import threading
import requests
# Create a global lock
lock = threading.Lock()

def get_random_proxy():
    """
    get random proxy from proxypool
    :return: proxy
    """
    proxypool_url = 'http://x.x.x.x:x/random'
    return {'http': 'http://' + requests.get(proxypool_url).text.strip()}

def get_similar_entities(input_word, tar_lang, lang):
    # Set the SPARQL endpoint
    endpoint_url = "https://query.wikidata.org/sparql"
    proxy = get_random_proxy()
    # SPARQL query template
    query = f"""
    SELECT DISTINCT ?similarEntity ?similarEntityLabel ?image ?germanLabel WHERE {{
      # 1. Find the relevant entities through the input words
      ?entity rdfs:label "{input_word}"@{lang}.
     
      # 2. Search for the category or instance of this entity
      ?entity (wdt:P279|wdt:P31) ?class.
     
      # 3. Search for other entities belonging to the same category
      ?similarEntity (wdt:P279|wdt:P31) ?class.
      FILTER(?similarEntity != ?entity).

      # 4. Make sure that the returned entity has images
      ?similarEntity wdt:P18 ?image.

      # 5. Get translation
      ?similarEntity rdfs:label ?germanLabel.
      FILTER(LANG(?germanLabel) = "{tar_lang}").

      # 6. Obtain tags in other languages
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],{lang}". }}
    }}
    LIMIT 1000
    """

    # Set SPARQLWrapper
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    retry = True
    while retry:
        retry = False
        try:
            # make requests using a proxy
            sparql.setRequestHeader("User-Agent", "Mozilla/5.0")  # set User-Agent
            sparql.setRequestHeader("Proxy", proxy['http'])  # set proxy
            
            # execute query
            results = sparql.query().convert()
            # analyse result
            for result in results["results"]["bindings"]:
                if lang == 'en':
                    similar_entity_label = result["similarEntityLabel"]["value"]
                    image_url = result["image"]["value"]
                    german_label = result["germanLabel"]["value"]
                else:
                    similar_entity_label = result["germanLabel"]["value"]
                    image_url = result["image"]["value"]
                    german_label = result["similarEntityLabel"]["value"]
                if similar_entity_label in ent_trans_img.keys() or not re.match(r'^[^\d\W_]+$', similar_entity_label):
                    continue
                # Use locks to synchronize the modifications to the shared dictionary
                with lock:
                    ent_trans_img[similar_entity_label] = (german_label, image_url)
                    collected[input_word] = similar_entity_label
        except Exception as e:
            retry = True
            wait_time = 1  # Set the waiting time
            time.sleep(wait_time)
            print(f"Error querying Wikidata: {e}")
    
    # save file
    with open(f"MMKG_MMT/datasets/mmkg-dataset/collected_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", "wb") as f:
        pickle.dump(collected, f)
    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_trans_img_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", "wb") as f:
        pickle.dump(ent_trans_img, f)

def get_data_words(data_type = 'multi30k', tar_lang = 'de', lang = 'en'):# Obtain the categories from the Wiki based on the training data
    src_lang = 'en'
    data_dir = {
        "multi30k": f"MMKG_MMT/datasets/multi30k-dataset/data/task1/raw/train.{lang}",
        "ikea": f"MMKG_MMT/datasets/IKEA-Dataset-master/IKEA/data.en.{tar_lang}/data.norm.tok.lc/train.norm.tok.lc.{lang}"
    }
    with open(data_dir[data_type], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [re.sub(r'[.,„(){}\d\'\"]','',i) for i in lines]
    words = set()
    lang_stem = {'en': "english", 'de': 'german', 'fr': 'french'}
    for i in lines:
        for j in i.split():
            word = SnowballStemmer(lang_stem[lang]).stem(j)
            if word in collected.keys(): continue
            words.add(word)
    return words

def clean_entity(data_type, tar_lang, lang):
    src_lang = 'en'
    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_trans_img_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl") as f:
        ent_trans_img = pickle.load(f)
    cleaned_ent_trans_img = {}
    for entity, img in ent_trans_img.items():
        # Use regular expressions to remove entities containing numbers and punctuation
        if re.match(r'^[^\d\W_]+$', entity):
            cleaned_ent_trans_img[entity] = img
    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_trans_img_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", "wb") as f:# store entity en-de translation pairs
        pickle.dump(cleaned_ent_trans_img, f)

if __name__ == "__main__":
    for data_type in ['ikea']:
        src_lang = 'en'
        for tar_lang in ['de', 'fr']:
            if tar_lang == 'fr' and data_type == 'multi30k': continue
            for lang in [src_lang, tar_lang]:#    
                item_img = {}
                ent_trans_img = {}
                collected = {}
                if os.path.exists(f"MMKG_MMT/datasets/mmkg-dataset/ent_trans_img_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl"):
                    with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_trans_img_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", 'rb') as f:
                        ent_trans_img = pickle.load(f)
                else:
                    ent_trans_img = {}
                if os.path.exists(f"MMKG_MMT/datasets/mmkg-dataset/collected_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl"):
                    with open(f"MMKG_MMT/datasets/mmkg-dataset/collected_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", 'rb') as f:
                        collected = pickle.load(f)
                else:
                    collected = {}
                datas = get_data_words(data_type=data_type, tar_lang=tar_lang, lang=lang)           
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = []
                    for i, data in enumerate(datas):
                        futures.append(executor.submit(get_similar_entities, data, tar_lang, lang))# 获得源句子所有词的类别: all_class_{data_type}_{lang}.pkl
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Exception occurred: {e}")
                # save file
                with open(f"MMKG_MMT/datasets/mmkg-dataset/collected_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", "wb") as f:
                    pickle.dump(collected, f)
                with open(f"MMKG_MMT/datasets/mmkg-dataset/ent_trans_img_{data_type}_{src_lang}_{tar_lang}_{lang}.pkl", "wb") as f:
                    pickle.dump(ent_trans_img, f)
                    
                # clean_entity(data_type=data_type, tar_lang=tar_lang, lang=lang)           
