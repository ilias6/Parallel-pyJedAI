#!/usr/bin/env python
import numpy as np
#from gensim.models import Word2Vec
#from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from sentence_transformers import SentenceTransformer
from transformers import AlbertTokenizer, AlbertModel
import transformers
transformers.logging.set_verbosity_error()
import torch

from time import time
import psutil
import os
import pandas as pd
import json


def create_embeddings(text, vectorizer, log, log_file, output_path, output_index, 
                      b=500):
   if vectorizer in ['word2vec', 'fasttext']:
       t1 = time()
       text2 = [t.split(' ') for t in text]
       
       voc = None
       vector_size = 300
       if vectorizer == 'word2vec':
           #model = Word2Vec(sentences=text2, vector_size=vector_size, window=5, 
           #                 min_count=1, workers=4)
           #voc = model.wv
            model = api.load('word2vec-google-news-300')
            voc = model        
       elif vectorizer == 'fasttext': 
           #model = FastText(text2, min_count=1, vector_size=vector_size)
           #voc = model.wv
            model = load_facebook_model('wiki.simple.bin')
            voc = model.wv
           
       vectors = []
       for nos, sentence in enumerate(text2):
           if nos % 1000 == 0:
                print(f'\r\t {nos}/{len(text2)}', end='')
           vector = np.zeros(vector_size)
           no_words = 0
           for word in sentence:
               if word in voc:
                   vector += voc[word]
                   no_words += 1
           if no_words > 0:
               vector = vector / no_words
           vectors.append(vector)
       vectors = np.array(vectors)
       t2 = time()
       total_time = t2-t1
       
       df = pd.DataFrame(vectors)
       df.index = output_index
       df.to_csv(output_path, index=True, header=False)
   
   elif vectorizer in ['bert', 'distilbert', 'roberta', 'xlnet', 'albert']:
       total_time = time()
       if vectorizer == 'bert':
           tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
           model = BertModel.from_pretrained("bert-base-uncased")
       elif vectorizer == 'distilbert':
           tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
           model = DistilBertModel.from_pretrained("distilbert-base-uncased")            
       elif vectorizer == 'roberta':
           tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
           model = RobertaModel.from_pretrained('roberta-base')
       elif vectorizer == 'xlnet':
           tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
           model = XLNetModel.from_pretrained('xlnet-base-cased')                
       elif vectorizer == 'albert':
           tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
           model = AlbertModel.from_pretrained("albert-base-v2")  
       total_time = time() - total_time
       
       
       with open(output_path, 'w') as o:
           total = len(range(0, len(text), b))
           for i in range(0, len(text), b):
               print(f'\r\t {i//b}/{total}', end='')
               t1 = time()
               temp_text = text[i:i+b]
               temp_index = output_index[i:i+b]
               encoded_input = tokenizer(temp_text, return_tensors='pt', truncation=True,
                                         max_length=100, padding='max_length')
               output = model(**encoded_input)
               vectors = output.last_hidden_state[:,0,:]
               t2 = time()
               total_time += t2-t1
               
               #flushing
               vectors = vectors.detach().numpy()
               df = pd.DataFrame(vectors)
               df.index = temp_index
               df.to_csv(o, index=True, header=False)

   elif vectorizer in ['smpnet', 'st5', 'glove',
                       'sdistilroberta', 'sminilm']:
       total_time = time()
       device = torch.device('cuda')
       if vectorizer == 'smpnet':
           model = SentenceTransformer('all-mpnet-base-v2', device=device)
       elif vectorizer == 'st5':
           model = SentenceTransformer('gtr-t5-large', device=device)
       elif vectorizer == 'sdistilroberta':
           model = SentenceTransformer('all-distilroberta-v1', device=device)
       elif vectorizer == 'sminilm':
           model = SentenceTransformer('all-MiniLM-L12-v2', device=device)      
       elif vectorizer == 'glove':
           model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device=device)
       total_time = time() - total_time
       
       with open(output_path, 'w') as o:
           total = len(range(0, len(text), b))
           for i in range(0, len(text), b):
               print(f'\r\t {i//b}/{total}', end='')
               t1 = time()
               temp_text = text[i:i+b]
               temp_index = output_index[i:i+b]
               vectors = model.encode(temp_text)
               t2 = time()
               total_time += t2-t1
           
               #flushing
               df = pd.DataFrame(vectors)
               df.index = temp_index
               df.to_csv(o, index=True, header=False)
              
   pid = os.getpid()
   python_process = psutil.Process(pid)
   process_memory = python_process.memory_info()
   process_memory = {k: v for k, v in zip(process_memory._fields, process_memory)}
    
   total_memory = psutil.virtual_memory()
   total_memory = {k: v for k, v in zip(total_memory._fields, total_memory)}
   
   log['time'] = total_time
   log['memory'] = {'process': process_memory,
                     'total': total_memory}
   log['dimensions'] = vectors.shape[1]
    
   with open(log_file, 'a') as f:
       f.write(json.dumps(log)+"\n")
   
   return vectors
