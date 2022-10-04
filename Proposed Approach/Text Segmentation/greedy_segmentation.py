# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:14:43 2021

@author: Abdullah
"""

import os
#import word2vec
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from textsplit.tools import SimpleSentenceTokenizer
#%matplotlib inline
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal, split_greedy, get_total

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import re
import os

import sys

arg = len(sys.argv)
if (arg!=3):

    sys.exit('Usage: python greedy_segmentation.py input_csv_file output_csv_file')
in_file = sys.argv[1]
out_file = sys.argv[2]


#word2vec pre trained model path
w2v_file = 'E:\\Masters Thesis\\word embeddings model\\word2vec\\GoogleNews-vectors-negative300.bin.gz'

#load model
model =  KeyedVectors.load_word2vec_format(w2v_file,binary = True,limit = 1000000)

#make a dataframe
wrdvecs = pd.DataFrame(model.vectors, index=model.key_to_index)
##print('dataframe shape: ',wrdvecs.shape)

vecr = CountVectorizer(vocabulary=wrdvecs.index)
#print(vecr)

# Read input data file
#data = pd.read_csv("E:\\fairy_tale_dataset\\country.csv")
data = pd.read_csv(in_file)

#Apply text segmentation approach (greedy segmentation) based on word embeddings

segmented_article_text = []
for i in range (0,len(data)): #len(data) 
    ##print('text id no: ',i)
    text = data['Text'][i]
    sentenced_text = sent_tokenize(text)
    sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
    total_words = len(text.split())
    
    if(total_words<=1000): 
        segment_list = []
        segment_list.append(text)
        
                   
    
    else:
        number_of_segments = (int(total_words/1000)+1)
        ##print('\n')
        greedy_segmentation = split_greedy(sentence_vectors, max_splits=number_of_segments)
        segmented_text = get_segments(sentenced_text, greedy_segmentation)
        ##print('%d sentences, %d segments, avg %4.2f sentences per segment' % (len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text)))
        segment_list = []
        for seg_number, seg in enumerate (segmented_text): 
            seg = ' '.join(seg)
            segment_list.append(seg)

    segmented_article_text.append(segment_list)
    
# make a new column in the dataframe for segmented data    
data['segmented_text']=segmented_article_text    
#Save segmented text in a csv file
data.to_csv(out_file, encoding='utf-8', index=False)
print('csv file generated')
