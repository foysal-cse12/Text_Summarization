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

    sys.exit('Usage: python optimal_segmentation.py input_csv_file output_csv_file')
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

#Apply text segmentation approach (optimal segmentation) based on word embeddings

segmented_article_text = []
for i in range (0,len(data)):#len(data)
    ##print('text id no: ',i)
    text = data['Text'][i]
    #text = data['story'][i]
    ##print('num of words in the text: ',len(text.split()))
    ##print('\n')
    sentenced_text = sent_tokenize(text)
    ##print('num of sentences in the text: ',len(sentenced_text))
    ##print('\n')
    sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
    ##print('length of the sentence vector: ',len(sentence_vectors))
    ##print('\n')
    #segment_len = 12#15
    total_words = len(text.split())
    
    if(total_words<=1000): # if total number of words is 800 we will not make any segments #default 1000
        ##print('inside if condition')
        segment_list = []
        segment_list.append(text)
        ##print('no segment create. original text assigned')
        ##print('\n')
           
        
    
    else:
        ##print('inside else condition')
        segment_len = 6 # segment_length maximum means less segment will create ..else vise verse #defaults 15
        penalty = get_penalty([sentence_vectors], segment_len)
        ##print('penalty %4.2f' % penalty)
        if (penalty==0.00):
            ##print('\n')
            
            penalty = 7.5
            ##print('new penalty %4.2f' % penalty)
            optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
            segmented_text = get_segments(sentenced_text, optimal_segmentation)
            ##print('%d sentences, %d segments, avg %4.2f sentences per segment' % (len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text)))
            segment_list = []
            for seg_number, seg in enumerate (segmented_text): 
                seg = ' '.join(seg)
                segment_list.append(seg)
                ##print('segment {} :\n{}\n===='.format(seg_number,seg))
                ##print('Total number of word in the segmented text: ',len(seg.split()),'\n')
        else:
            optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
            segmented_text = get_segments(sentenced_text, optimal_segmentation)
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