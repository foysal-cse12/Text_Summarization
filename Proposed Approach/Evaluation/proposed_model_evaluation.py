#pip install datasets==1.0.2
#pip install transformers
#pip install rouge
import datasets
import transformers
import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from clean_text import *
from clean_summary import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import re
import os
import sys
from rouge import Rouge
import pprint
from transformers import AutoTokenizer
from transformers import T5Tokenizer
import torch
rouge = Rouge()

arg = len(sys.argv)
if (arg!=2):

    sys.exit('Usage: python proposed_model_evaluation.py test_csv_file')
test_file = sys.argv[1]
#val_file = sys.argv[2]

data_test=pd.read_csv(test_file,sep = ',',converters={'segmented_text':eval})#100000 #,nrows=500
#data_test[['Text','Summary','segmented_text']].sample(5)
#data_test.head(10)

#####################################
########### clean data ###############
#####################################

for j in range (0,len(data_test)): #len(data_test)
  #print('text id: ',j)
  if (j%5==0):
    print('text id: ',j)
  splitted_text = data_test['segmented_text'][j]

  ##print(splitted_text)
  ##print(len(splitted_text))
  for k in range(0,len(splitted_text)):
    text = splitted_text[k]
    ##print('before clean : ',text)
    clean_segmented_text = clean_text(text, remove_stopwords=True)
    #print('number of words: ',len(clean_segmented_text.split()))
    if(len(clean_segmented_text.split())<=20): #<50
      clean_segmented_text = ''
    splitted_text[k] = clean_segmented_text

  splitted_text =  [x for x in splitted_text if x]
  data_test['segmented_text'][j] = splitted_text

  
  
####################################
#### Test the pre trained model ####  
####################################
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

model_dir = ''
tokenizer = AutoTokenizer.from_pretrained(model_dir)
#tokenizer = AutoTokenizer.from_pretrained('t5-base')
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(torch_device)
print('model loaded')

article = []
human_generated_summary = []
machine_generated_summary = []
for i in range(0,len(data_test)): #len(data_test)
  #print('test id: ',i)
  if (i%5==0):
    print('text id: ',i)
  ###print('\n')
  splitted_text = data_test['segmented_text'][i]
  num_of_split = len(splitted_text)

  article.append(data_test['Text'][i])
  human_generated_summary.append(data_test['Summary'][i])
  split_summary = []
  for split_no in range(0,num_of_split):
    inputs = tokenizer(splitted_text[split_no], return_tensors='pt',truncation=True, padding='longest').to(torch_device)#padding="max_length"
    prediction =  model.generate(**inputs,max_length = 300,num_beams=5, early_stopping=True, no_repeat_ngram_size=3)[0] # num_beams=5, repetition_penalty=1,length_penalty=1,  early_stopping=True, no_repeat_ngram_size=2
    #prediction = tokenizer.batch_decode(prediction)
    prediction = tokenizer.decode(prediction, skip_special_tokens=True) #abdullah

    prediction = ''.join(prediction) # convert list to string
    split_summary.append(prediction)

  sum_merge = ''.join(split_summary)  
  machine_generated_summary.append(sum_merge)

#print(len(article))
#print(len(human_generated_summary)) 
#print(len(machine_generated_summary))

data_ = {'Article':article,'original_Summary':human_generated_summary,'Predicted_Summary':machine_generated_summary}
df = pd.DataFrame(data_)
#sys.setrecursionlimit(2000)

print(rouge.get_scores(machine_generated_summary, human_generated_summary, avg=True))

############################
##### save output file #####
############################
#df.to_csv('',index = False)
