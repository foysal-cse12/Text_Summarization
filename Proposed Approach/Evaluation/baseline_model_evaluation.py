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

    sys.exit('Usage: python baseline_model_evaluation.py test_csv_file')
test_file = sys.argv[1]
#val_file = sys.argv[2]

data_test=pd.read_csv(test_file,sep = ',')#100000 #,nrows=500
#data_test[['Text','Summary','segmented_text']].sample(5)
#data_test.head(10)

#####################################
########### clean data ###############
#####################################

cleaned_summary = []
cleaned_text = []

for text in data_test['Text']:
    cleaned_text.append(clean_text(text, remove_stopwords=True))
print("Text preprocessing done.")  
  
####################################
#### Test the pre trained model ####  
####################################
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

model_dir = ''
tokenizer = AutoTokenizer.from_pretrained('t5-base')
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(torch_device)
print('model loaded')

article = []
human_generated_summary = []
machine_generated_summary = []
for i in range(len(data_test)):
#for i in range(0,5):
  if (i%5==0):
    print('text id: ',i)
  #print('text no: ',i)
  input_ids = tokenizer(data_test['Text'][i], return_tensors="pt",truncation=True, padding='longest').to(torch_device)#,max_length=512
  # Generate Summary
  output_ids = model.generate(**input_ids,max_length= 300,min_length=60, num_beams=5, repetition_penalty=1, length_penalty=6, early_stopping=True, no_repeat_ngram_size=3)[0] # num_beams=5, repetition_penalty=1,length_penalty=1,  early_stopping=True, no_repeat_ngram_size=2
  ### make length_penalty = 4,,min_length=60,length_penalty=4,

  prediction = tokenizer.decode(output_ids, skip_special_tokens=True)


  article.append(data_test['Text'][i])
  human_generated_summary.append(data_test['Summary'][i])

  machine_generated_summary.append(prediction)

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