#pip install datasets==1.0.2
#pip install transformers
#pip install rouge_score
#pip install SentencePiece
#pip install Rouge
import datasets
import transformers
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
import numpy as np 
import os
import sys

import nltk
#nltk.download('stopwords') 
#nltk.download('wordnet') 
from clean_text import *
from clean_summary import *
import torch
from transformers import AutoTokenizer
from transformers import T5Tokenizer

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

arg = len(sys.argv)
if (arg!=3):

    sys.exit('Usage: python summarizer.py training_csv_file validation_csv_file')
tr_file = sys.argv[1]
val_file = sys.argv[2]


print('reading training data')
data_train=pd.read_csv(tr_file,sep = ',')#nrows=780000
#data_train[['Text','Summary']].sample(5)
#data_train.head()


#####################################
####### data cleaning          ######
#####################################

def data_cleaning(data):

    cleaned_summary = []
    cleaned_text = []
    for text in data['Text']:
        cleaned_text.append(clean_text(text, remove_stopwords=True))
    print("Text preprocessing done.")

    for summary in data['Summary']:
        cleaned_summary.append(clean_summary(summary,remove_stopwords=False))
    print("summary preprocessing done.")
    
    return cleaned_text,cleaned_summary
     
cleaned_text,cleaned_summary = data_cleaning(data_train)



#if summary has no value then delete the rows
data_train['Text']=cleaned_text
data_train['Summary']=cleaned_summary
data_train['Summary'].replace('', np.nan, inplace=True)
data_train.dropna(subset=['Summary'],inplace=True)
# resetting the DatFrame index
data_train = data_train.reset_index(drop=True)

#apply condition (if number of words in a text less then a certaion number delete it)
data_train = data_train[data_train['Text'].str.split().str.len()>20] #less then = lt(2) greater than = ge(4)
data_train = data_train.reset_index(drop=True)

#apply condition (if number of words in a summary less then a certaion number delete it)
data_train = data_train[data_train['Summary'].str.split().str.len()>5] #less then = lt(2) greater than = ge(4)
data_train = data_train.reset_index(drop=True)

# make x train and y train
X_train = data_train['Text']
Y_train = data_train['Summary']
print('length of x_train: ',len(X_train))
print('length of y_train: ',len(Y_train))

#####################################
####### Validation data         #####
#####################################

print('reading training data')
data_val=pd.read_csv(val_file,sep = ',')#nrows=5000
#data_val[['Text','Summary']].sample(5)
#data_val.head()

cleaned_text,cleaned_summary = data_cleaning(data_val)

#if summary has no value then delete the rows
data_val['Text']=cleaned_text
data_val['Summary']=cleaned_summary
data_val['Summary'].replace('', np.nan, inplace=True)
data_val.dropna(subset=['Summary'],inplace=True)
# resetting the DatFrame index
data_val = data_val.reset_index(drop=True)

#apply condition (if number of words in a text less then a certaion number delete it)
data_val = data_val[data_val['Text'].str.split().str.len()>20] #less then = lt(2) greater than = ge(4)
data_val = data_val.reset_index(drop=True)

#apply condition (if number of words in a text less then a certaion number delete it)
data_val = data_val[data_val['Summary'].str.split().str.len()>5] #less then = lt(2) greater than = ge(4)
data_val = data_val.reset_index(drop=True)

# make x val and y val
X_val = data_val['Text']
Y_val = data_val['Summary']
print('length of x_val: ',len(X_val))
print('length of y_val: ',len(Y_val))

train_data=Dataset.from_pandas(data_train)
val_data=Dataset.from_pandas(data_val)

#####################################
#######        Tokenization    ######
#####################################
tokenizer = AutoTokenizer.from_pretrained('t5-base')

#####################################
#######    Parameter setting   ######
#####################################
batch_size=32  # 512,256
max_source=512 # text word length
max_target=128 # summary word length

def tokenize(batch):
    tokenized_input = tokenizer(batch['Text'], padding='max_length', truncation=True, max_length=max_source)
    tokenized_label = tokenizer(batch['Summary'], padding='max_length', truncation=True, max_length=max_target)

    tokenized_input['labels'] = tokenized_label['input_ids']

    return tokenized_input
    
#train_data = train_data.map(tokenize, batched=True, batch_size=32) 
#val_data = val_data.map(tokenize, batched=True, batch_size=32) 
train_data = train_data.map(tokenize, batched=True, batch_size=32,remove_columns=["Text", "Summary"]) #512  #new one also works
val_data = val_data.map(tokenize, batched=True, batch_size=32,remove_columns=["Text", "Summary"]) #len(val_dataset) #new one also works    

#train_data.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])
#val_data.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])
train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

#####################################
#######  Start Training Phase  ######
#####################################

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model_path = 't5-base'
model = T5ForConditionalGeneration.from_pretrained(model_path).to(torch_device)

output_dir = "E:\\fine_tuned_model\\"
log_dir = "E:\\fine_tuned_model\\log_dir\\"

##################################################################################################################
#### if you want to run this script using google colub then cooment above teo lines uncomment below four lines ####
##################################################################################################################
print('temporary file creating')
#os.mkdir("/content/T5_proposed_test")
#output_dir = "/content/T5_proposed_test"
#os.mkdir("/content/T5_proposed_test/log_dir")
#log_dir = "/content/T5_proposed_test/log_dir"


print('setting training arguments')
# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir = log_dir,
    num_train_epochs=10, #default 8
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,#batch_size,
    ##predict_with_generate=True,
    logging_steps=1000,  
    save_steps=1000,  
    eval_steps=1000,  
    warmup_steps=2000,  
    ###max_steps=16, # delete for full training
    overwrite_output_dir=True,
    save_total_limit=1, #3
    #fp16=True, 
    weight_decay=0.1,
    remove_unused_columns=True, # Removes useless columns from the dataset
    prediction_loss_only=True, # If I need co compute only loss and not other metrics, setting this to true will use less RAM

    #additional
    load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
    metric_for_best_model="loss", # Use loss to evaluate best model.
    greater_is_better=False # Best model is the one with the lowest loss, not highest.
)

print('starts training')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer = tokenizer
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # default block it
)
trainer.train()

#trainer.save_model(output_dir + '/model')
#tr_path = '/content/drive/MyDrive/transformer_model/after_ideatalk/arxiv_models/'
#!cp -r /content/T5_ep_10_arxiv $tr_path
#print('new done ep 10 v2 full data  segmented')











