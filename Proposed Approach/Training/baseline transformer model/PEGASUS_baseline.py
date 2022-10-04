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

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

###############################
########## check GPU ##########
###############################

        


arg = len(sys.argv)
if (arg!=3):

    sys.exit('Usage: python PEGASUS_baseline.py training_csv_file validation_csv_file')
tr_file = sys.argv[1]
val_file = sys.argv[2]

def check_gpu():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
check_gpu()

class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)
        
def prepare_data(model_name, 
                 train_texts, train_labels, 
                 val_texts, val_labels, 
                 test_texts=None, test_labels=None):  
  """
  Prepare input data for model fine-tuning
  """
  print('inside prepare data')
  print('\n')
  tokenizer = PegasusTokenizer.from_pretrained(model_name)
  print('pretrain tokenizer load done')
  print('\n')

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True
  #'''
  def tokenize_data(texts, labels):
    print('inside tokenize_data function')
    print('\n')
    #encodings = tokenizer(texts, truncation=True, padding=True,return_tensors="pt")#.to(device) 
    encodings = tokenizer(texts, truncation=True, padding='max_length',return_tensors="pt",max_length=512)
    ##print('encodings: ',encodings)
    ##print('\n')
    #decodings = tokenizer(labels, truncation=True, padding=True,return_tensors="pt")#.to(device)
    decodings = tokenizer(labels, truncation=True, padding='max_length',return_tensors="pt",max_length=128)
    ##print('decodings: ',decodings)
    ##print('\n')
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized
  #'''
  print('train_dataset tokenizer operation')
  print('\n')
  train_dataset = tokenize_data(train_texts, train_labels)
  
  ##val_dataset = tokenize_data(val_texts, val_labels) 
  ##test_dataset = tokenize_data(test_texts, test_labels) 
  print('val_dataset tokenizer operation')
  print('\n')
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  print('test_dataset tokenizer operation')
  print('\n')
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer  

output_dir = "E:\\fine_tuned_model\\"
log_dir = "E:\\fine_tuned_model\\log_dir\\"
######################################################################################################################
#### if you want to run this script using google colub then comment above two lines and uncomment below four lines ####
######################################################################################################################
print('temporary file creating')
#os.mkdir("/content/T5_proposed_test")
#output_dir = "/content/T5_proposed_test"
#os.mkdir("/content/T5_proposed_test/log_dir")
#log_dir = "/content/T5_proposed_test/log_dir"  


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
     

    
    
def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset, freeze_encoder=False, output_dir=output_dir):  
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=4,           # total number of training epochs
      per_device_train_batch_size=2,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=2,    # batch size for evaluation, can increase if memory allows
      save_steps=1000,                  # number of updates steps before checkpoint saves #500 #1000
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=1000,                  # number of update steps before evaluation #200 #10000
      warmup_steps=2000,                # number of warmup steps for learning rate scheduler #500 #2000
      weight_decay=0.1,               # strength of weight decay
      #logging_dir='./logs',            # directory for storing logs
      logging_dir = log_dir,
      logging_steps=1000,#10
      overwrite_output_dir=True, #abdullah
      #learning_rate = 1e-5, # abdullah

      #additional
      load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
      metric_for_best_model="loss", # Use loss to evaluate best model.
      greater_is_better=False # Best model is the one with the lowest loss, not highest.
     
  
    )

    trainer = Trainer(
      model=model,                         # the instantiated ðŸ¤— Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer,
      #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # default block it
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2,           # total number of training epochs
      per_device_train_batch_size=2,   # batch size per device during training, can increase if memory allows
      save_steps=1000,                  # number of updates steps before checkpoint saves #500
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=2000,                # number of warmup steps for learning rate scheduler #500
      weight_decay=0.1,               # strength of weight decay
      #logging_dir='./logs',            # directory for storing logs
      logging_dir = log_dir,
      logging_steps=1000, #10
      overwrite_output_dir=True, #abdullah
      #learning_rate = 1e-5, # abdullah
      
     
    )

    trainer = Trainer(
      model=model,                         # the instantiated ðŸ¤— Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer    
  
if __name__=='__main__':
  print('reading training data')
  data_train=pd.read_csv(tr_file,sep = ',',nrows=3000)#nrows=140000
  # Data Cleaning
  cleaned_text,cleaned_summary = data_cleaning(data_train)

  #if summary has no value then delete the rows
  data_train['Text']=cleaned_text
  data_train['Summary']=cleaned_summary
  data_train['Summary'].replace('', np.nan, inplace=True)
  data_train.dropna(subset=['Summary'],inplace=True)
  # resetting the DatFrame index
  data_train = data_train.reset_index(drop=True)

  #apply condition (if number of words in a text less then a certaion number delete it)
  data_train = data_train[data_train['Text'].str.split().str.len()>40] #less then = lt(2) greater than = ge(4)
  data_train = data_train.reset_index(drop=True)

  #apply condition (if number of words in a summary less then a certaion number delete it)
  data_train = data_train[data_train['Summary'].str.split().str.len()>5] #less then = lt(2) greater than = ge(4)
  data_train = data_train.reset_index(drop=True)


  #Validation data
  print('reading validation data')
  data_val=pd.read_csv(val_file,sep = ',',nrows=1000)#nrows=5000
  cleaned_text_val,cleaned_summary_val = data_cleaning(data_val)


  data_val['Text']=cleaned_text_val
  data_val['Summary']=cleaned_summary_val
  data_val['Summary'].replace('', np.nan, inplace=True)
  data_val.dropna(subset=['Summary'],inplace=True)
  # resetting the DatFrame index
  data_val = data_val.reset_index(drop=True)

  #apply condition (if number of words in a text less then a certaion number delete it)
  data_val = data_val[data_val['Text'].str.split().str.len()>40] #less then = lt(2) greater than = ge(4)
  data_val = data_val.reset_index(drop=True)

  #apply condition (if number of words in a text less then a certaion number delete it)
  data_val = data_val[data_val['Summary'].str.split().str.len()>5] #less then = lt(2) greater than = ge(4)
  data_val = data_val.reset_index(drop=True)

  # convert pandas series to list
  train_texts = data_train['Text'].tolist()
  train_labels = data_train['Summary'].tolist()
  print('length of train_texts: ',len(train_texts))
  print('length of train_labels: ',len(train_labels))
  
  val_texts = data_val['Text'].tolist()
  val_labels = data_val['Summary'].tolist()
  print('length of val_texts: ',len(val_texts))
  print('length of val_labels: ',len(val_labels))

  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-large'
  #train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
  print('tokenization data')
  print('\n')
  print('start prepare data')
  print('\n')
  train_dataset,val_dataset, _, tokenizer = prepare_data(model_name, train_texts, train_labels,val_texts, val_labels)#abdullah
  print('start fine tuning')
  print('\n')
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset,val_dataset)
  trainer.train()
  



#trainer.save_model(output_dir + '/model')
#tr_path = '/content/drive/MyDrive/transformer_model/after_ideatalk/arxiv_models/'
#!cp -r /content/T5_ep_10_arxiv $tr_path
#print('new done ep 10 v2 full data  segmented')











