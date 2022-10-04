import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import re
import os
import sys
from rouge_score import rouge_scorer

arg = len(sys.argv)
if (arg!=3):

    sys.exit('Usage: python summary_splitting.py input_csv_file output_csv_file')
in_file = sys.argv[1]
out_file = sys.argv[2]

new_data = pd.read_csv(in_file, converters={'segmented_text': eval})
##print(new_data.shape)

section_text_list = []
split_summary_list = []
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

data_len = len(new_data['Summary'])
for i_data in range(0,len(new_data)): #len(new_data)
    ### control data id (csv file) ###
    ##print('data id: ',i_data)
    ##print('\n')
    sections_list =  new_data['segmented_text'][i_data] #
    section_length = len(new_data['segmented_text'][i_data])
    
    #### control summary sentence #####
    summary_text = new_data['Summary'][i_data]
    summary_sentences = sent_tokenize(summary_text)
    
    all_section_sentences = []
    ##### control section text #####
    for i_sections in range(0,section_length):
        section_text = sections_list[i_sections]
        section_sentences = sent_tokenize(section_text)
        all_section_sentences.append(section_sentences)
    
    if(len(all_section_sentences)==1): # if there is only one section then we don't need to split summary

        new_section_text = new_data['Text'][i_data] # if the section length is one copy article text
        section_text_list.append(new_section_text) # list for segmented text (section)
        split_summary_list.append(summary_text) # list for individual summary for segmented text
        
    else: # if there is more than one section then we need to split summary
   
        
        max_index_list = [] # this list will store the index number of segmented text which has higher precision 
        ##### new logic will start from here #####
        
        for i in range(0,len(summary_sentences)):
            # this loop controls each sentence from summary
            precision_result = []
            max_precision_list = [] ##  max precision list from each section text against easch summary sentence

            summary_sentence = summary_sentences[i]
            
            for j in range(0,len(all_section_sentences)):
                # this loop controls each individual segments
                section_texts_sen = all_section_sentences[j]
                precision_list = [] #  precision list from exach section text against easch summary sentence
                
                for k in range(0,len(section_texts_sen)):
                    # this loop controls each individual sentences from each segments
                    scores = scorer.score(section_texts_sen[k],summary_sentences[i])
                    precision = scores['rougeL']
                    precision, recall, fmeasure = scores['rougeL']
                    precision_result.append(precision)
                    precision_list.append(precision) ### saving pricision value of each individual sentence in section text with summary tsentences

                max_precision =  max(precision_list) # max precision from the precision list  from exach section text against easch summary sentence

                max_precision_list.append(max_precision)    
            
            ###find index of maximum value from the list
            max_value = max(max_precision_list)
            max_index = max_precision_list.index(max_value)

            max_index_list.append(max_index) # saving the higher precision segmented text index

    
    ####print('---------------------------- assigning summary to individual segmented text------------------------\n')
    
        for s in range(0,section_length):
            s_text = new_data['segmented_text'][i_data][s]
            ####+print(s_text,'\n')

            section_summary = ''
            for m in range(0,len(max_index_list)):
                if(s==max_index_list[m]):
                    section_summary = section_summary+summary_sentences[m]
            ####+print('section_summary: ',section_summary,'\n')   

            section_text_list.append(s_text)
            split_summary_list.append(section_summary)
                
                
####print(len(section_text_list))
####print(len(split_summary_list)) 

final_data = {'Text':section_text_list,'Summary':split_summary_list}
df = pd.DataFrame(final_data)
##print(df.head(5))
##print('before: ',len(df))

###delete empty coloumns
df['Summary'].replace('', np.nan, inplace=True)
df.dropna(subset=['Summary'],inplace=True)
# resetting the DatFrame index
df = df.reset_index(drop=True)
##print(df.head(5))
##print('after: ',len(df))

## save as csv file
df.to_csv(out_file, encoding='utf-8',index=0)

print('csv file generated')