# This code has been modified from the original [code](https://github.com/abisee/cnn-dailymail) 

I have modified this script to be easier (in case we need to reprocess our own data) the original script expects 
the data to be provided in a .story format , which is a data file contains both the text and its summary in the 
same file , so i just edited to be much simpler , now we can provide our data to this script in a csv format.
I have also replaced the need to download a specific java script (Stanford CoreNLP ) for tokenization , with the simpler nltk tokenizer.
Here I have created two different file (csv_to_binary_conversion_1 and csv_to_binary_conversion_2). If we have only one csv file and from there we have to split it into train, test and validationthen [csv_to_binary_conversion_1](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Read%20BigPatent%20Dataset/converting%20csv%20file%20to%20binary%20file/csv_to_binary_conversion_1.ipynb) file will be used. But if we have alreday splitted csv file then [csv_to_binary_conversion_2](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Read%20BigPatent%20Dataset/converting%20csv%20file%20to%20binary%20file/csv_to_binary_conversion_2.ipynb) file will be used. 

## My modification
1. input data has been simplified to be in a csv format 

| content        | title         
| -------------  |:-------------:
| tex1           | summary1
| tex2           | summary2      
| tex3           | asummary3   

2. replacing Stanford CoreNLP with nltk tokenizer to make it easy to process data without the need to download java files

-------------------------------------------------
## How to Run 
the requirements are , having
1. csv of your dataset must have 2 columns

| content        | title         
| -------------  |:-------------:

2.  modify the variable `cnn_stories_dir`
     to point to your main directory 

3.  replace `reviews_csv`
    with your csv path
 
-------------------------------------------------

## Output 
1. folder (cnn_stories_tokenized) used internally here 
2. **finished files** (the folder that we would use)
    |--> **(folder) chunks** ==> (used in upload)
    |--> test.bin  |--> not used in upload
    |--> train.bin |--> not used in upload
    |--> val.bin  |--> not used in upload
    |--> **vocab**  ==> (used in upload)


then 
put both 
  >>|--> **(folder) chunks** ==> (used in upload)
  >>|--> **vocab**  ==> (used in upload)
  
