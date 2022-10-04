# Introduction
For text segmentation, we have followed the paper [Text Segmentation based on Semantic Word Embeddings](https://arxiv.org/pdf/1503.05543.pdf). We used two different types of algorithms to split long texts, which are (1) Greedy approach and (2) Optimal approach or dynamic programming. Both approaches depend on a penalty hyperparameter, that defined the granularity of the split. We have used Word2Vec model for word embeddings. Please download the word2vec model from [here](https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download) and give the file path in w2v variable in both [greedy_segmentation.py](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/greedy_segmentation.py) and [optimal_segmentation.py](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/optimal_segmentation.py) before run the code.

#### [Greedy Approach](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/greedy_segmentation.py)
It splits the text iteratively at the position where the gain is highest until this gain would be below a given penalty threshold. The gain is the sum of norms of the left and right segments minus the norm of the segment that is to be split.

To segment a long text using greedy approach, run the following line
```bash
greedy_segmentation.py input_csv_file output_csv_file
```
#### [Optimal Approach(Dynamic Programming)](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/optimal_segmentation.py)
Iteratively construct a data structure storing the results of optimally splitting a prefix of the document. This results in a matrix storing a score for making a segment from position i to j, given a optimal segmentation up to i.
To segment a long text using optimal approach, run the following line
```bash
optimal_segmentation.py input_csv_file output_csv_file
```
#### Penalty hyperparameter choice
The greedy implementation does not need the penalty parameter, but can also be run by limiting the number of segments. This is leveraged by the [get_penalty](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/textsplit/tools.py) function to approximately determine a penalty parameter for a desired average segment length computed over a set of documents.

## Input CSV file format
Input CSV file should have at leasts two columns namely 'Text' and 'Summary'. If you have different columns then either you have to rename the columns before run the code or you need to change some part of the code.

| Text        | Summary         
| -------------  |:-------------:
| long tex1           | summary1
| long tex2           | summary2      
| long tex3           | asummary3   

## Summary Splitting
After generating the output csv file using [greedy approach](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/greedy_segmentation.py) or [dynamic programming](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/optimal_segmentation.py), we need to use this output file as an input to run following code for summary splitting.
```bash
summary_splitting.py input_csv_file output_csv_file
```

We have used a method that automatically splits the summary of a document into sections and pairs each of these sections to the appropriate section of the document, in order to creat distinct target summaries. we used ROUGE metrics in order to match each part of the summary with a section of the document and assigned the most suitable summary sentences to the appropriate section of the document. This summary splitting technique is inspired by [here](https://arxiv.org/pdf/2004.06190.pdf)

The output file of [summary splitting](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/summary_splitting.py) is our final file or modified dataset which is used for model training and validation. For testing or evaluation we do not need to perform summary splitting method.







