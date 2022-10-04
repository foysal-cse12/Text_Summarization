# Introduction

In our proposed approach we have used text  segmentation  method along with text summarization model to  do  long  textsummarization. The main reason to use text segmentation model is to create chunks of long text. This approach helps us to scan long text easily to extract important information.  It also solves the limitation of maximum sequence length problem of different state of the art models as well as  we  can  effectively  use  the  short  text  summarization model to summarize long text documents.

We have followed below steps to implement our approach:

- [Apply segmentation method ](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/tree/master/Proposed%20Approach/Text%20Segmentation)
to segment long texts and create modified dataset for training and testing

- [Training Summarization Model with modified dataset](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/tree/master/Proposed%20Approach/Training)

- [Evaluation of the model](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/tree/master/Proposed%20Approach/Evaluation)



## Installation

To run all the source codes we need to install the following python libraries

```bash
pip install datasets==1.0.2
pip install transformers
pip install rouge_score
pip install SentencePiece
pip install Rouge
```








