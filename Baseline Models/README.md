# Baseline Models Implementation

Five baseline models have been implemented here using BigPatent dataset where four models are Extractive Summarization Model and one is Abstractive Summarization Model.

## Installation

Use the package manager [pip](https://pypi.org/project/sumy/) to install sumy

```bash
pip install sumy
```


## Extractive Summarization Models

- [SumBasic](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Baseline%20Models/Extractive%20Summarization%20Models/SumBasic.ipynb):
The SumBasic algorithm was developed in 2005 and uses only the word probability approach to determine sentence importance

- [LexRank](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Baseline%20Models/Extractive%20Summarization%20Models/LEXRANK.ipynb):
This is a unsupervised approach to text summarization based on graph-based centrality scoring of sentences. The main idea is that sentences “recommend” other similar sentences to the reader. Thus, if one sentence is very similar to many others, it will likely be a sentence of great importance.

- [LSA](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Baseline%20Models/Extractive%20Summarization%20Models/LSA-Latent_Semantic_Analysis.ipynb)
Latent semantic analysis is an unsupervised method of summarization it combines term frequency techniques with singular value decomposition to summarize texts. It is one of the most recent suggested technique for summerization

- [TextRank](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Baseline%20Models/Extractive%20Summarization%20Models/TextRank.ipynb)
Text rank is a graph-based summarization technique with keyword extractions in from document.

## Abstractive Summarization Models
- [Sequence to Sequence Model](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Baseline%20Models/Abstractive%20Summarization%20Models/seq2seq.ipynb)





