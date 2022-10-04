# Introduction
To evaluate the proposed model, run the following line
```bash
python proposed_model_evaluation.py test_csv_file
```

Here test_csv_file is the output file from [greedy_segmentation.py](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/greedy_segmentation.py) or [optimal_segmentation.py](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/optimal_segmentation.py)

To evaluate the baseline models, run the following line
```bash
python baseline_model_evaluation.py test_csv_file
```

Here test_csv_file is the original file from any summarization dataset. We do not need to perform any segmentation method to test or evaluate baseline models.

Note: Before exeuting the python script, please add the location of model file or checkpoint in the 'model_dir' variable inside the script.
You can find the pretrained model in this [link](https://drive.google.com/file/d/1HCZvVEdv6rNQ8Q7OL4pXm0I0jQtVDBNV/view?usp=sharing) 
