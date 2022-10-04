# Introduction
To run a full training session use the following line
```bash
python summarizer.py training_csv_file validation_csv_file
```

Here training_csv_file and validation_csv_file are the output file from [summary_splitting.py](https://gitlab.com/genie-enterprise/research/automatic-summarization-of-long-documents/-/blob/master/Proposed%20Approach/Text%20Segmentation/summary_splitting.py). We have set the following parameters in the training arguments. Feel free to change according to your plan

```bash
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir = log_dir,
    num_train_epochs=10, 
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=1000, 
    save_steps=1000,  
    eval_steps=1000,  
    warmup_steps=2000,  
    overwrite_output_dir=True,
    save_total_limit=1, 
    weight_decay=0.1,
    remove_unused_columns=True, 
    prediction_loss_only=True, 
    load_best_model_at_end=True, 
    metric_for_best_model="loss", 
    greater_is_better=False 
)

```
