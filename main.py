from transformers import BertForQuestionAnswering, BertTokenizerFast, default_data_collator, \
    TrainingArguments, IntervalStrategy, Trainer
from datasets import load_dataset
from pathlib import Path
from preprocessing import flatten_data, prepare_train_features


# dataset SQuAD v1 pt_BR: https://drive.google.com/file/d/1Q0IaIlv2h2BC468MwUFmUST0EyN7gNkn/view



def run():

    train_file = './data/flat_squad-train-v1.1.json' \
        if Path('./data/flat_squad-train-v1.1.json').exists() else flatten_data('./data/squad-train-v1.1.json')

    validation_file = './data/flat_squad-dev-v1.1.json' \
        if Path('./data/flat_squad-dev-v1.1.json').exists() else flatten_data('./data/squad-dev-v1.1.json')

    qa_dataset = load_dataset('json', data_files={'train': train_file, 'validation': validation_file}, field='data')

    model_type = 'base'
    # model_type = 'large'

    stride = 128
    max_length = 512 - stride

    tokenizer = BertTokenizerFast.from_pretrained(f'neuralmind/bert-{model_type}-portuguese-cased')

    tokenized_datasets = qa_dataset.map(
        prepare_train_features,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'stride': stride, 'padding_right': True},
        batched=True, remove_columns=qa_dataset['train'].column_names)

    model = BertForQuestionAnswering.from_pretrained(f'neuralmind/bert-{model_type}-portuguese-cased')

    batch_size = 3
    train_epochs = 3
    experiment_name = f'{train_epochs}_epochs_{model_type}_qa'

    training_args = TrainingArguments(
        output_dir=f'./results/{experiment_name}',  # output directory
        logging_dir=f'./logs/{experiment_name}',
        num_train_epochs=train_epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        # learning_rate=3e-05,    # default:  5e-05,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_steps=10,
        evaluation_strategy=IntervalStrategy.EPOCH,
        metric_for_best_model='f1',
        save_total_limit=1,
        fp16=True,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_datasets['train'],  # training dataset
        eval_dataset=tokenized_datasets['validation'],  # evaluation dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        #compute_metrics=compute_metrics
    )

    print(trainer.train())
    print(trainer.evaluate())


if __name__ == '__main__':
    run()