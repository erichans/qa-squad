from transformers import BertForQuestionAnswering, BertTokenizerFast, default_data_collator, \
    TrainingArguments, IntervalStrategy, Trainer, EvalPrediction
from datasets import load_dataset, load_metric
from pathlib import Path
from preprocessing import flatten_data, prepare_train_features
from postprocessing import postprocess_qa_predictions

from typing import Tuple, List, Dict


# dataset SQuAD v1 pt_BR: https://drive.google.com/file/d/1Q0IaIlv2h2BC468MwUFmUST0EyN7gNkn/view

def run():

    train_file = './data/flat_squad-train-v1.1.json' \
        if Path('./data/flat_squad-train-v1.1.json').exists() else flatten_data('./data/squad-train-v1.1.json')

    validation_file = './data/flat_squad-dev-v1.1.json' \
        if Path('./data/flat_squad-dev-v1.1.json').exists() else flatten_data('./data/squad-dev-v1.1.json')

    qa_dataset = load_dataset('json', data_files={'train': train_file, 'validation': validation_file}, field='data')

    model_type = 'base'
    # model_type = 'large'

    max_length = 512
    stride = 128

    tokenizer = BertTokenizerFast.from_pretrained(f'neuralmind/bert-{model_type}-portuguese-cased', local_files_only=True)

    tokenized_datasets = qa_dataset.map(
        prepare_train_features,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'stride': stride, 'padding_right': True},
        batched=True, remove_columns=qa_dataset['train'].column_names)

    model = BertForQuestionAnswering.from_pretrained(f'neuralmind/bert-{model_type}-portuguese-cased')

    metric = load_metric('squad')

    def compute_metrics(p: EvalPrediction) -> Dict:
        final_predictions = postprocess_qa_predictions(qa_dataset['validation'],
                                                       tokenized_datasets['validation'], p.predictions)

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in qa_dataset['validation']]
        return metric.compute(predictions=formatted_predictions, references=references)

    batch_size = 16
    train_epochs = 2
    experiment_name = f'{train_epochs}_epochs_{model_type}_qa'

    training_args = TrainingArguments(
        output_dir=f'./results/{experiment_name}',  # output directory
        # logging_dir=f'./logs/{experiment_name}',
        num_train_epochs=train_epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training

        per_device_eval_batch_size=batch_size*2,  # batch size for evaluation
        learning_rate=4.25e-05,    # default:  5e-05,
        # warmup_steps=500,  # number of warmup steps for learning rate scheduler
        warmup_ratio=0.0,
        weight_decay=0.01,  # strength of weight decay
        logging_steps=10,
        evaluation_strategy=IntervalStrategy.EPOCH,
        metric_for_best_model='f1',
        save_total_limit=1,
        fp16=False,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_datasets['train'],  # training dataset
        eval_dataset=tokenized_datasets['validation'],  # evaluation dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    print(trainer.train())
    print(trainer.evaluate())


import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning


old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


if __name__ == '__main__':
    with no_ssl_verification():
        run()

