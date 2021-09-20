# question-answering-squad-pt-br

Source code to fine tune BERTimbau on portuguese translated SQuAD v1.1 dataset from Deep Learning Brasil.
The train and dev datasets are in [data](data) folder.

To fine tune BERTimbau or other huggingface model, just run [main.py](main.py).

The train and evaluation code is based on the following notebooks: 
- [BERT Base](https://github.com/piegu/language-models/blob/master/colab_question_answering_BERT_base_cased_squad_v11_pt.ipynb) 
- [BERT Large](https://github.com/piegu/language-models/blob/master/question_answering_BERT_large_cased_squad_v11_pt.ipynb)

To run the predictions after training, use [main_predict.py](main_predict.py)
