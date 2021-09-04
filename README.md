# Debiasing KnowBERT using the knowledge graph embeddings
Code for the report *Identifying and Mitigating Gender Bias in Knowledge Enhanced Contextual Word Representations*


## Requirements
- Python 3.6.7 or higher
- PyTorch 1.2.0

## Getting started
```
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
pip install --editable .
```


## Reproducing results

### Debiasing KnowBERT

### Measuring bias in KnowBERT representation

Modify the path to the directory of your model.tar.gz files in *bert_expose_bias_with_prior.py*.
```python
archive_file = os.path.join('path_to_your_model_directory', 'model.tar.gz')
```

Then, run the following command:
```console
python3 measuring/exposing_bias_bert.py --output_dir path_to_the_output_dir
```

It will create pdfs of pie charts that shows the
percentage of attributes that are more associated with each gender, in the mask prediction on the model.








