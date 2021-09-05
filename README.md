# Debiasing KnowBERT using the knowledge graph embeddings
Code for the report *Identifying and Mitigating Gender Bias in Knowledge Enhanced Contextual Word Representations* by Chani Jung.



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

<!-- ### Debiasing KnowBERT -->

Tutorial __How to pretrain pure/debiased KnowBERT__ is referenced from https://github.com/allenai/kb and partly modified for adding debiasing method.

### How to pretrain pure/debiased KnowBert

We can build KnowBert-WordNet, KnowBert-Wiki, and KnowBert-WordNet+Wiki using the code and resources provided by https://github.com/allenai/kb.
However, I only used KnowBert-WordNet for this work.

Roughly speaking, the process to fine tune BERT into KnowBert is:

1. Prepare your corpus.
2. Prepare the knowledge bases (not necessary if you are using Wikipedia or WordNet as we have already prepared these).
3. For each knowledge base:
    1. Pretrain the entity linker while freezing everything else.
    2. Fine tune all parameters (except entity embeddings).


#### Prepare your corpus.
1. Sentence tokenize your training corpus using spacy, and prepare input files for next-sentence-prediction sampling.  Each file contains one sentence per line with consecutive sentences on subsequent lines and blank lines separating documents.
2. Run `bin/create_pretraining_data_for_bert.py` to group the sentences by length, do the NSP sampling, and write out files for training.
3. Reserve one or more of the training files for heldout evaluation.

#### Prepare the input knowledge bases.
1. We have already prepared the knowledge bases for Wikipedia and WordNet.  The necessary files will be automatically downloaded as needed when running evaluations or fine tuning KnowBert.
2. If you would like to add an additional knowledge source to KnowBert, these are roughly the steps to follow:

    A. Compute entity embeddings for each entity in your knowledge base.
    B. Write a candidate generator for the entity linkers.  Use the existing WordNet or Wikipedia generators as templates.

3.  Our Wikipedia candidate dictionary list and embeddings were extracted from [End-to-End Neural Entity Linking, Kolitsas et al 2018](https://github.com/dalab/end2end_neural_el) via a manual process.

4. Our WordNet candidate generator is rule based (see code).  The embeddings were computed via a multistep process that combines [TuckER](https://arxiv.org/abs/1901.09590) and [GenSen](https://github.com/Maluuba/gensen) embeddings.  The prepared files contain everything needed to run KnowBert and include:

    A. `entities.jsonl` - metadata about WordNet synsets.
    
    B. `wordnet_synsets_mask_null_vocab.txt` and `wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5` - vocabulary file and embedding file for WordNet synsets.
    
    C. `semcor_and_wordnet_examples.json` annotated training data combining SemCor and WordNet examples for supervising the WordNet linker.

5. If you would like to generate these files yourself from scratch, follow these steps.

    A. Extract the WordNet metadata and relationship graph.
        ```
        python bin/extract_wordnet.py --extract_graph --entity_file $WORKDIR/entities.jsonl --relationship_file $WORKDIR/relations.txt
        ```
        
    B. Download the [Words-in-Context dataset](https://pilehvar.github.io/wic/) to exclude from the extracted WordNet example usages.
        ```
        WORKDIR=.
        cd $WORKDIR
        wget https://pilehvar.github.io/wic/package/WiC_dataset.zip
        unzip WiC_dataset.zip
        ```
        
    C. Download the [word sense diambiguation data](http://lcl.uniroma1.it/wsdeval/):
        ```
        cd $WORKDIR
        wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
        unzip WSD_Evaluation_Framework.zip
        ```
        
    D. Convert the WSD data from XML to jsonl, and concatenate all evaluation files for easy evaluation:
        ```
        mkdir $WORKDIR/wsd_jsonl
        python bin/preprocess_wsd.py --wsd_framework_root $WORKDIR/WSD_Evaluation_Framework  --outdir $WORKDIR/wsd_jsonl
        cat $WORKDIR/wsd_jsonl/semeval* $WORKDIR/wsd_jsonl/senseval* > $WORKDIR/semeval2007_semeval2013_semeval2015_senseval2_senseval3.json
        ```
        
    E. Extract all the synset example usages from WordNet (after removing sentences from WiC heldout sets):
        ```
        python bin/extract_wordnet.py --extract_examples_wordnet --entity_file $WORKDIR/entities.jsonl --wic_root_dir $WORKDIR --wordnet_example_file $WORKDIR/wordnet_examples_remove_wic_devtest.json
        ```
        
    F. Combine WordNet examples and definitions with SemCor for training KnowBert:
        ```
        cat $WORKDIR/wordnet_examples_remove_wic_devtest.json $WORKDIR/wsd_jsonl/semcor.json > $WORKDIR/semcor_and_wordnet_examples.json
        ```
        
    G. Create training and test splits of the relationship graph.
        ```
        python bin/extract_wordnet.py --split_wordnet --relationship_file $WORKDIR/relations.txt --relationship_train_file $WORKDIR/relations_train99.txt --relationship_dev_file $WORKDIR/relations_dev01.txt
        ```
        
    H. Train TuckER embeddings on the extracted graph.  The configuration files uses relationship graph files on S3, although you can substitute them for the files generated in the previous step by modifying the configuration file.
        ```
        allennlp train -s $WORKDIR/wordnet_tucker --include-package kb.kg_embedding --file-friendly-logging training_config/pretraining/wordnet_tucker.json
        ```
        
    I. Generate a vocabulary file useful for WordNet synsets with special tokens
        ```
        python bin/combine_wordnet_embeddings.py --generate_wordnet_synset_vocab --entity_file $WORKDIR/entities.jsonl --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt
        ```
        
    J. Get the [GenSen](https://github.com/Maluuba/gensen) embeddings from each synset definition.  First install the code from this link.  Then run
        ```
        python bin/combine_wordnet_embeddings.py --generate_gensen_embeddings --entity_file $WORKDIR/entities.jsonl --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt --gensen_file $WORKDIR/gensen_synsets.hdf5
        ```
        
    K. Extract the TuckER embeddings for the synsets from the trained model
        ```
        python bin/combine_wordnet_embeddings.py --extract_tucker --tucker_archive_file $WORKDIR/models/wordnet_tucker/model.tar.gz --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt --tucker_hdf5_file $WORKDIR/pure/tucker_embeddings.hdf5
        ```
        
    L. Debias tucker embeddings.
    ```
    python bin/combine_wordnet_embeddings.py --debias_tucker --tucker_archive_file models/wordnet_tucker/model.tar.gz --tucker_hdf5_file tucker_embeddings/pure/tucker_embeddings.hdf5 --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt --deb_tucker_hdf5_file tucker_embeddings/debiased/tucker_embeddings.hdf5
    ```
        
    M. Finally combine the TuckER and GenSen embeddings into one file
        ```
        python bin/combine_wordnet_embeddings.py --combine_tucker_gensen --tucker_hdf5_file tucker_embeddings/debiased/tucker_embeddings.hdf5 --gensen_file $WORKDIR/gensen_synsets.hdf5 --all_embeddings_file tucker_gensen_embeddings/debiased/tucker_gensen_embeddings.hdf5
        ```

#### Pretraining the entity linkers

This step pretrains the entity linker while freezing the rest of the network using only supervised data.

Config files are  `training_config/pretraining/knowbert_wordnet_linker.jsonnet` and
`training_config/pretraining/debiased_knowbert_wordnet_linker.jsonnet`.

To train the pure WordNet linker for KnowBert-WordNet run:
```
allennlp train -s OUTPUT_DIRECTORY --file-friendly-logging --include-package kb.include_all training_config/pretraining/knowbert_wordnet_linker.jsonnet
```

To train the debiased WordNet linker for debiased KnowBert-WordNet run:
```
allennlp train -s OUTPUT_DIRECTORY --file-friendly-logging --include-package kb.include_all training_config/pretraining/debiased_knowbert_wordnet_linker.jsonnet
```

#### Fine tuning BERT

After pre-training the entity linkers from the step above, fine tune BERT.
The pretrained models in our paper were trained on a single GPU with 24GB of RAM.  For multiple GPU training, change `cuda_device` to a list of device IDs.

Config files are `training_config/pretraining/knowbert_wordnet.jsonnet` and
`training_config/pretraining/debiased_knowbert_wordnet.jsonnet`.

Before training, modify the following keys in the config file (or use `--overrides` flag to `allennlp train`):

* `"language_modeling"`
* `"model_archive"` to point to the `model.tar.gz` from the previous linker pretraining step.


### Measuring bias in KnowBERT representation (Table 1)

Modify the path to the directory of your model.tar.gz files in *bert_expose_bias_with_prior.py*.
```python
archive_file = os.path.join('path_to_your_model_directory', 'model.tar.gz')
```

Then, run the following command:
```console
python measuring/exposing_bias_bert.py --output_dir path_to_the_output_dir
```

It will create pdfs of pie charts that shows the
percentage of attributes that are more associated with each gender, in the mask prediction on the model.


### How to run intrinisic evaluation

In this work, only heldout perplexity is evaluated for intrinsic evaluation.
However, you can test more evaluation tasks for your KnowBert model, which can be found in https://github.com/allenai/kb

#### Heldout perplexity (Table 2)

Download the [heldout data](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/data/wikipedia_bookscorpus_knowbert_heldout.txt). Then run:

```
MODEL_ARCHIVE=..location of model
HELDOUT_FILE=wikipedia_bookscorpus_knowbert_heldout.txt
python bin/evaluate_perplexity.py -m $MODEL_ARCHIVE -e $HELDOUT_FILE
```

The heldout perplexity is key `exp(lm_loss_wgt)`.






