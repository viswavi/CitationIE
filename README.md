# ScigraphIE

## Citation Graph Preparation

## Model training
To install all dependencies for model training, please follow the instructions in the [SciREX repository](https://github.com/viswavi/SciREX). In particular, you will need to download an official release of the SciREX dataset, and download a trained copy of SciBERT to your machine. All commands listed here assume you are at the root of the SciREX directory (which is a submodule of this repository).

We provide scripts for training 4 kinds of models:
- End-to-End Information Extraction (`scirex/commands/train_scirex_model.sh`)
- Mention Identification Only (`scirex/commands/train_ner_only.sh`)
- Salient Entity Classification Only (`scirex/commands/train_salient_classification_only.sh`)
- Relation Extraction Only (`scirex/commands/train_relations_only.sh`)

### Training Scripts
For training baseline models for each, run the following commands from.
```
export BERT_BASE_FOLDER=<PATH_TO_SCIBERT>
CUDA_DEVICE=0 bash <TRAINING_SCRIPT> main
```

In order to train *citation graph-enhanced models* for any of the above, set the following environment variables before running the above commands:
```
export use_citation_graph_embeddings=true
export citation_embedding_file=graph_embeddings/embeddings.npy
export citation_embedding_file=graph_embeddings/scirex_docids.json
```

In order to train *citance-enhanced models*, do `export DATA_BASE_PATH=scirex_dataset/data_with_citances` before running your desired training command.

### Evaluation Scripts
#### Primary
Depending on the type of model you trained, use one of the following evaluation scripts
- `scirex/commands/predict_scirex_model.sh`
- `scirex/commands/predict_ner_only_gold.sh`
- `scirex/commands/predict_salient_only_gold.sh`
- `scirex/commands/predict_salient_only_gold.sh`

And then, run:
```
export BERT_BASE_FOLDER=scibert_scivocab_uncased
export PYTHONPATH=$PYTHONPATH:$(pwd)
export scirex_archive=<PATH TO TRAINED MODEL>>
export scirex_coreference_archive=<PATH TO TRAINED COREFERENCE MODEL
export cuda_device=2
export test_output_folder=<PREDICTION OUTPUTS DIRECTORY>
bash scirex/commands/predict_salient_only_gold.sh
```

#### Extra
We also have written a number of additional evaluation scripts for the results in our paper, which are all in `scirex/evaluation_scripts`:
*Bootstrap Evaluation Scripts*
relation_bootstrap_comparison.py
relation_bootstrap_comparison_multi_model.py
salient_bootstrap_comparison.py
salient_bootstrap_comparison_multi_model.py

*Bucketing and Visualization Scripts*
relation_bootstrap_comparison_bucketing_on_cluster_distance.py
relation_bootstrap_comparison_bucketing_on_graph_degree.py
relation_bootstrap_comparison_bucketing_on_graph_degree_multi_model.py

salient_bootstrap_comparison_bucketing_on_graph_degree.py
