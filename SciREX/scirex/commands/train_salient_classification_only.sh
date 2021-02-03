if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/salient_only_main.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

if [ -z ${DATA_BASE_PATH+x} ]; then export DATA_BASE_PATH=scirex_dataset/release_data; fi

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_salient_only/$1}

export bert_fine_tune=10,11,pooler
export finetune_embedding=false
if [ -z citation_embedding_file ]; then
    citation_embedding_file=""
    doc_to_idx_mapping_file="";
fi
nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
use_citation_graph_embeddings=true \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
