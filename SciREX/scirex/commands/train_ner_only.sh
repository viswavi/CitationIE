if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/ner_only_main.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

if [ -z ${DATA_BASE_PATH+x} ]; then export DATA_BASE_PATH=scirex_dataset/release_data; fi

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

if [ -z ${random_seed+x} ]; then export random_seed=13370; fi
if [ -z ${numpy_seed+x} ]; then export numpy_seed=1337; fi
if [ -z ${pytorch_seed+x} ]; then export pytorch_seed=133; fi

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_ner_only_${pytorch_seed}/$1}

export bert_fine_tune=10,11,pooler

nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
