export test_file=scirex_dataset/release_data/test.jsonl
mkdir -p $test_output_folder

# These scripts require you've already generated files for NER and cluster predictions
if [ ! -f $test_output_folder/cluster_predictions.jsonl ]; then
    echo "Need to generate $test_output_folder/cluster_predictions.jsonl, from the general SciREX prediction script"
    exit 1
elif [ ! -f $test_output_folder/ner_predictions.jsonl ]; then
    echo "Need to generate $test_output_folder/ner_predictions.jsonl, from the general SciREX prediction script"
    exit 1
fi

echo "Predicting Salient Mentions"
python3.7 scirex/predictors/predict_salient_mentions.py \
$scirex_archive \
scirex_dataset/release_data/test.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$cuda_device

echo "Predicting Salient Clustering "
python3.7  scirex/predictors/predict_salient_clusters.py \
$test_output_folder/cluster_predictions.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl

echo "Evaluating on all Predicted steps "
python3.7  scirex/evaluation_scripts/salient_only_evaluate.py \
--gold-file $test_file \
--ner-file $test_output_folder/ner_predictions.jsonl \
--salient-mentions-file $test_output_folder/salient_mentions_predictions.jsonl \
--clusters-file $test_output_folder/salient_clusters_predictions.jsonl \
--salient-mentions-file $test_output_folder/salient_mentions_predictions.jsonl
