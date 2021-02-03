export test_file=scirex_dataset/release_data/test.jsonl

mkdir -p $test_output_folder

echo "Predicting Salient Mentions"
python scirex/predictors/predict_salient_mentions.py \
$salient_only_archive \
scirex_dataset/release_data/test.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$cuda_device

echo "Predicting Salient Clustering "
python scirex/predictors/predict_salient_clusters.py \
$test_file \
$test_output_folder/salient_mentions_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl

echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/salient_only_evaluate.py \
--gold-file $test_file \
--ner-file $test_file \
--clusters-file $test_output_folder/salient_clusters_predictions.jsonl \
--salient-mentions-file $test_output_folder/salient_mentions_predictions.jsonl
