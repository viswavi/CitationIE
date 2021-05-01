export test_file=scirex_dataset/release_data/test.jsonl
mkdir -p test_output_folder

echo "Predicting Relations on gold clusters"
python scirex/predictors/predict_n_ary_relations.py \
$relations_only_archive \
$test_file \
$test_file \
$test_output_folder/relations_predictions_gold_clusters.jsonl \
$cuda_device

echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/relations_only_evaluate.py \
--gold-file $test_file \
--ner-file $test_file \
--clusters-file $test_file \
--relations-file $test_output_folder/relations_predictions_gold_clusters.jsonl