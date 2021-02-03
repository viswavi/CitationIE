export test_file=scirex_dataset/release_data/test.jsonl
mkdir -p $test_output_folder

echo "Predicting NER"
python scirex/predictors/predict_ner.py \
$scirex_archive \
$test_file \
$ner_output_folder/ner_predictions.jsonl \
$cuda_device

echo "Evaluating on NER only"
python scirex/evaluation_scripts/ner_evaluate.py \
--gold-file $test_file \
--ner-file $ner_output_folder/ner_predictions.jsonl