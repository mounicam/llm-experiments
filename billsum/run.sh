# VERSION=$1

# python run_inference.py --input_file dataset/org_with_cefr_labels/us_test.jsonl  --output_file \
# experiments/gemma_1b_oob/us_test.jsonl

# python run_inference.py --input_file dataset/org_with_cefr_labels/org_splits/ca_test_data_final_OFFICIAL.jsonl  --output_file \
# experiments/gemma_1b_oob/ca_test_data_final_OFFICIAL.jsonl

python run_inference.py --input_file dataset/org_with_cefr_labels/us_RL_train_4k.jsonl  --output_file \
experiments/gemma_1b_oob/us_RL_train_4k.jsonl --rollouts

python run_inference.py --input_file dataset/org_with_cefr_labels/us_RL_dev_1k.jsonl  --output_file \
experiments/gemma_1b_oob/us_RL_dev_1k.jsonl --rollouts

python run_metrics.py --input_file experiments/gemma_1b_oob/us_test.jsonl --output_file \
experiments/gemma_1b_oob/us_test_w_metrics.jsonl

python run_metrics.py --input_file experiments/gemma_1b_oob/ca_test_data_final_OFFICIAL.jsonl --output_file \
experiments/gemma_1b_oob/ca_test_data_final_OFFICIAL_w_metrics.jsonl

python run_metrics.py --input_file experiments/gemma_1b_oob/us_RL_train_4k.jsonl --output_file \
experiments/gemma_1b_oob/us_RL_train_4k_w_metrics.jsonl

python run_metrics.py --input_file experiments/gemma_1b_oob/us_RL_dev_1k.jsonl --output_file \
experiments/gemma_1b_oob/us_RL_dev_1k_w_metrics.jsonl

# =========================================================================================================================

# python run_inference.py --input_file dataset/org_with_cefr_labels/us_test.jsonl  --output_file \
# experiments/gemma_1b_sft/$VERSION/us_test.jsonl --model_name models/gemma_sft/$VERSION/merged

# python run_metrics.py --input_file experiments/gemma_1b_sft/$VERSION/us_test.jsonl --output_file \
# experiments/gemma_1b_sft/$VERSION/us_test_w_metrics.jsonl 

# python run_inference.py --input_file dataset/org_with_cefr_labels/org_splits/ca_test_data_final_OFFICIAL.jsonl  --output_file \
# experiments/gemma_1b_sft/$VERSION/ca_test_data_final_OFFICIAL.jsonl --model_name models/gemma_sft/$VERSION/merged

# python run_metrics.py --input_file experiments/gemma_1b_sft/$VERSION/ca_test_data_final_OFFICIAL.jsonl --output_file \
# experiments/gemma_1b_sft/$VERSION/ca_test_data_final_OFFICIAL_w_metrics.jsonl

# python run_metrics.py --input_file experiments/gemma_1b_sft/$VERSION/us_test_w_metrics.jsonl 

# python run_metrics.py --input_file experiments/gemma_1b_sft/$VERSION/ca_test_data_final_OFFICIAL_w_metrics.jsonl

# python run_inference.py --input_file dataset/org_with_cefr_labels/us_RL_train_4k.jsonl  --output_file \
# experiments/gemma_1b_sft/$VERSION/us_RL_train_4k.jsonl --model_name models/gemma_sft/$VERSION/merged --rollouts

# python run_inference.py --input_file dataset/org_with_cefr_labels/us_RL_dev_1k.jsonl  --output_file \
# experiments/gemma_1b_sft/$VERSION/us_RL_dev_1k.jsonl --model_name models/gemma_sft/$VERSION/merged --rollouts

# python run_metrics.py --input_file experiments/gemma_1b_sft/$VERSION/us_RL_train_4k.jsonl --output_file \
# experiments/gemma_1b_sft/$VERSION/us_RL_train_4k_w_metrics.jsonl

# python run_metrics.py --input_file experiments/gemma_1b_sft/$VERSION/us_RL_dev_1k.jsonl --output_file \
# experiments/gemma_1b_sft/$VERSION/us_RL_dev_1k_w_metrics.jsonl

