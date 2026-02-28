
OUTPUT_DIR=$1
MODEL_NAME=$2

# mkdir -p $OUTPUT_DIR/metrics

python run_inference.py --input_file experiments/v2/dataset/us_test_data_final_OFFICIAL_with_fkgl.jsonl \
                        --output_file $OUTPUT_DIR/us_test_data_final_OFFICIAL.jsonl \
                        --model_name $MODEL_NAME \
                        --metrics \
                        --metrics_output $OUTPUT_DIR/metrics/us_test_data_final_OFFICIAL.jsonl


python run_inference.py --input_file experiments/v2/dataset/ca_test_data_final_OFFICIAL_with_fkgl.jsonl \
                        --output_file $OUTPUT_DIR/ca_test_data_final_OFFICIAL.jsonl \
                        --model_name $MODEL_NAME \
                        --metrics \
                        --metrics_output $OUTPUT_DIR/metrics/ca_test_data_final_OFFICIAL.jsonl
