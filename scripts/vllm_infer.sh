#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # change to your available gpus

# datset_dir is the dir of dataset_info.json
python -m vllm_infer \
--model_name_or_path MianzhiPan/MOF-LLM \
--dataset mof_infer_demo \
--dataset_dir ../data \
--template qwen3 \
--cutoff_len 2048 \
--max_new_tokens 2048 \
--temperature 0.1 \
--top_p 0.6 \
--top_k 10