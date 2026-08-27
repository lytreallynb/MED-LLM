#!/bin/bash
# Full v2 training and eval chain for a Kaggle P100/T4 session.
# Run from the repo root: nohup bash scripts/kaggle_chain.sh > chain.log 2>&1 &
# The final step zips results and adapters into the parent directory so they
# can be downloaded before the session dies (lesson from the Colab run that
# lost its artifacts to a runtime recycle).
set -x

pip -q install transformers==4.57.6 trl peft datasets accelerate
pip uninstall -y torchao
nvidia-smi -L

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m medllm.finetune train \
    --train-data data/finetune/train.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir output/lora_grounding_v2 \
    --epochs 3 --batch-size 2 --lr 2e-4 || exit 1

python medllm/grpo_finetune.py \
    --train-data data/finetune/train.jsonl \
    --sft-adapter output/lora_grounding_v2/lora_adapter \
    --output-dir output/grpo_grounding_v2 \
    --max-steps 200 || exit 1

python -m medllm.eval_behavior --test-data data/finetune/test.jsonl \
    --output results/behavior_base_v2.json
python -m medllm.eval_behavior --test-data data/finetune/test.jsonl \
    --adapter output/lora_grounding_v2/lora_adapter \
    --output results/behavior_sft_v2.json
python -m medllm.eval_behavior --test-data data/finetune/test.jsonl \
    --base-adapter output/lora_grounding_v2/lora_adapter \
    --adapter output/grpo_grounding_v2/grpo_adapter \
    --output results/behavior_sft_grpo_v2.json

cd .. && zip -r medllm_v2_artifacts.zip \
    MED-LLM/results \
    MED-LLM/output/lora_grounding_v2/lora_adapter \
    MED-LLM/output/grpo_grounding_v2/grpo_adapter
echo CHAIN_DONE
