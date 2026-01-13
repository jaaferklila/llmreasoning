#!/bin/bash
#SBATCH --job-name=maskLLM_eval
#SBATCH --output=logs_maskLLM/%x_%j.out
#SBATCH --error=logs_maskLLM/%x_%j.err
#SBATCH --mem=30000
#SBATCH --time=2-00:00:00
#SBATCH -p  gpuh100p
#SBATCH --gres=gpu:2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=klilajafer@gmail.com


source ~/miniforge3/etc/profile.d/conda.sh
conda activate decker
echo "Début du job sur l'hôte : $(hostname)"
echo "Démarrage du traitement..."
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1 
unset HF_HUB_OFFLINE
# ================= Run Python Script ================= #

#MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b")
MODEL_NAMES=("qwen2.5-32b" "llama-3.1-70B")
DATASET_NAMES=("strategyqa" "MQuAKE-CF-3k-v2")
#DATASET_NAMES=("MQuAKE-CF-3k-v2" "strategyqa")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        echo "Evaluating model $MODEL_NAME on dataset $DATASET_NAME"
        python answerMask.py \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME"
    done
done

echo "Done. Job finished."






