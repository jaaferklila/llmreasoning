#!/bin/bash
#SBATCH --output=logs_LLM_Only/o%x.out
#SBATCH --error=logs_LLM_Only/e%x.err
#SBATCH --mem=30000
#SBATCH --time=2-00:00:00
#SBATCH -p  gpuh100p
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=llmOnly

# ================= Environment ================= #
source ~/miniforge3/etc/profile.d/conda.sh
conda activate decker

echo "Début du job sur l'hôte : $(hostname)"
echo "Démarrage du traitement..."

# ================= Offline HF Settings ================= #
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
#unset HF_HUB_OFFLINE  # à décommenter si vous voulez revenir en ligne

# ================= Run Python Script ================= #
# Définir les modèles et datasets (Bash array)
#MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b")
TIME_DIR="./output/LLM_Only/timing"
mkdir -p $TIME_DIR
MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b" "qwen2.5-32b" "llama-3.1-70B")
DATASET_NAMES=("strategyqa" "MQuAKE-CF-3k-v2")
#DATASET_NAMES=("strategyqa")

# Boucle sur tous les modèles et datasets
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do

        echo "Running $MODEL_NAME on $DATASET_NAME"

        START=$(date +%s)

        srun python LLM_Only.py \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME"

        END=$(date +%s)
        DURATION=$((END - START))

        HMS=$(printf "%02dh:%02dm:%02ds" \
            $((DURATION/3600)) \
            $(((DURATION%3600)/60)) \
            $((DURATION%60)))

        OUT_FILE="${TIME_DIR}/${MODEL_NAME}_${DATASET_NAME}_time.json"

        cat <<EOF > $OUT_FILE
{
  "model": "${MODEL_NAME}",
  "dataset": "${DATASET_NAME}",
  "time_seconds": ${DURATION},
  "time_hms": "${HMS}",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)"
}
EOF

        echo "Saved timing to $OUT_FILE"

    done
done

echo "All jobs finished."
