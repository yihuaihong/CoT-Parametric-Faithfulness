#!/bin/bash

# ======================
# 配置区域 - 按需修改
# ======================

SEEDS=(5555 6666 7777 8888 9999)
MODELS=("gemma-2-9b-it" "Meta-Llama-3-8B-Instruct" "Qwen3-8B")

DATASET_NAME="TwoHopFact"
BATCH_SIZE=64
DEVICE=0

EVAL_FLAGS="--eval_acc --use_cot_prompt"
# EVAL_FLAGS="--eval_cpf --use_cot_prompt"

LOG_DIR="logs"
RESULT_CSV="results/results_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p "$LOG_DIR" results

# CSV header
echo "model,seed,dataset,sample_num,metric,value" > "$RESULT_CSV"

# ======================
# 主循环
# ======================

TOTAL=$((${#SEEDS[@]} * ${#MODELS[@]}))
COUNT=0

echo "=============================="
echo "Total runs: $TOTAL"
echo "Seeds: ${SEEDS[*]}"
echo "Models: ${MODELS[*]}"
echo "Dataset: $DATASET_NAME | sample_num: $SAMPLE_NUM | batch_size: $BATCH_SIZE"
echo "=============================="

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        LOG_FILE="${LOG_DIR}/${DATASET_NAME}_s${MODEL}_seed${SEED}.log"

        echo ""
        echo "[${COUNT}/${TOTAL}] Model: $MODEL | Seed: $SEED"
        echo "  Log -> $LOG_FILE"

# SAMPLE_NUM 为空时不传参数，使用默认值 0（跑全部数据）
        SAMPLE_NUM_ARG=""
        if [ -n "$SAMPLE_NUM" ]; then
            SAMPLE_NUM_ARG="--sample_num $SAMPLE_NUM"
        fi

        python cpf_evaluation.py \
            $EVAL_FLAGS \
            --model_name "$MODEL" \
            --seed "$SEED" \
            --dataset_name "$DATASET_NAME" \
            $SAMPLE_NUM_ARG \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            2>&1 | tee "$LOG_FILE"

        EXIT_CODE="${PIPESTATUS[0]}"
        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "  [ERROR] Run failed for Model=$MODEL Seed=$SEED. Check $LOG_FILE"
            echo "$MODEL,$SEED,$DATASET_NAME,$SAMPLE_NUM,ERROR,FAILED" >> "$RESULT_CSV"
        else
            echo "  [DONE] Model=$MODEL Seed=$SEED"

            # 从 log 中提取数值结果并写入 CSV
            # 兼容 accuracy_results 和 CPF_results 两种格式
            RESULT_LINE=$(grep -E "accuracy_results|CPF_results" "$LOG_FILE" | tail -1)
            METRIC=$(echo "$RESULT_LINE" | grep -oE "accuracy_results|CPF_results")
            VALUE=$(echo "$RESULT_LINE" | grep -oE "[0-9]+\.[0-9]+" | tail -1)

            if [ -n "$VALUE" ]; then
                echo "$MODEL,$SEED,$DATASET_NAME,$SAMPLE_NUM,$METRIC,$VALUE" >> "$RESULT_CSV"
                echo "  [RESULT] $METRIC = $VALUE"
            else
                echo "  [WARN] Could not parse result from log. Raw: $RESULT_LINE"
                echo "$MODEL,$SEED,$DATASET_NAME,$SAMPLE_NUM,$METRIC,PARSE_FAILED" >> "$RESULT_CSV"
            fi
        fi
    done
done

echo ""
echo "=============================="
echo "All $TOTAL runs completed."
echo "Results saved to: $RESULT_CSV"
echo "=============================="

# ======================
# 汇总结果（从 CSV 中打印）
# ======================

echo ""
echo "====== Results Summary ======"
column -t -s',' "$RESULT_CSV"