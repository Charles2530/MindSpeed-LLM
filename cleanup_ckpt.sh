#!/bin/bash
# cleanup_ckpt.sh
CKPT_DIR="$(cd "$(dirname "$0")/saved_ckpt" && pwd)"   # 自动定位到 saved_ckpt 绝对路径
PREFIX="iter_"

mapfile -t ALL < <(find "$CKPT_DIR" -maxdepth 1 -type d -name "${PREFIX}*" | xargs -n1 basename | sort -t_ -k2 -nr)

TOTAL=${#ALL[@]}
KEEP=2

if (( TOTAL > KEEP )); then
    echo "检测到 $TOTAL 个 ${PREFIX}* 检查点，准备保留最大的 $KEEP 个。"
    # 从第 $KEEP 个开始删除
    for (( i=KEEP; i<TOTAL; i++ )); do
        DIR_TO_RM="$CKPT_DIR/${ALL[$i]}"
        echo "[$(date +'%F %T')] 删除 $DIR_TO_RM"
        rm -rf "$DIR_TO_RM"
    done
else
    echo "当前只有 $TOTAL 个 ${PREFIX}* 检查点，无需清理。"
fi
