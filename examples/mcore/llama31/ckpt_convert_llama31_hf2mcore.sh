# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf llama2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir /afs-c-mtc/gongruihao/Meta-Llama-3.1-8B/ \
   --save-dir ./model_weights/llama31-mcore/ \
   --tokenizer-model /afs-c-mtc/gongruihao/Meta-Llama-3.1-8B/  # --num-layer-list 17,20,22,21 等参数根据模型需求添加
