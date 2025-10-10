# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# mkdir ./dataset

python ./preprocess_data.py \
  --input ./dataset/wikipedia/ \
  --tokenizer-name-or-path /afs_c_mtc/gongruihao/deepseek_910b/DeepSeek-V2-Lite/ \
  --output-prefix ./dataset/enwiki \
  --workers 128 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralPretrainHandler  \
  --json-keys text
