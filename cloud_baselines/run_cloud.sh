#!/bin/bash
#   bash run_cloud.sh test        # 环境检查 + 2 个数据集 × 1 seed × w=16 的快速测试（几分钟）
#   bash run_cloud.sh status      # 看进度
#   bash run_cloud.sh merge       # 跑完后合并结果，打印速览表
# 中途断了：再执行一次 start 即可，已完成的行会跳过。
cd "$(dirname "$0")"
mkdir -p logs
CMD=${1:-test}
N=${2:-4}

case "$CMD" in
  test)
    python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
    python -c "import pandas" 2>/dev/null || pip install -q pandas
    ls data/*.npz >/dev/null 2>&1 || echo "!! 没找到 data/*.npz，会尝试用 aeon 联网下载（需要 pip install aeon）"
    python -u fscil_deep.py --quick --widths 16 --out ../results/results_deep_quick.csv 2>&1 | tee logs/quick.log
    ;;
  start)
    for i in $(seq 0 $((N-1))); do
      nohup python -u fscil_deep.py --shard $i --n-shards $N > logs/shard$i.log 2>&1 &
      echo "启动分片 $i/$N，日志 logs/shard$i.log"
    done
    echo "$N" > logs/n_shards
    ;;
  status)
    N=$(cat logs/n_shards 2>/dev/null || echo "$N")
    echo "运行中的进程：$(pgrep -fc 'fscil_deep.py --shard')"
    for i in $(seq 0 $((N-1))); do
      f=../results/results_deep_v6_shard$i.csv
      rows=$( [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0 )
      echo "分片 $i：已完成 $rows 行 | 最近：$(grep '###' logs/shard$i.log 2>/dev/null | tail -1)"
    done
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    ;;
  merge)
    python merge_results.py
    ;;
  *)
    echo "用法：bash run_cloud.sh [test|start N|status|merge]"
    ;;
esac
