#!/bin/zsh
# Mac mini 本地 Qwen 模型测试脚本
cd ~

# 修改为本地地址
sed -i '' 's|http://xinmeitideMac-mini.local:8000/v1|http://localhost:8000/v1|g' ~/ai_annotator_macqwen.py
echo "=== BASE_URL ==="
grep 'BASE_URL' ~/ai_annotator_macqwen.py | head -1

# 创建测试目录
mkdir -p ~/test_qwen/img ~/test_qwen/label_in ~/test_qwen/label_out
cp ~/9.1p/OreXis945_f0001.png ~/test_qwen/img/
cp ~/9.1d/OreXis945_f0001.txt ~/test_qwen/label_in/
echo ""
echo "=== 测试图片 ==="
ls -lh ~/test_qwen/img/

# 启动测试（核查模式+快速模式+1线程+1张图）
echo ""
echo "=== 启动测试 ==="
export AI_MODE=review
export AI_INPUT="$HOME/test_qwen/img"
export AI_SRC_LABEL="$HOME/test_qwen/label_in"
export AI_OUT_LABEL="$HOME/test_qwen/label_out"
export AI_THINK=0
export AI_THREADS=1

nohup python3 -u ~/ai_annotator_macqwen.py > ~/test_qwen_log.txt 2>&1 &
PID=$!
echo "进程PID: $PID"
echo "日志文件: ~/test_qwen_log.txt"

sleep 10
echo ""
echo "=== 10秒后日志 ==="
cat ~/test_qwen_log.txt
