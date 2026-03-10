export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-32B-Thinking \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 1}' \
    --served-model-name qwen3-vl-thinking > vllm.log 2>&1 &
    