python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B-Thinking \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --dtype bfloat16 \
    --trust-remote-code \
    --port 8000