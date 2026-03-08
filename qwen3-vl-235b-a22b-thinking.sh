# vLLM 최신 버전이 설치되어 있다고 가정합니다. (pip install vllm)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B-Thinking \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --port 8000