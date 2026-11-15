#!/usr/bin/env bash
# Serve one local VLM with vLLM (OpenAI-compatible API on 127.0.0.1). Usage: scripts/serve_vlm.sh <preset> <gpu[,gpu]>
#   preset               weights (HF id; or set CE_CKPT_<NAME> to a local copy)   port   notes
#   qwen3-vl-32b         Qwen/Qwen3-VL-32B-Instruct                                8000
#   medgemma-27b         google/medgemma-27b-it                                    8010
#   internvl3.5-38b-fp8  OpenGVLab/InternVL3_5-38B                                 8020   one 80 GB GPU, weights quantised to FP8
#                                                                                         by vLLM (setting used in the paper)
#   internvl3.5-38b      OpenGVLab/InternVL3_5-38B                                 8020   bf16, tensor parallel over 2 GPUs
# Tested with vLLM 0.14. Context 24576 tokens, at most 16 images per prompt, bf16. The tool-call parser is needed only by
# the DeepTumorVQA reference agent (scripts/dtvqa_agent.py, native function calling).
set -euo pipefail
PRESET=$1; GPU=$2
TP=1; EXTRA=""; LEN=24576
case $PRESET in
  qwen3-vl-32b) P=${CE_CKPT_QWEN:-Qwen/Qwen3-VL-32B-Instruct}; PORT=8000
                EXTRA="--enable-auto-tool-choice --tool-call-parser hermes" ;;
  medgemma-27b) P=${CE_CKPT_MEDGEMMA:-google/medgemma-27b-it}; PORT=8010 ;;
  internvl3.5-38b-fp8) P=${CE_CKPT_INTERNVL:-OpenGVLab/InternVL3_5-38B}; PORT=8020; NAME=internvl3.5-38b
                       EXTRA="--trust-remote-code --quantization fp8 --enable-auto-tool-choice --tool-call-parser hermes" ;;
  internvl3.5-38b) P=${CE_CKPT_INTERNVL:-OpenGVLab/InternVL3_5-38B}; PORT=8020; TP=2; EXTRA="--trust-remote-code" ;;
  *) echo "unknown preset $PRESET"; exit 2 ;;
esac
NAME=${NAME:-$PRESET}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec vllm serve "$P" --served-model-name "$NAME" --host 127.0.0.1 --port $PORT --tensor-parallel-size $TP $EXTRA --dtype bfloat16 \
  --max-model-len $LEN --gpu-memory-utilization 0.92 --max-num-seqs 8 --limit-mm-per-prompt '{"image": 16}'
