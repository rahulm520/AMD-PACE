# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer
from pace.utils.logging import PACE_INFO
import sys
import argparse

from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    LLMOperatorType,
    LLMBackendType,
    OperatorConfig,
)

from pace.server.model_list import MODEL_LIST

global PRELOAD_MODEL

TORCH_DTYPE = torch.bfloat16

app = FastAPI()

loaded_models = {}  # model_name -> (model, tokenizer)


def load_model(model_name):
    opconfig = OperatorConfig(
        **{
            LLMOperatorType.Norm: LLMBackendType.NATIVE,
            LLMOperatorType.QKVProjection: LLMBackendType.TPP,
            LLMOperatorType.Attention: LLMBackendType.JIT,
            LLMOperatorType.OutProjection: LLMBackendType.TPP,
            LLMOperatorType.MLP: LLMBackendType.TPP,
            LLMOperatorType.LMHead: LLMBackendType.NATIVE,
        }
    )
    if model_name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = LLMModel(
            model_name,
            dtype=TORCH_DTYPE,
            kv_cache_type=KVCacheType.BMC,
            opconfig=opconfig,
        )
        loaded_models[model_name] = (model, tokenizer)
    return loaded_models[model_name]


class InferenceRequest(BaseModel):
    model: str
    prompts: List[str]
    gen_kwargs: Dict[str, Any]


@app.on_event("startup")
def preload_requested_model():
    PACE_INFO(f"Preloading model: {PRELOAD_MODEL}")
    load_model(PRELOAD_MODEL)


@app.post("/infer")
def infer(req: InferenceRequest):
    if req.model not in MODEL_LIST:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{req.model}' is not available. Available models: {MODEL_LIST}",
        )
    model, tokenizer = load_model(req.model)

    prompts = req.prompts
    gen_kwargs = req.gen_kwargs

    input_encoded = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding="longest"
    )
    sampling_config = SamplingConfig(**gen_kwargs)
    output = model.generate(input_encoded, sampling_config=sampling_config)
    outputs = []
    for i in range(len(prompts)):
        output_text = tokenizer.decode(
            output.output_token_ids[i], skip_special_tokens=True
        )
        # Remove prompt from output if present
        if output_text.startswith(prompts[i]):
            output_text = output_text[len(prompts[i]) :]
        outputs.append(output_text.strip())
    return {"outputs": outputs}


@app.get("/get_models")
def list_models():
    return {"data": [{"id": model_name} for model_name in MODEL_LIST]}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACE Inference Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the server (default: 8000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_LIST[0],
        help=f"Model to preload (default: {MODEL_LIST[0]})",
    )
    args = parser.parse_args()

    if args.model not in MODEL_LIST:
        PACE_INFO(
            f"Error: Model '{args.model}' is not in the list of available models: {MODEL_LIST}"
        )
        sys.exit(1)

    PRELOAD_MODEL = args.model

    PACE_INFO(f"Starting inference server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
