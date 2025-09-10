# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import asyncio
import time
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Union
import httpx
import logging
import argparse

from pace.utils.logging import PACE_INFO

global SERVER_URL, MAX_BATCH_SIZE, BATCH_TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
request_queue = asyncio.Queue()
server_ready = False
available_models = set()


class Message(BaseModel):
    role: Optional[str] = None
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "facebook/opt-6.7b"
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50  # Default top_k for sampling
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = "facebook/opt-6.7b"
    prompt: Union[str, List[str]]  # Changed from messages to prompt
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50  # Default top_k for sampling
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class BatchedRequest:
    def __init__(
        self, req: Union[CompletionRequest, ChatCompletionRequest]
    ):  # Updated type hint
        self.req = req
        self.future = asyncio.get_event_loop().create_future()


# This formatting is specific to v1/chat/completions endpoint
def format_messages_to_prompt_v1_chat_completions(messages: List[Message]) -> str:
    formatted_prompt = ""
    for msg in messages:
        role_prefix = ""
        if msg.role.lower() == "system":
            role_prefix = "System: "
        elif msg.role.lower() == "user":
            role_prefix = "User: "
        elif msg.role.lower() == "assistant":
            role_prefix = "Assistant: "
        formatted_prompt += f"{role_prefix}{msg.content}\n"
    formatted_prompt += "Assistant: "
    PACE_INFO(f"Formatted prompt: {formatted_prompt}")
    return formatted_prompt


async def queue_put(item):
    await request_queue.put(item)


async def queue_get():
    item = await request_queue.get()
    return item


async def health_check_loop():
    global server_ready, available_models
    while not server_ready:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{SERVER_URL}/health", timeout=2)
                if resp.status_code == 200:
                    # Get model list at startup
                    model_resp = await client.get(f"{SERVER_URL}/get_models", timeout=2)
                    if model_resp.status_code == 200:
                        model_data = model_resp.json()
                        available_models = set([m["id"] for m in model_data["data"]])
                        logger.info(f"Available models cached: {available_models}")
                        server_ready = True
                        logger.info("Server is available. Router is ready.")
                        break
        except Exception:
            logger.info("Waiting for server to be available...")
        await asyncio.sleep(5)


async def batch_worker():
    while True:
        # Wait for at least one item in the queue
        req = await queue_get()
        batch = [req]
        # Start the batch window
        start = time.monotonic()
        while len(batch) < MAX_BATCH_SIZE:
            remaining = BATCH_TIMEOUT - (time.monotonic() - start)
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(queue_get(), timeout=remaining)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        # Print queue size before draining
        logger.info(
            f"Batch size: {len(batch)} sent for processing. Queue size after batch removed: {request_queue.qsize()}"
        )

        prompts = []
        gen_kwargs = None
        model_name = batch[0].req.model  # All requests in batch must use the same model
        for br in batch:
            if br.req.model != model_name:
                logger.error("Mixed models in batch! This is not supported.")
                br.future.set_result("Error: Mixed models in batch.")
                continue

            # Handle different request types
            if isinstance(br.req, CompletionRequest):
                # For v1/completions, use the prompt directly
                if isinstance(br.req.prompt, list):
                    prompt = "".join(br.req.prompt)  # Join if it's a list
                else:
                    prompt = br.req.prompt
            else:
                # For v1/chat/completions, format messages
                prompt = format_messages_to_prompt_v1_chat_completions(br.req.messages)

            prompts.append(prompt)
            if gen_kwargs is None:
                stop_strings = br.req.stop if br.req.stop else []
                if isinstance(stop_strings, str):
                    stop_strings = [stop_strings]
                gen_kwargs = {
                    "max_new_tokens": br.req.max_tokens,
                    "do_sample": True if br.req.temperature > 0 else False,
                    "temperature": br.req.temperature,
                    "top_p": br.req.top_p,
                    "top_k": br.req.top_k,
                    "random_seed": (
                        br.req.seed if br.req.seed is not None else int(time.time())
                    ),
                    "stop_strings": stop_strings,
                }

        # Send batch to server
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{SERVER_URL}/infer",
                    json={
                        "model": model_name,
                        "prompts": prompts,
                        "gen_kwargs": gen_kwargs,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                outputs = resp.json()["outputs"]
            except Exception as e:
                outputs = [""] * len(batch)
                logger.error(f"Inference server error: {e}")

        # Set result for each future
        for br, out in zip(batch, outputs):
            br.future.set_result(out)


@app.on_event("startup")
async def startup_event():
    # Wait 5 seconds before starting health check
    await asyncio.sleep(5)
    asyncio.create_task(health_check_loop())
    asyncio.create_task(batch_worker())


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if not server_ready:
        return {"error": "Inference server not ready. Please try again later."}
    if request.model not in available_models:
        return {"error": f"Requested model '{request.model}' is not available."}
    batched_req = BatchedRequest(request)
    await queue_put(batched_req)
    output = await batched_req.future
    # Format OpenAI-style response
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",  # Changed from "completion"
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": output,  # Changed from "message" to "text"
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def list_models():
    # Proxy the server's model list and return it
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/get_models", timeout=2)
        return resp.json()


@app.get("/health")
def health_check():
    return {"status": "ok"}


def parse_args():
    parser = argparse.ArgumentParser(description="PACE Router arguments")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="PACE inference server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum number of items in a batch (default: 4)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.5,
        help="Number of seconds to wait before starting batch processing (default: 0.5)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Router host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Router port (default: 8001)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SERVER_URL = args.server_url
    MAX_BATCH_SIZE = args.max_batch_size
    BATCH_TIMEOUT = args.batch_timeout
    PACE_INFO(f"Starting router on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
