# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import asyncio
import time
import json
from typing import Dict, Any
import httpx

from pace.utils.logging import PACE_DEBUG, PACE_INFO


async def call_api_router(
    model_name: str, router_url: str, prompt: str, gen_kwargs: Dict[str, Any]
) -> str:
    """
    Call the router asynchronously with a single prompt and generation parameters.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": gen_kwargs.get("max_new_tokens", 50),
        "temperature": gen_kwargs.get("temperature", 0.7),
        "top_p": gen_kwargs.get("top_p", 1.0),
        "top_k": gen_kwargs.get("top_k", 50),  # Added top_k parameter
        "seed": gen_kwargs.get("random_seed", None),
    }
    if "stop_strings" in gen_kwargs and gen_kwargs["stop_strings"]:
        payload["stop"] = gen_kwargs["stop_strings"]

    headers = {"Content-Type": "application/json", "Authorization": "API KEY"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{router_url}/v1/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"]
            else:
                PACE_INFO(f"Warning: Unexpected response format: {result}")
                return ""
        except Exception as e:
            PACE_INFO(f"Error calling API: {type(e).__name__}: {e}")
            PACE_INFO(f"Failed to get response from {router_url}/v1/completions")
            PACE_INFO(f"Payload: {json.dumps(payload, indent=2)}")
            return ""


async def generate_and_time(
    model_name, router_url, input_prompts, gen_kwargs, interval=0.5
):
    start = time.time()

    async def single_prompt(prompt):
        PROMPT_COLOR = "\033[94m"  # Blue for prompt issued
        RESPONSE_COLOR = "\033[92m"  # Green for response
        PROMPT_IN_RESPONSE_COLOR = (
            "\033[96m"  # Cyan (light blue) for prompt in response
        )
        RESET_COLOR = "\033[0m"

        PACE_INFO(f"\n{PROMPT_COLOR}Prompt issued: {prompt}{RESET_COLOR}")
        output = await call_api_router(model_name, router_url, prompt, gen_kwargs)
        PACE_INFO(
            f"\n{RESPONSE_COLOR}Model output for prompt [{PROMPT_IN_RESPONSE_COLOR}{prompt}{RESPONSE_COLOR}]: {output}{RESET_COLOR}"
        )
        return output

    tasks = []
    for prompt in input_prompts:
        task = asyncio.create_task(single_prompt(prompt))
        tasks.append(task)
        await asyncio.sleep(interval)

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    PACE_DEBUG("Time taken: ", elapsed)
    return results


async def run_pace_llm():
    """
    Run the PACE model via async API calls to the router.
    """
    router_url = "http://localhost:8001"

    model_name = "facebook/opt-6.7b"
    inputs_str = [
        "A lone astronaut discovers a hidden message on Mars,",
        "The world's last bookstore receives a mysterious",
        "Suddenly, all clocks in the city stop at the same",
        "2 + 5 =",
        "The American Civil War was fought",
        "The cat stared at the empty hallway, waiting for",
        "Rain fell softly on the old wooden roof,",
        "A forgotten letter was discovered",
        "The lighthouse blinked twice before",
        "She found a key taped to the back of",
        "The elevator stopped at a floor that",
        "Every mirror in the house cracked",
        "The radio played a song no one",
        "A single red balloon floated",
        "The library's clock chimed",
        "The garden gate creaked open",
        "He woke up to find his reflection missing",
        "The phone rang, but there was only",
        "A trail of footprints led into the woods, but",
        "The candle flickered even though",
        "She wrote a message in the fogged-up window, but",
        "The old photograph showed someone standing behind them who",
        "The streetlights blinked out one by one as",
        "A strange melody drifted in from.",
        "The shadows on the wall didn't match",
    ]

    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "random_seed": 123,
        "stop_strings": ["\n\n"],
    }

    PACE_INFO("\nRunning PACE model via async API")
    PACE_INFO("Model Name: ", model_name)
    PACE_INFO("Input: ", inputs_str)
    PACE_INFO("Router URL: ", router_url)

    # Check if the router is available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{router_url}/v1/models")
            if response.status_code == 200:
                available_models = response.json()
                PACE_INFO("Available models: ", json.dumps(available_models, indent=2))
            else:
                PACE_INFO(
                    f"Warning: Router returned status code {response.status_code}"
                )
    except Exception as e:
        PACE_INFO(f"Error connecting to router: {e}")
        PACE_INFO("\nMake sure to start the router with $ python router.py\n")
        return

    # Generate outputs via router API
    await generate_and_time(model_name, router_url, inputs_str, gen_kwargs, interval=3)


if __name__ == "__main__":
    asyncio.run(run_pace_llm())
