# AMD PACE Inference Server

## Introduction to Inference Serving Solutions

Modern AI deployments require not just powerful models, but also efficient mechanisms to serve those models to end users, applications, or other services. **Inference serving** is the process of deploying models behind an API or endpoint so they can be accessed for real-time or batch predictions. Effective inference serving solutions handle:

- **Concurrent requests**
- **Request latency**
- **Resource utilization**
- **Model management (loading, unloading, swapping)**
- **Batching and scheduling**

---

## Using the AMD PACE Inference Server

All components (server and router) are started easily via a single launcher script.

### 1. **Installation & Setup**

Make sure your Python environment is correctly set up, dependencies are installed, and you are in the appropriate directory.

### 2. **Launching the Inference Server**

Run:

```bash
pace-server --help
```

You will see:

```bash
usage: pace-server [-h] [--server-host SERVER_HOST] [--server-port SERVER_PORT] [--server-model SERVER_MODEL] [--router-host ROUTER_HOST]
                   [--router-port ROUTER_PORT] [--max-batch-size MAX_BATCH_SIZE] [--batch-timeout BATCH_TIMEOUT]

Launcher for AMD PACE Server and Router

options:
  -h, --help            show this help message and exit
  --server-host SERVER_HOST
                        Server host (default: 0.0.0.0)
  --server-port SERVER_PORT
                        Server port (default: 8000)
  --server-model SERVER_MODEL
                        Model to load at startup
  --router-host ROUTER_HOST
                        Router host (default: 0.0.0.0)
  --router-port ROUTER_PORT
                        Router port (default: 8001)
  --max-batch-size MAX_BATCH_SIZE
                        Maximum number of items in a batch (default: 4)
  --batch-timeout BATCH_TIMEOUT
                        Number of seconds to wait before starting batch processing (default: 0.5)
```

To start with defaults:

```bash
pace-server
```

Example: Custom ports and batch settings

```bash
pace-server --server-port 9000 --router-port 9001 --max-batch-size 8 --batch-timeout 1.0
```

### 3. **Testing the Inference Server with `curl`**

Once your AMD PACE inference server and router are running, you can interact with the API endpoints using simple curl commands. This is an easy way to verify server functionality and try out your models before writing client code.

Below are examples for some typical requests.

1. Check Available Models

Fetch the list of models currently served by your router. This helps you confirm which models are active before making completion requests.

```bash
curl -X GET http://localhost:8001/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: API KEY"
```

2. Generate a Text Completion

Invoke the completions endpoint with a user prompt and desired generation parameters.
Change the "model" ID or parameters as needed to match your use case.

*Note: Currently we only support single request in curl and not batch requests.*

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: API KEY" \
  -d '{
    "model": "facebook/opt-6.7b",
    "prompt": "The cat stared at the empty hallway, waiting for",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 1.0,
    "seed": 123,
    "stop": ["\n\n"]
  }' | jq
```

3. Simple Test Example

Here's a minimal example for a quick math-completion check:

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: API KEY" \
  -d '{
    "prompt": "2+5=",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_k": 50,
    "seed": 123,
    "stop": ["\n\n"]
  }' | jq
```

4. Test Server Health (i.e., that the endpoint exists)

```bash
curl -X GET http://localhost:8001/health
```

*Note: Currently we only suport OpenAI compatible v1/completions endpoint for text generation only.*

## What Happens Under the Hood
- Server is started, loading a default model and exposing REST API endpoints for inference and health/model management.
- Router is started, performing health/model checks with the server and batching input requests before sending them for inference.
- The router manages batching (static batching: requests are batched up to --max-batch-size or until --batch-timeout seconds elapse).
- Clients send requests to the router, and receive responses after batching/inference.