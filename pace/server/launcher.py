# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import argparse
import subprocess
import sys
import os

from pace.utils.logging import PACE_INFO


def main():
    parser = argparse.ArgumentParser(description="Launcher for PACE Server and Router")

    # Server args
    parser.add_argument(
        "--server-host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--server-model",
        type=str,
        default="facebook/opt-6.7b",
        help="Model to load at startup",
    )

    # Router args
    parser.add_argument(
        "--router-host",
        type=str,
        default="0.0.0.0",
        help="Router host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--router-port", type=int, default=8001, help="Router port (default: 8001)"
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

    args = parser.parse_args()

    # Start the server subprocess
    server_cmd = [
        sys.executable,
        "-m",
        "pace.server.server",
        "--host",
        str(args.server_host),
        "--port",
        str(args.server_port),
        "--model",
        str(args.server_model),
    ]

    # Start the router subprocess
    router_cmd = [
        sys.executable,
        "-m",
        "pace.server.router",
        "--host",
        str(args.router_host),
        "--port",
        str(args.router_port),
        "--server-url",
        f"http://{args.server_host}:{args.server_port}",
        "--max-batch-size",
        str(args.max_batch_size),
        "--batch-timeout",
        str(args.batch_timeout),
    ]

    PACE_INFO(f"Launching SERVER: {' '.join(server_cmd)}")
    PACE_INFO(f"Launching ROUTER: {' '.join(router_cmd)}")

    server_proc = subprocess.Popen(server_cmd, env=os.environ)
    router_proc = subprocess.Popen(router_cmd, env=os.environ)

    try:
        server_proc.wait()
        router_proc.wait()
    except KeyboardInterrupt:
        PACE_INFO("Stopping server and router...")
        server_proc.terminate()
        router_proc.terminate()


if __name__ == "__main__":
    main()
