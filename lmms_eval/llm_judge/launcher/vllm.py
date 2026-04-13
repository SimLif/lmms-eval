import os
import signal
import subprocess
import time

import requests
from loguru import logger as eval_logger

from .base import BaseLauncher

# Path to the serve directory relative to project root
_SERVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "serve")


class VLLMLauncher(BaseLauncher):
    """Launch a vLLM OpenAI-compatible server for LLM judge scoring.

    Uses the isolated serve/ environment (vllm==0.17.1) to avoid
    conflicts with the main eval environment.

    Usage as context manager:
        with VLLMLauncher(model="Qwen/Qwen3-VL-32B-Instruct", tp=4, gpu_ids="0,1,2,3") as server:
            # server.base_url = "http://localhost:8000"
            run_judge(server.base_url, server.model)
        # server auto-stopped on exit
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-32B-Instruct",
        port: int = 8000,
        host: str = "localhost",
        timeout: int = 600,
        tp: int = 4,
        gpu_ids: str = "0,1,2,3",
        **kwargs,
    ):
        super().__init__(port=port, host=host, timeout=timeout, model=model, **kwargs)
        self.tp = tp
        self.gpu_ids = gpu_ids
        self.process = None

    def launch(self):
        """Launch vLLM serve via the serve/start_judge_server.sh script."""
        serve_dir = os.path.abspath(_SERVE_DIR)
        script = os.path.join(serve_dir, "start_judge_server.sh")
        if not os.path.exists(script):
            raise FileNotFoundError(f"Judge server script not found: {script}")

        eval_logger.info(
            f"Starting vLLM judge server: model={self.model} "
            f"tp={self.tp} port={self.port} gpus={self.gpu_ids}"
        )

        self._log_path = os.path.join(
            os.path.dirname(serve_dir), "logs", "judge_server.log"
        )
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._log_file = open(self._log_path, "w")

        eval_logger.info(f"Server log: {self._log_path}")

        self.process = subprocess.Popen(
            ["bash", script, self.model, str(self.port), str(self.tp), self.gpu_ids],
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        self._wait_for_ready()
        eval_logger.info(f"Judge server ready at {self.base_url}")

    def _wait_for_ready(self):
        """Poll health endpoint until server is ready or timeout."""
        deadline = time.time() + self.timeout
        poll_interval = 5
        while time.time() < deadline:
            # Check if process died
            if self.process.poll() is not None:
                log_tail = ""
                if hasattr(self, "_log_path") and os.path.exists(self._log_path):
                    with open(self._log_path) as f:
                        log_tail = f.read()[-2000:]
                raise RuntimeError(
                    f"Judge server exited with code {self.process.returncode}. "
                    f"Last output:\n{log_tail}"
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=5)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Judge server failed to start within {self.timeout}s. "
            f"Check server logs."
        )

    def clean(self):
        """Kill the server process tree and close log file."""
        if self.process is None:
            return
        try:
            # Kill the entire process group (server + child workers)
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.process.wait(timeout=15)
        except (ProcessLookupError, OSError):
            pass
        except subprocess.TimeoutExpired:
            eval_logger.warning("Server did not stop gracefully, sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except (ProcessLookupError, OSError):
                pass
        self.process = None

        # Close log file
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()
            self._log_file = None
