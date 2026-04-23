import os
import subprocess
from abc import ABC, abstractmethod

from loguru import logger as eval_logger


class BaseLauncher(ABC):
    def __init__(self, port: int = 8000, host: str = "localhost", timeout: int = 1200, model: str = "Qwen/Qwen3-8B", **kwargs):
        super().__init__()
        self.port = port
        self.host = host
        self.timeout = timeout
        self.model = model
        self.base_url = f"http://{self.host}:{self.port}"

    @abstractmethod
    def launch(self, *args, **kwargs):
        """Launch the LLM judge server."""
        pass

    @abstractmethod
    def clean(self):
        """Clean up resources or processes after the launch."""
        pass

    def __enter__(self):
        # NOTE: no pre-flight gpu_clean here — it would lsof /dev/nvidia*
        # and kill the current Python process (which has torch imported and
        # opens /dev/nvidia for device detection). GPU pre-flight must run
        # OUTSIDE the judge Python process — see eval_ckpt.sh / gen_launch.py
        # Phase 2b hooks.
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        eval_logger.info(f"Shutting down server (model={self.model}, port={self.port})")
        try:
            self.clean()
        except Exception as e:
            eval_logger.warning(f"Error during server cleanup: {e}")
        return False
