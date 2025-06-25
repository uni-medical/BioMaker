import os
import subprocess
import multiprocessing

class ResourceManager:
    """Simplified Resource Manager for checking system device info."""

    @staticmethod
    def get_cpu_info():
        cpu_count = multiprocessing.cpu_count()
        return {"cpu_count": cpu_count}

    @staticmethod
    def get_gpu_count():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return 1  # Apple Silicon GPU
            else:
                return 0
        except ImportError:
            return 0

    @staticmethod
    def get_gpu_memory():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            return [int(x) for x in output.strip().split("\n")]
        except Exception:
            return []

    @staticmethod
    def summary():
        cpu_info = ResourceManager.get_cpu_info()
        gpu_count = ResourceManager.get_gpu_count()
        gpu_memory = ResourceManager.get_gpu_memory()

        print("=== Device Info Summary ===")
        print(f"CPU Count       : {cpu_info['cpu_count']}")
        print(f"GPU Count       : {gpu_count}")
        print(f"GPU Memory (MB) : {gpu_memory if gpu_memory else 'Unavailable or No NVIDIA GPU'}")
        print("============================")

    @staticmethod
    def should_use_gpu(min_memory_gb=4):
        """Return True if GPU is available and has enough memory"""
        gpu_memories = ResourceManager.get_gpu_memory()
        if not gpu_memories:
            return False
        return any(mem >= min_memory_gb * 1024 for mem in gpu_memories)

# 示例用法
if __name__ == "__main__":
    ResourceManager.summary()
    if ResourceManager.should_use_gpu():
        print("✅ Recommend: Use GPU")
    else:
        print("❌ Recommend: Use CPU")
