import os
import sys
import logging
import subprocess

logger = logging.getLogger(__name__)

def get_gpu_status():
    """Queries NVIDIA GPUs and returns a list of dictionaries with status info."""
    gpus = []
    try:
        # Run nvidia-smi query to get details in CSV format
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        if not output:
            return gpus

        for line in output.split("\n"):
            idx, name, mem_used, mem_total, util = [x.strip() for x in line.split(",")]
            idx = int(idx)
            mem_used = int(mem_used)
            mem_total = int(mem_total)
            util = int(util)

            # Determine status threshold
            status = "Free"
            if util > 30 or (mem_used / mem_total) > 0.3:
                status = "Busy"
            if util > 70 or (mem_used / mem_total) > 0.7:
                status = "Critical"

            gpus.append({
                "index": idx,
                "name": name,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "util": util,
                "status": status
            })
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise RuntimeError(
            "CRITICAL: nvidia-smi failed — CPU fallback must be an explicit user decision "
            "(pass --use_mock_data or set CUDA_VISIBLE_DEVICES='' before invoking)."
        ) from e
    return gpus

def select_gpus(interactive=None) -> str:
    """
    Displays available GPUs, prompts the user to select indices,
    and returns a comma-separated string of selected indices.
    """
    gpus = get_gpu_status()
    if not gpus:
        logger.info("No physical GPUs detected. Defaulting to CPU execution / system default device.")
        return ""

    # If interactive is not explicitly set, determine it via standard tty check
    if interactive is None:
        interactive = sys.stdin.isatty()

    if not interactive:
        # Auto-select the first completely free GPU, or GPU 0 if all are busy
        free_gpus = [g["index"] for g in gpus if g["status"] == "Free"]
        selected = str(free_gpus[0]) if free_gpus else "0"
        logger.info(f"Non-interactive session. Automatically selected GPU: {selected}")
        return selected

    print("\n" + "="*60)
    print("                 AVAILABLE SERVER GPUs")
    print("="*60)
    print(f"{'ID':<3} | {'Name':<20} | {'VRAM Usage':<15} | {'Compute':<8} | {'Status':<10}")
    print("-"*60)

    for g in gpus:
        # Format strings with ANSI colors for high visibility
        status_color = "\033[92m"  # Green
        if g["status"] == "Busy":
            status_color = "\033[93m"  # Yellow
        elif g["status"] == "Critical":
            status_color = "\033[91m"  # Red
        
        status_str = f"{status_color}{g['status']:<10}\033[0m"
        vram_str = f"{g['mem_used']}MB / {g['mem_total']}MB"

        print(f"{g['index']:<3} | {g['name'][:20]:<20} | {vram_str:<15} | {g['util']}%     | {status_str}")
    print("="*60)

    while True:
        try:
            val = input("\nSelect GPU(s) to run on (e.g., '0' or '0,1' or '2') [Default=0]: ").strip()
            if not val:
                return "0"
            
            # Validate input indices
            indices = [int(i.strip()) for i in val.split(",")]
            valid_indices = {g["index"] for g in gpus}
            if all(idx in valid_indices for idx in indices):
                selected_str = ",".join(str(i) for i in indices)
                logger.info(f"User selected GPU(s): {selected_str}")
                return selected_str
        except ValueError:
            pass
        print("\033[91mInvalid input. Please enter valid integer GPU indices separated by commas.\033[0m")
