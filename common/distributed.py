import builtins
import os
import torch

def _is_main_process() -> bool:
    """Determine if the current process is the main process."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass

    return True

_builtins_print = builtins.print

def print(*args, force: bool = False, **kwargs):  # type: ignore[override]
    """Customized print that only logs from the main process unless forced."""
    if force or _is_main_process():
        _builtins_print(*args, **kwargs)