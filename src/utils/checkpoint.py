from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(
    dirpath: str,
    monitor: str = "ssim",
    mode: str = "max",
    save_top_k: int = 1,
    save_last: bool = True,
    filename: Optional[str] = None,
) -> ModelCheckpoint:
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = "run {epoch}-{ssim:.2f}"
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
    )
