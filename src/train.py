import argparse
import random
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch

from src.datasets.fastmri_dataset import DataTransform
from src.training import VarNetWCNNDataModule, VarNetWCNNLightningModule
from src.utils.checkpoint import build_checkpoint_callback
from src.utils.config import load_yaml, merge_dicts
from src.utils.data import load_file_list_from_csv
from src.utils.mask import create_mask_for_mask_type


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_config(dataset_cfg: Dict, model_cfg: Dict, train_cfg: Dict) -> Dict:
    return merge_dicts(dataset_cfg, model_cfg, train_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    config = build_config(dataset_cfg, model_cfg, train_cfg)

    set_seed(config.get("seed", 42))

    mask = create_mask_for_mask_type(
        config["mask_type"], config["center_fractions"], config["accelerations"]
    )
    train_transform = DataTransform(
        uniform_train_resolution=config["image_size"],
        mask_func=mask,
        use_seed=config.get("use_seed", True),
    )

    train_files = load_file_list_from_csv(
        config["train_csv"],
        acquisition=config.get("acquisition"),
        acquisition_index=config.get("acquisition_index", 0),
        data_root=config.get("data_root"),
    )
    val_files = load_file_list_from_csv(
        config["val_csv"],
        acquisition=config.get("acquisition"),
        acquisition_index=config.get("acquisition_index", 0),
        data_root=config.get("data_root"),
    )

    data_module = VarNetWCNNDataModule(
        train_files,
        val_files,
        config["batch_size"],
        train_transform,
        config["challenge"],
    )
    model_module = VarNetWCNNLightningModule(config)

    checkpoint_callback = build_checkpoint_callback(
        dirpath=config["checkpoint_dir"],
        monitor=config.get("checkpoint_monitor", "ssim"),
        mode=config.get("checkpoint_mode", "max"),
        save_top_k=config.get("checkpoint_save_top_k", 1),
        save_last=config.get("checkpoint_save_last", True),
        filename=config.get("checkpoint_filename"),
    )

    trainer_kwargs = {
        "accelerator": config.get("accelerator", "auto"),
        "max_epochs": config["epochs"],
        "deterministic": True,
        "callbacks": [checkpoint_callback],
        "num_sanity_val_steps": config.get("num_sanity_val_steps", 0),
    }
    if config.get("limit_train_batches") is not None:
        trainer_kwargs["limit_train_batches"] = config["limit_train_batches"]
    if config.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = config["limit_val_batches"]

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    main()
