import argparse

import pytorch_lightning as pl

from src.datasets.fastmri_dataset import DataTransform
from src.training import VarNetWCNNDataModule, VarNetWCNNLightningModule
from src.utils.config import load_yaml, merge_dicts
from src.utils.data import load_file_list_from_csv
from src.utils.mask import create_mask_for_mask_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    config = merge_dicts(dataset_cfg, model_cfg, train_cfg)

    mask = create_mask_for_mask_type(
        config["mask_type"], config["center_fractions"], config["accelerations"]
    )
    val_transform = DataTransform(
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
        val_transform,
        config["challenge"],
    )

    model_module = VarNetWCNNLightningModule.load_from_checkpoint(
        args.checkpoint, config=config
    )

    trainer = pl.Trainer(
        accelerator=config.get("accelerator", "auto"),
        deterministic=True,
    )
    trainer.validate(model_module, data_module)


if __name__ == "__main__":
    main()
