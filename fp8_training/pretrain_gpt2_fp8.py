"""
GPT2 FP8 Low-Precision Training with Transformer Engine Backend
使用 transformer_engine backend 进行 FP8 低精度训练 GPT2 模型
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hy_utils

from train_utils import logger, GPT2FP8Trainer


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry point. Configuration is provided by `configs/config.yaml`."""
    logger.info("Hydra configuration:\n" + OmegaConf.to_yaml(cfg))

    # Resolve output directory to absolute path
    output_dir = hy_utils.to_absolute_path(cfg.output_dir)

    # Create trainer from config
    trainer = GPT2FP8Trainer(
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        learning_rate=cfg.learning_rate,
        per_device_batch_size=cfg.per_device_batch_size,
        num_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_seq_length=cfg.max_seq_length,
        seed=cfg.seed,
        output_dir=output_dir,
        use_fp8=cfg.use_fp8,
        val_split=cfg.val_split,
        swanlab_log=cfg.swanlab_log,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
