"""
GPT2 FP8 Low-Precision Training with Transformer Engine Backend
使用 transformer_engine backend 进行 FP8 低精度训练 GPT2 模型
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hy_utils

from train_utils import logger, GPT2FP8Trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Hydra entry point. Configuration is provided by `configs/config.yaml`."""
    logger.info("Hydra configuration:\n" + OmegaConf.to_yaml(cfg))

    # Resolve output directory to absolute path
    output_dir = hy_utils.to_absolute_path(cfg.training.output_dir)

    # Create trainer from config
    trainer = GPT2FP8Trainer(
        model_name=cfg.model.name,
        model_name_or_path=cfg.model.model_name_or_path,
        tokenizer_name_or_path=cfg.model.tokenizer_name_or_path,
        revision=cfg.model.revision,
        load_pretrained_weights=cfg.training.load_pretrained_weights,
        dataset_name=cfg.dataset.name,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        per_device_batch_size=cfg.training.per_device_batch_size,
        num_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.max_steps,
        warmup_steps=cfg.training.warmup_steps,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_seq_length=cfg.dataset.max_seq_length,
        seed=cfg.training.seed,
        output_dir=output_dir,
        use_fp8=cfg.optimization.use_fp8,
        val_split=cfg.dataset.val_split,
        swanlab_log=cfg.optimization.swanlab_log,
        # model architecture hints
        max_position_embeddings=cfg.model.max_position_embeddings,
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        n_inner=cfg.model.n_inner,
        layer_norm_epsilon=cfg.model.layer_norm_epsilon,
        resid_pdrop=cfg.model.resid_pdrop,
        embd_pdrop=cfg.model.embd_pdrop,
        attn_pdrop=cfg.model.attn_pdrop,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
