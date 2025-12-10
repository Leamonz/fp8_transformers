import os
import torch
import logging
import datetime
import math
from typing import Optional


import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hy_utils

import transformers
from transformers import (
    AutoTokenizer,
    GPT2Config,
    get_linear_schedule_with_warmup,
    set_seed,
)
from fp8_training.fp8_models.te_gpt2 import TEGPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("fp8_trainer")

try:
    import transformer_engine.pytorch as te
    from transformer.engine.common.recipe import Format, DelayedScaling
    from transformer_engine.common import recipe_handler, RecipeHandler

    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    logger.warning("Warning: transformer_engine not available. Please install it with:")
    logger.warning("pip install transformer_engine[pytorch] --no-build-isolation")

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs, TERecipeKwargs


class DataProcessor:
    """dataset preprocessing"""

    def __init__(self, tokenizer, max_seq_length=1024, streaming=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.streaming = streaming

    def tokenize_function(self, examples):
        """Tokenize function for datasets"""
        return self.tokenizer(
            examples["text"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def group_texts(self, examples, block_size=1024):
        """Group texts and create language modeling dataset"""
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def load_dataset(self, dataset_name="openwebtext", split="train", streaming=False):
        """Load and preprocess dataset"""
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name == "openwebtext":
            dataset = load_dataset(
                "openwebtext",
                split=split if not streaming else None,
                streaming=streaming,
            )
        elif dataset_name == "wikitext":
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-v1",
                split=split if not streaming else None,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=split if not streaming else None,
                streaming=streaming,
            )

        # Tokenize
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=(
                ["text"] if "text" in dataset.column_names else dataset.column_names
            ),
            desc="Running tokenizer on dataset",
        )

        # Group texts
        logger.info("Grouping texts...")
        grouped_dataset = tokenized_dataset.map(
            self.group_texts,
            batched=True,
            desc="Grouping texts",
        )

        return grouped_dataset


class GPT2FP8Trainer:
    """GPT2 FP8 Training with Transformer Engine"""

    def __init__(
        self,
        model_name: str,
        model_name_or_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        revision: str = "",
        load_pretrained_weights: bool = False,
        # dataset / training
        dataset_name: str = "openwebtext",
        learning_rate: float = 5e-5,
        betas: tuple[int] = (0.9, 0.999),
        per_device_batch_size: int = 8,
        num_epochs: int = 3,
        max_steps: int = -1,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_seq_length: int = 1024,
        seed: int = 42,
        output_dir: str = "./fp8_gpt2_output",
        use_fp8: bool = True,  # use TransformerEngine fp8 training or not
        mixed_precision: str = "fp8",
        swanlab_log: bool = False,
        val_split: float = 0.2,
        # model architecture hints
        max_position_embeddings: int = 1024,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        layer_norm_epsilon: float = 1e-5,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        last_linear_fp8: bool = True,
    ):
        self.model_name = model_name
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.revision = revision
        self.load_pretrained_weights = load_pretrained_weights
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.betas = betas
        self.per_device_batch_size = per_device_batch_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.output_dir = output_dir
        self.use_fp8 = use_fp8 and TRANSFORMER_ENGINE_AVAILABLE
        self.mixed_precision = mixed_precision if self.use_fp8 else "fp16"
        self.swanlab_log = swanlab_log
        self.val_split = val_split
        # model architecture attributes
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.last_linear_fp8 = last_linear_fp8

        set_seed(seed)

        # Initialize accelerator
        self._init_accelerator()

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"FP8 Training: {self.use_fp8}")
        logger.info(f"Mixed Precision: {self.mixed_precision}")

    def _config(self):
        return {
            "model": {
                "name": self.model_name,
                "max_position_embeddings": self.max_position_embeddings,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "n_inner": self.n_inner,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "resid_pdrop": self.resid_pdrop,
                "embd_pdrop": self.embd_pdrop,
                "attn_pdrop": self.attn_pdrop,
                "last_linear_fp8": self.last_linear_fp8,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "per_device_batch_size": self.per_device_batch_size,
                "num_epochs": self.num_epochs,
                "max_steps": self.max_steps,
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.weight_decay,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_seq_length": self.max_seq_length,
                "seed": self.seed,
                "val_split": self.val_split,
            },
            "optimization": {
                "use_fp8": self.use_fp8,
                "precision": self.mixed_precision,
                "load_pretrained_weights": self.load_pretrained_weights,
            },
            "output_dir": self.output_dir,
        }

    def _init_accelerator(self):
        """Initialize Accelerator with appropriate settings"""
        kwargs_handlers = [
            InitProcessGroupKwargs(
                backend="nccl", timeout=datetime.timedelta(seconds=600)
            )
        ]
        if self.use_fp8:
            logger.info("Setting up FP8 training with Transformer Engine...")
            kwargs_handlers.append(
                TERecipeKwargs(
                    fp8_format=Format.HYBRID,
                    amax_compute_algo="max",
                    amax_history_len=16,
                    margin=0,
                )
            )
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"
            )

        if self.swanlab_log:
            self.accelerator = Accelerator(
                mixed_precision=self.mixed_precision,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                kwargs_handlers=kwargs_handlers,
                log_with="swanlab",
                project_dir=self.output_dir,
            )
            self.accelerator.init_trackers(
                project_name="transformers_fp8_training",
                config=self._config(),
                init_kwargs={
                    "swanlab": {
                        "experiment_name": f"{self.model_name}_{self.dataset_name}_lr{self.learning_rate}_bs{self.per_device_batch_size}_{self.mixed_precision}"
                    }
                },
            )
        else:
            self.accelerator = Accelerator(
                mixed_precision=self.mixed_precision,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                kwargs_handlers=kwargs_handlers,
                log_with=None,
                project_dir=self.output_dir,
            )

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

    def _load_model_and_tokenizer(self):
        """Load GPT2 model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        # Map model names to HuggingFace model IDs
        model_mapping = {
            "gpt2-s": "openai-community/gpt2",
            "gpt2-m": "openai-community/gpt2-medium",
            "gpt2-l": "openai-community/gpt2-large",
            "gpt2-xl": "openai-community/gpt2-xl",
            "gpt2": "openai-community/gpt2",
        }

        model_id = model_mapping.get(self.model_name, self.model_name)

        # Resolve overrides for model/tokenizer ids
        if self.model_name_or_path:
            model_id = self.model_name_or_path

        tokenizer_id = self.tokenizer_name_or_path or model_id

        # Load tokenizer (we still load tokenizer vocab even if not loading model weights)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, revision=self.revision
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build GPT2Config from trainer attributes
        cfg = GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.max_position_embeddings,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_inner=self.n_inner,
            activation_function=self.activation_function,
            resid_pdrop=self.resid_pdrop,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            layer_norm_epsilon=self.layer_norm_epsilon,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Instantiate model: either load pretrained weights or initialize from config
        if self.load_pretrained_weights:
            try:
                # prefer TE-backed from_pretrained if available
                self.model = TEGPT2LMHeadModel.from_pretrained(
                    model_id, config=cfg, last_linear_fp8=self.last_linear_fp8
                )
            except Exception:
                logger.warning(
                    "TE model from_pretrained failed; instantiating from config"
                )
                self.model = TEGPT2LMHeadModel(
                    cfg,
                    fp8_recipe=self.fp8_recipe,
                    last_linear_fp8=self.last_linear_fp8,
                )
        else:
            # Do not load pretrained weights — initialize model from config only
            self.model = TEGPT2LMHeadModel(
                cfg, fp8_recipe=self.fp8_recipe, last_linear_fp8=self.last_linear_fp8
            )

        logger.info(
            f"Model loaded successfully. Total parameters: {self.model.num_parameters()/1000**2:.2f}M"
        )

    def prepare_dataloaders(self):
        """Prepare train and eval dataloaders"""
        logger.info("Preparing dataloaders...")

        data_processor = DataProcessor(
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            streaming=False,
        )
        # Attempt to load tokenized/grouped train and validation splits via DataProcessor
        train_dataset = None
        eval_dataset = None

        try:
            train_dataset = data_processor.load_dataset(
                dataset_name=self.dataset_name, split="train", streaming=False
            )
        except Exception:
            logger.warning(
                "Failed to load train split via DataProcessor; will try raw load and split."
            )

        try:
            eval_dataset = data_processor.load_dataset(
                dataset_name=self.dataset_name, split="validation", streaming=False
            )
        except Exception:
            eval_dataset = None

        # If eval not present, or train couldn't be loaded as grouped, load raw and split
        if train_dataset is None or eval_dataset is None:
            logger.info("Loading raw dataset to perform train/validation split...")
            raw = load_dataset(self.dataset_name, split="train")
            # Use provided validation split fraction (val_split is proportion for validation)
            split = raw.train_test_split(test_size=self.val_split)
            raw_train = split["train"]
            raw_eval = split["test"]

            # Tokenize and group both
            tokenized_train = raw_train.map(
                data_processor.tokenize_function,
                batched=True,
                remove_columns=raw_train.column_names,
                desc="Tokenizing train split",
            )
            train_dataset = tokenized_train.map(
                data_processor.group_texts,
                batched=True,
                desc="Grouping train split",
            )

            tokenized_eval = raw_eval.map(
                data_processor.tokenize_function,
                batched=True,
                remove_columns=raw_eval.column_names,
                desc="Tokenizing eval split",
            )
            eval_dataset = tokenized_eval.map(
                data_processor.group_texts,
                batched=True,
                desc="Grouping eval split",
            )

        # Create dataloaders for both
        dataset_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset_collator,
        )

        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.per_device_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=dataset_collator,
            )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Train dataloader batches: {len(train_dataloader)}")
        if eval_dataloader is not None:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
            logger.info(f"Eval dataloader batches: {len(eval_dataloader)}")

        return train_dataloader, eval_dataloader

    def setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer and learning rate scheduler"""
        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, betas=self.betas
        )

        # Calculate total steps
        if self.max_steps > 0:
            total_steps = self.max_steps
        else:
            total_steps = (
                len(train_dataloader)
                * self.num_epochs
                // self.gradient_accumulation_steps
            )

        # Learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Total training steps: {total_steps}")

        return optimizer, lr_scheduler

    def train(self):
        """Main training loop"""
        train_dataloader, eval_dataloader = self.prepare_dataloaders()
        optimizer, lr_scheduler = self.setup_optimizer_and_scheduler(train_dataloader)

        # Prepare everything with accelerator
        self.model, optimizer, train_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                self.model, optimizer, train_dataloader, lr_scheduler
            )
        )

        # Training loop
        progress_bar_update = 0
        global_step = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.autocast():
                    with self.accelerator.accumulate(
                        self.model
                    ):  # for gradient accumulation
                        # Forward pass
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch.get("labels", batch["input_ids"]),
                        )
                        loss = outputs.loss

                        # Backward pass
                        self.accelerator.backward(loss)

                        # Gradient accumulation step
                        grad_norm = 0.0
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), 1.0
                            )

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                total_loss += loss.detach().item()
                batch_count += 1
                global_step += 1
                progress_bar_update += 1

                if self.accelerator.is_main_process:
                    avg_loss = total_loss / batch_count
                    logger.info(
                        f"Epoch {epoch + 1}, Step {step + 1}, Loss: {avg_loss:.4f}, "
                        f"Gradient Norm: {grad_norm:.2f}, "
                        f"Learning Rate: {lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    if self.swanlab_log:
                        self.accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )

                if self.max_steps > 0 and global_step >= self.max_steps:
                    break

            # Run evaluation at end of epoch (if eval dataset available)
            if eval_dataloader is not None:
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                self.model.eval()
                total_eval_loss = 0.0
                eval_steps = 0
                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=eval_batch["input_ids"],
                            attention_mask=eval_batch.get("attention_mask"),
                            labels=eval_batch.get("labels", eval_batch["input_ids"]),
                        )
                        loss = outputs.loss
                        # try to gather across processes
                        try:
                            gathered = self.accelerator.gather(loss.detach())
                            if isinstance(gathered, torch.Tensor):
                                loss_val = gathered.mean().item()
                            else:
                                # fallback
                                loss_val = float(gathered)
                        except Exception:
                            loss_val = loss.detach().item()

                    total_eval_loss += loss_val
                    eval_steps += 1

                if eval_steps > 0:
                    avg_eval_loss = total_eval_loss / eval_steps
                    try:
                        perplexity = (
                            math.exp(avg_eval_loss)
                            if avg_eval_loss < 100
                            else float("inf")
                        )
                    except Exception:
                        perplexity = float("inf")

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"Epoch {epoch + 1} eval_loss: {avg_eval_loss:.4f}, perplexity: {perplexity:.2f}"
                        )
                        if self.swanlab_log:
                            self.accelerator.log(
                                {
                                    "eval/loss": avg_eval_loss,
                                    "eval/perplexity": perplexity,
                                },
                                step=global_step,
                            )

            # Save model at end of epoch
            if self.accelerator.is_main_process:
                epoch_output_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
                os.makedirs(epoch_output_dir, exist_ok=True)

                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(epoch_output_dir)
                self.tokenizer.save_pretrained(epoch_output_dir)
                logger.info(f"Checkpoint saved to {epoch_output_dir}")

            if self.max_steps > 0 and global_step >= self.max_steps:
                break

        self.accelerator.end_training()
        logger.info("Training completed!")
