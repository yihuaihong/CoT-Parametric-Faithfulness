# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############

    handlers = [logging.StreamHandler(sys.stdout)]
    # 仅在主进程（local_rank 为 -1 或 0）添加文件日志，避免多进程写入冲突
    # mode="w" 表示每次启动时覆盖文件，从而只保留最新的一次训练日志
    if training_args.local_rank in [-1, 0]:
        handlers.append(logging.FileHandler("grpo_training.log", mode="w", encoding="utf-8"))


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,  # <--- 关键修改：这里之前你写的是 [logging.StreamHandler(sys.stdout)]，所以没生效
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    # Optional dev sub-sampling (set env CIA_DEV_FRAC=<denom> to enable).
    # Loud warning when active so it can't silently ruin training.
    _dev_frac = os.environ.get("CIA_DEV_FRAC")
    if _dev_frac:
        denom = int(_dev_frac)
        print(f"WARN: CIA_DEV_FRAC={denom} — using 1/{denom} of every split")
        for split in dataset.keys():
            n = len(dataset[split])
            dataset[split] = dataset[split].select(range(max(1, n // denom)))

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Wrap reward funcs so each call's (prompt, completion, reward) is
    # cached and the WandbCompletionTableCallback can flush to wandb.
    from open_r1.utils.completion_logger import (
        install_completion_capture,
        WandbCompletionTableCallback,
    )
    reward_funcs = install_completion_capture(reward_funcs)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=(
            get_callbacks(training_args, model_args)
            + [WandbCompletionTableCallback(
                every_n_steps=max(1, int(training_args.logging_steps)),
                max_rows_per_flush=8)]
        ),
        processing_class=tokenizer,
    )

    # TRL 0.14's GRPOTrainer builds a GenerationConfig WITHOUT
    # eos_token_id set. For Llama-3 / Gemma / Qwen instruct models, the
    # chat template emits <|eot_id|>-like tokens that the model uses to
    # signal end-of-turn, but without eos_token_id the rollout never
    # stops and always pads to max_completion_length. Explicitly pull
    # EOS candidates from the tokenizer + model config.
    eos_ids = []
    for src in (tokenizer, model.config):
        tid = getattr(src, "eos_token_id", None)
        if tid is None:
            continue
        if isinstance(tid, (list, tuple)):
            eos_ids.extend(int(x) for x in tid)
        else:
            eos_ids.append(int(tid))
    # Llama-3 specific: also include <|eot_id|> if present in the vocab.
    for tok in ("<|eot_id|>", "<|end_of_text|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                eos_ids.append(int(tid))
        except Exception:
            pass
    eos_ids = sorted(set(eos_ids))
    if eos_ids:
        logger.info(f"Setting GRPO generation eos_token_id = {eos_ids}")
        trainer.generation_config.eos_token_id = eos_ids

    # ------------------------------------------------------------------
    # Monkey-patch TRL's unwrap_model_for_generation to also toggle the
    # model into eval() during rollout. Required because HF's
    # gradient_checkpointing in train mode + use_cache=False makes
    # autoregressive generation produce incoherent output (diagnostic
    # confirmed via logs/gen_matrix_6347094.out on 2026-04-15: only the
    # train+ckpt+no_cache combination rambles; any one disabled fixes).
    # ------------------------------------------------------------------
    import contextlib
    import trl.trainer.grpo_trainer as _gt
    _original_unwrap = _gt.unwrap_model_for_generation

    @contextlib.contextmanager
    def _unwrap_eval(model, accelerator, is_peft_model=False,
                     gather_deepspeed3_params=True):
        with _original_unwrap(model, accelerator,
                              is_peft_model=is_peft_model,
                              gather_deepspeed3_params=gather_deepspeed3_params) as m:
            was_training = m.training
            m.eval()
            try:
                yield m
            finally:
                if was_training:
                    m.train()

    _gt.unwrap_model_for_generation = _unwrap_eval
    logger.info("Patched unwrap_model_for_generation to toggle eval() "
                "around rollout (fixes gradient_checkpointing gibberish)")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
