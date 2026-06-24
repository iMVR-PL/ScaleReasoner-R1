#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/.env"

export WANDB_MODE=offline
export WANDB_DIR=$WANDB_DIR
export HYDRA_FULL_ERROR=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$PROCESSED_DIR/rl_parquet/train.parquet \
    data.val_files=$PROCESSED_DIR/rl_parquet/val.parquet \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=32 \
    data.max_prompt_length=5120 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_DIR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    custom_reward_function.path="$REPO_ROOT/verl/verl/utils/reward_score/triplet_mcq.py" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='triplet_new_mcq' \
    trainer.experiment_name='grpo_pathor1' \
    trainer.default_local_dir=$RESULTS_DIR/checkpoints/\${trainer.project_name}/\${trainer.experiment_name} \
    trainer.rollout_data_dir=$RESULTS_DIR/rollouts/\${trainer.project_name}/\${trainer.experiment_name} \
    trainer.validation_data_dir=$RESULTS_DIR/val-generation/\${trainer.project_name}/\${trainer.experiment_name} \
    trainer.resume_mode=auto \
    trainer.total_epochs=5 \
    trainer.save_freq=20 \
    trainer.test_freq=5 2>&1 | tee "$LOG_DIR/new_grpo_pathor1.log"


