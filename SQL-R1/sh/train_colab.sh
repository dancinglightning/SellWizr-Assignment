#!/bin/bash
# SQL-R1 Training Script for 24GB GPU (Google Colab / RTX 3090/4090)
# Optimized for ~3B parameter LLM with aggressive memory efficiency
# Assignment: Text-to-SQL RL Integration

set -e

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Critical: Reduce Ray memory overhead and prevent aggressive OOM killing
export RAY_memory_usage_threshold=0.98
export RAY_memory_monitor_refresh_ms=0
export PYTHONUNBUFFERED=1

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Clear any existing GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true

# ============================================================================
# CONFIGURATION
# ============================================================================

RUN_ID=3B-24GB-final
GPU_ENV=1GPU
MODEL_ENV=Qwen2.5-Coder-3B-Instruct
PROJECT_NAME=SQL-R1-Assignment
DATA_DIR_PATH=data
LOG_PATH=logs/$PROJECT_NAME

MODEL_PATH=models/$MODEL_ENV
EXPERIMENT_NAME=$GPU_ENV-$MODEL_ENV-$RUN_ID

mkdir -p $LOG_PATH/$MODEL_ENV

# ============================================================================
# DATA & MODEL PREPARATION
# ============================================================================

# Ensure data directory exists and has files
if [ ! -d "$DATA_DIR_PATH" ] || [ ! -f "$DATA_DIR_PATH/train.parquet" ]; then
    echo "Data directory missing or incomplete. Setting up from example_data..."
    mkdir -p "$DATA_DIR_PATH"
    if [ -d "example_data" ]; then
        cp example_data/*.parquet "$DATA_DIR_PATH/" 2>/dev/null || true
        echo "✓ Data prepared from example_data"
    else
        echo "Error: Neither 'data/' nor 'example_data/' found. Please run Step 5 in the Colab notebook."
        exit 1
    fi
fi

# Ensure model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please run Step 4 in the Colab notebook to download the model."
    exit 1
fi

echo "================================"
echo "SQL-R1 Training Configuration"
echo "================================"
echo "Model: $MODEL_ENV"
echo "GPU Memory: 24GB target"
echo "Method: GRPO (Group Relative Policy Optimization)"
echo "LoRA: Enabled (Rank 16)"
echo "================================"

# Display GPU info
nvidia-smi

# ============================================================================
# TRAINING
# ============================================================================

# Install dependencies for reward function
pip install sqlparse timeout-decorator

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    `# -------- Data Configuration --------` \
    data.train_files=$DATA_DIR_PATH/train.parquet \
    data.val_files=$DATA_DIR_PATH/test.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    \
    `# -------- Model Configuration --------` \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.lora_rank=16 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.lora_dropout=0.05 \
    +actor_rollout_ref.model.target_modules=all-linear \
    \
    `# -------- Actor Configuration (CRITICAL: Enable CPU offloading) --------` \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=fp16 \
    \
    `# -------- Rollout Configuration (vLLM - Very Conservative) --------` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.05 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.dtype=float16 \
    ++actor_rollout_ref.rollout.max_model_len=1536 \
    ++actor_rollout_ref.rollout.max_num_seqs=1 \
    ++actor_rollout_ref.rollout.enforce_eager=True \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
    ++actor_rollout_ref.rollout.swap_space=4 \
    \
    `# -------- Reference Model Configuration (Maximum Offloading) --------` \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    ++actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.grad_offload=True \
    +actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp16 \
    \
    `# -------- Algorithm Configuration --------` \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    `# -------- Reward Configuration --------` \
    +reward.db_path=data/database.db \
    \
    `# -------- Trainer Configuration --------` \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$LOG_PATH/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=3 \
    $@ 2>&1 | tee $LOG_PATH/$MODEL_ENV/training_log.txt

echo "================================"
echo "Training completed!"
echo "Logs saved to: $LOG_PATH/$MODEL_ENV/training_log.txt"
echo "Checkpoints saved to: $LOG_PATH/$EXPERIMENT_NAME"
echo "================================"Assignment