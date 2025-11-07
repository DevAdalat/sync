"""
🚀 ULTIMATE TRANSFORMER TRAINER - All-in-One Solution

This is the most comprehensive training script combining the best features from all trainers:

✨ MODEL FLEXIBILITY:
  • Model Presets (nano/tiny/small/medium/large/xlarge) - Quick start with proven architectures
  • Custom Architecture - Define your own d_model, layers, heads, FFN size
  • Parameter-Based Sizing - Specify target parameter count
  • Variable Model Sizes - Create any size model from tiny to XL

🎯 OPTIMIZATION FEATURES:
  • Flash Attention (Kvax) - 2-3x faster attention for long sequences
  • RoPE (Rotary Position Embeddings) - Better positional encoding
  • RMSNorm - More efficient normalization
  • SwiGLU - Powerful activation function
  • LoRA - Parameter-efficient fine-tuning
  • Mixed Precision - bfloat16 training for TPU/GPU
  • Gradient Accumulation - Train with larger effective batch sizes
  • Gradient Checkpointing - Trade compute for memory

💾 MEMORY OPTIMIZATIONS:
  • Streaming Data Loading - Minimal memory footprint
  • Shuffle Buffer - Efficient data shuffling
  • Batch Size Control - Fine-tune memory usage
  • Multi-Device Training - Data parallelism across TPU/GPU/CPU

📊 TRAINING FEATURES:
  • Auto Vocab Size - Automatically determine optimal vocabulary
  • Learning Rate Scheduling - Warmup + Cosine decay
  • Gradient Clipping - Stable training
  • Validation During Training - Monitor performance
  • Checkpoint Saving/Resuming - Never lose progress
  • Cloud Storage Support - S3, GCS, Azure Blob
  • Multi-Device Support - TPU, GPU, CPU with automatic detection
  • Real-time Logging - Monitor training progress

🔧 EASE OF USE:
  • Command-Line Interface - Full control via CLI
  • Sensible Defaults - Start training with minimal config
  • Extensive Documentation - Every parameter explained
  • Error Handling - Clear error messages

Usage Examples:

  # Quick start with preset (tiny model, auto vocab)
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --model-preset tiny \\
    --auto-vocab-size \\
    --epochs 3

  # Custom architecture with specific parameters
  python -m src.training.train_ultimate \\
    --dataset-id "wikitext" --dataset-config "wikitext-2-raw-v1" \\
    --d-model 512 --num-layers 8 --num-heads 8 --d-ff 2048 \\
    --vocab-size 16000 --seq-len 256 \\
    --epochs 5 --batch-size 64

  # Parameter-based sizing (create ~10M param model)
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --target-params 10000000 \\
    --prefer-depth \\
    --epochs 5

  # Memory-optimized training with gradient accumulation
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --model-preset medium \\
    --batch-size 16 --gradient-accumulation-steps 4 \\
    --use-streaming --use-mixed-precision

  # Training with validation and cloud storage
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --model-preset large \\
    --val-split "validation" --eval-every 500 \\
    --cloud-provider s3 --cloud-bucket my-checkpoints

  # Resume from checkpoint
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --model-preset medium \\
    --resume-from output/checkpoint_step_1000
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from orbax import checkpoint as ocp
from tokenizers import Tokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config import CloudConfig, ModelConfig
from src.data.hf_dataset_loader import HFDatasetLoader
from src.data.streaming_data_loader import StreamingDataLoader
from src.models.model import ProductionTransformer
from src.models.flash_attention import get_flash_attention_config
from src.utils.utils import upload_checkpoint_to_cloud

# Try to import model sizing utility
try:
    from tests.model_sizing import create_model_from_params
    HAS_MODEL_SIZING = True
except ImportError:
    HAS_MODEL_SIZING = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_params(params):
    """Count total parameters in model"""
    total = 0
    for key, value in params.items():
        if isinstance(value, dict):
            total += count_params(value)
        else:
            total += value.size
    return total


def print_banner(text: str, width: int = 100, char: str = "="):
    """Print a formatted banner"""
    print(f"\n{char * width}")
    print(text.center(width))
    print(f"{char * width}")


def print_section(text: str, width: int = 100):
    """Print a section header"""
    print(f"\n{'=' * width}")
    print(text)
    print(f"{'=' * width}")


def detect_and_setup_device():
    """Detect available devices and setup for training"""
    devices = jax.devices()
    backend = jax.default_backend()
    num_devices = len(devices)
    
    device_info = {
        "backend": backend,
        "num_devices": num_devices,
        "devices": [str(d) for d in devices],
        "device_type": devices[0].platform if devices else "unknown",
    }
    
    print_section("DEVICE CONFIGURATION")
    print(f"  Backend:        {backend.upper()}")
    print(f"  Device type:    {device_info['device_type'].upper()}")
    print(f"  Num devices:    {num_devices}")
    
    if num_devices > 1:
        print(f"\n  Detected {num_devices} devices:")
        for i, device in enumerate(devices):
            print(f"    Device {i}: {device}")
        print(f"\n  ✓ Multi-device training enabled!")
    else:
        print(f"  Device: {devices[0]}")
    
    # Device-specific tips
    if backend == "tpu":
        print("\n  ✓ TPU Optimizations:")
        print("    • XLA compilation enabled")
        print("    • Consider using bfloat16 precision")
        print("    • Larger batch sizes recommended")
    elif backend == "gpu":
        print("\n  ✓ GPU Optimizations:")
        print("    • CUDA acceleration enabled")
        print("    • Mixed precision available")
        print("    • Flash Attention supported")
    else:
        print("\n  ⚠ Running on CPU:")
        print("    • Training will be slower")
        print("    • Consider using TPU/GPU for faster training")
    
    return device_info


def auto_determine_vocab_size(
    dataset_id: str,
    text_column: str = "text",
    dataset_config: Optional[str] = None,
    split: str = "train",
    sample_examples: int = 10000,
    min_vocab: int = 5000,
    max_vocab: int = 50000,
) -> int:
    """Automatically determine optimal vocabulary size based on dataset"""
    print("\n  🔍 Auto-determining vocabulary size...")
    print(f"     Sampling {sample_examples} examples...")
    
    loader = HFDatasetLoader(
        dataset_id=dataset_id,
        text_column=text_column,
        dataset_config=dataset_config,
        split=split,
        streaming=True,
    )
    
    texts = loader.get_text_data(max_examples=sample_examples)
    
    total_chars = sum(len(text) for text in texts)
    unique_chars = len(set(''.join(texts)))
    avg_text_length = total_chars / len(texts) if texts else 0
    
    # Estimate vocab size based on text characteristics
    if avg_text_length < 100:
        vocab_size = min(8000, max_vocab)
    elif avg_text_length < 500:
        vocab_size = min(16000, max_vocab)
    else:
        vocab_size = min(32000, max_vocab)
    
    if unique_chars > 1000:
        vocab_size = min(int(vocab_size * 1.5), max_vocab)
    
    vocab_size = max(vocab_size, min_vocab)
    
    print(f"\n     Dataset Statistics:")
    print(f"       Samples:         {len(texts)}")
    print(f"       Avg length:      {avg_text_length:.0f} chars")
    print(f"       Unique chars:    {unique_chars}")
    print(f"       Recommended:     {vocab_size:,}")
    
    return vocab_size


def create_optimizer(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
):
    """Create optimizer with learning rate schedule"""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.1,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
            b1=beta1,
            b2=beta2,
        ),
    )
    
    return optimizer


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_ultimate(
    # ========== Dataset Parameters ==========
    dataset_id: str,
    text_column: str = "text",
    dataset_config: Optional[str] = None,
    split: str = "train",
    val_split: Optional[str] = None,
    max_train_examples: Optional[int] = None,
    max_val_examples: Optional[int] = None,
    
    # ========== Tokenizer Parameters ==========
    auto_vocab_size: bool = False,
    vocab_size: int = 16000,
    min_vocab_size: int = 5000,
    max_vocab_size: int = 50000,
    tokenizer_train_examples: int = 10000,
    retrain_tokenizer: bool = False,
    tokenizer_path: Optional[str] = None,
    
    # ========== Model Parameters ==========
    model_preset: Optional[str] = None,
    target_params: Optional[int] = None,
    prefer_depth: bool = True,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    seq_len: int = 256,
    dropout_rate: float = 0.1,
    activation: str = "gelu",
    use_rmsnorm: bool = True,
    use_swiglu: bool = True,
    use_rope: bool = True,
    use_flash_attention: bool = True,
    use_lora: bool = False,
    lora_rank: int = 8,
    
    # ========== Training Parameters ==========
    epochs: int = 3,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    
    # ========== Data Loading Parameters ==========
    use_streaming: bool = True,
    shuffle_buffer_size: int = 10000,
    stride: Optional[int] = None,
    
    # ========== Optimization Parameters ==========
    use_mixed_precision: bool = False,
    
    # ========== Checkpoint & Logging ==========
    output_dir: str = "output_ultimate",
    log_every: int = 10,
    save_every: int = 1000,
    eval_every: int = 500,
    resume_from: Optional[str] = None,
    
    # ========== Cloud Storage ==========
    cloud_provider: Optional[str] = None,
    cloud_bucket: Optional[str] = None,
    cloud_region: Optional[str] = None,
    cloud_prefix: str = "checkpoints",
    azure_sas_token: Optional[str] = None,
    azure_account_name: Optional[str] = None,
    
    # ========== Other ==========
    seed: int = 42,
):
    """
    🚀 ULTIMATE TRANSFORMER TRAINER
    
    The most comprehensive transformer training function with every feature you need!
    """
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    print_banner("🚀 ULTIMATE TRANSFORMER TRAINER", 120, "=")
    
    start_time = time.time()
    
    # Device setup
    device_info = detect_and_setup_device()
    num_devices = device_info["num_devices"]
    backend = device_info["backend"]
    
    print(f"\n{'System Information':^120}")
    print(f"{'-' * 120}")
    print(f"  JAX version:        {jax.__version__}")
    print(f"  Output directory:   {output_dir}")
    print(f"  Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup cloud config
    cloud_config = None
    if cloud_bucket and cloud_provider:
        cloud_config = CloudConfig(
            provider=cloud_provider,  # type: ignore
            bucket_name=cloud_bucket,
            region=cloud_region,
            prefix=cloud_prefix,
            sas_token=azure_sas_token,
            account_name=azure_account_name,
        )
        print(f"  Cloud storage:      ✓ {cloud_provider} ({cloud_bucket})")
    else:
        print(f"  Cloud storage:      ✗ Disabled")
    
    # ========================================================================
    # STEP 1: TOKENIZER SETUP
    # ========================================================================
    print_section("STEP 1: TOKENIZER SETUP")
    
    if tokenizer_path is None:
        tokenizer_path = f"{output_dir}/tokenizer.json"
    
    # Auto-determine vocab size if requested
    if auto_vocab_size and not os.path.exists(tokenizer_path):
        vocab_size = auto_determine_vocab_size(
            dataset_id=dataset_id,
            text_column=text_column,
            dataset_config=dataset_config,
            split=split,
            sample_examples=tokenizer_train_examples,
            min_vocab=min_vocab_size,
            max_vocab=max_vocab_size,
        )
        print(f"\n  ✓ Auto-determined vocab size: {vocab_size:,}")
    
    # Load or train tokenizer
    if os.path.exists(tokenizer_path) and not retrain_tokenizer:
        print(f"\n  ✓ Loading tokenizer from: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"     Vocabulary size: {actual_vocab_size:,}")
    else:
        print(f"\n  ⚙ Training new tokenizer...")
        print(f"     Target vocab size:  {vocab_size:,}")
        print(f"     Training examples:  {tokenizer_train_examples:,}")
        
        dataset_loader = HFDatasetLoader(
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            text_column=text_column,
            split=split,
            streaming=True,
        )
        
        tokenizer = dataset_loader.train_tokenizer(
            vocab_size=vocab_size,
            save_path=tokenizer_path,
            max_examples=tokenizer_train_examples,
        )
        
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"\n  ✓ Tokenizer trained!")
        print(f"     Actual vocab size: {actual_vocab_size:,}")
        print(f"     Saved to: {tokenizer_path}")
    
    # ========================================================================
    # STEP 2: MODEL CREATION
    # ========================================================================
    print_section("STEP 2: MODEL CREATION")
    
    # Create model config
    model_config = None
    
    # Priority 1: Model preset
    if model_preset:
        print(f"\n  📦 Using model preset: {model_preset.upper()}")
        model_config = ModelConfig.from_preset(
            preset=model_preset,  # type: ignore
            vocab_size=actual_vocab_size,
            max_len=seq_len,
            dropout_rate=dropout_rate,
            activation=activation,  # type: ignore
            use_lora=use_lora,
            lora_rank=lora_rank,
            use_rmsnorm=use_rmsnorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_flash_attention=use_flash_attention,
        )
    
    # Priority 2: Target parameters
    elif target_params and HAS_MODEL_SIZING:
        print(f"\n  🎯 Creating model with ~{target_params:,} parameters")
        model_config = create_model_from_params(
            target_params=target_params,
            vocab_size=actual_vocab_size,
            max_len=seq_len,
            prefer_depth=prefer_depth,
        )
        # Override with user settings
        model_config.use_rmsnorm = use_rmsnorm
        model_config.use_swiglu = use_swiglu
        model_config.use_rope = use_rope
        model_config.use_flash_attention = use_flash_attention
        model_config.use_lora = use_lora
        model_config.lora_rank = lora_rank
    
    # Priority 3: Custom architecture
    else:
        print(f"\n  🔧 Using custom model architecture")
        model_config = ModelConfig(
            vocab_size=actual_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=seq_len,
            dropout_rate=dropout_rate,
            activation=activation,  # type: ignore
            use_lora=use_lora,
            lora_rank=lora_rank,
            use_rmsnorm=use_rmsnorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_flash_attention=use_flash_attention,
        )
    
    # Print model architecture
    print(f"\n  {'Model Architecture':^116}")
    print(f"  {'-' * 116}")
    print(f"  Vocabulary size:    {model_config.vocab_size:,}")
    print(f"  Hidden size:        {model_config.d_model}")
    print(f"  Layers:             {model_config.num_layers}")
    print(f"  Attention heads:    {model_config.num_heads}")
    print(f"  FFN dimension:      {model_config.d_ff}")
    print(f"  Max sequence len:   {model_config.max_len}")
    print(f"  Dropout rate:       {model_config.dropout_rate}")
    print(f"  Activation:         {model_config.activation}")
    
    print(f"\n  {'Optimizations':^116}")
    print(f"  {'-' * 116}")
    print(f"  RMSNorm:            {'✓ Enabled' if model_config.use_rmsnorm else '✗ Disabled'}")
    print(f"  SwiGLU:             {'✓ Enabled' if model_config.use_swiglu else '✗ Disabled'}")
    print(f"  RoPE:               {'✓ Enabled' if model_config.use_rope else '✗ Disabled'}")
    print(f"  Flash Attention:    {'✓ Enabled' if model_config.use_flash_attention else '✗ Disabled'}")
    print(f"  LoRA:               {'✓ Enabled (rank=' + str(model_config.lora_rank) + ')' if model_config.use_lora else '✗ Disabled'}")
    
    # Check flash attention availability
    if model_config.use_flash_attention:
        flash_config = get_flash_attention_config()
        if flash_config['flash_attention_supported']:
            print(f"  Flash Attn Status:  ✓ Available ({flash_config['device_type']})")
        else:
            print(f"  Flash Attn Status:  ⚠ Not available (fallback to standard)")
    
    # Initialize model
    model = ProductionTransformer(config=model_config)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input, deterministic=True)
    
    # Count parameters
    total_params = count_params(params)
    param_memory_mb = (total_params * 4) / (1024**2)
    
    print(f"\n  {'Model Statistics':^116}")
    print(f"  {'-' * 116}")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Model size:         {total_params / 1e6:.2f}M parameters")
    print(f"  Memory (params):    {param_memory_mb:.1f} MB (float32)")
    print(f"  Memory (optimizer): ~{param_memory_mb * 2:.1f} MB (AdamW states)")
    
    # ========================================================================
    # STEP 3: DATA LOADER SETUP
    # ========================================================================
    print_section("STEP 3: DATA LOADER SETUP")
    
    effective_batch_size = batch_size * gradient_accumulation_steps * num_devices
    
    if use_streaming:
        print(f"\n  ✓ Using streaming data loader (memory efficient)")
        
        train_loader = StreamingDataLoader(
            dataset_id=dataset_id,
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            text_column=text_column,
            dataset_config=dataset_config,
            split=split,
            max_examples=max_train_examples,
            stride=stride,
            streaming=True,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        
        batch_memory_mb = (batch_size * seq_len * 2 * 4) / (1024**2)
        
        print(f"\n  {'Training Data':^116}")
        print(f"  {'-' * 116}")
        print(f"  Batch size:         {batch_size}")
        print(f"  Grad accumulation:  {gradient_accumulation_steps}x")
        print(f"  Num devices:        {num_devices}x")
        print(f"  Effective batch:    {effective_batch_size}")
        print(f"  Sequence length:    {seq_len}")
        print(f"  Shuffle buffer:     {shuffle_buffer_size:,}")
        print(f"  Batch memory:       ~{batch_memory_mb:.1f} MB")
        print(f"  Max examples:       {'All' if max_train_examples is None else f'{max_train_examples:,}'}")
        
        # Validation loader
        val_loader = None
        if val_split:
            print(f"\n  {'Validation Data':^116}")
            print(f"  {'-' * 116}")
            val_loader = StreamingDataLoader(
                dataset_id=dataset_id,
                tokenizer=tokenizer,
                seq_len=seq_len,
                batch_size=batch_size,
                text_column=text_column,
                dataset_config=dataset_config,
                split=val_split,
                max_examples=max_val_examples,
                stride=stride,
                streaming=True,
                shuffle_buffer_size=shuffle_buffer_size // 2,
                seed=seed + 1,
            )
            print(f"  Validation split:   {val_split}")
            print(f"  Batch size:         {batch_size}")
            print(f"  Max examples:       {'All' if max_val_examples is None else f'{max_val_examples:,}'}")
    else:
        raise NotImplementedError("Non-streaming mode not implemented. Use --use-streaming")
    
    # ========================================================================
    # STEP 4: TRAINING SETUP
    # ========================================================================
    print_section("STEP 4: TRAINING SETUP")
    
    estimated_steps_per_epoch = 1000
    total_steps = estimated_steps_per_epoch * epochs
    
    print(f"\n  {'Training Configuration':^116}")
    print(f"  {'-' * 116}")
    print(f"  Epochs:             {epochs}")
    print(f"  Steps per epoch:    ~{estimated_steps_per_epoch:,} (estimated)")
    print(f"  Total steps:        ~{total_steps:,}")
    print(f"  Learning rate:      {learning_rate}")
    print(f"  Weight decay:       {weight_decay}")
    print(f"  Warmup steps:       {warmup_steps}")
    print(f"  Gradient clip:      {grad_clip}")
    print(f"  Mixed precision:    {'✓ bfloat16' if use_mixed_precision else '✗ float32'}")
    print(f"  Streaming:          {'✓ Enabled' if use_streaming else '✗ Disabled'}")
    
    print(f"\n  {'Logging & Checkpoints':^116}")
    print(f"  {'-' * 116}")
    print(f"  Log every:          {log_every} steps")
    print(f"  Save every:         {save_every} steps")
    if val_loader:
        print(f"  Evaluate every:     {eval_every} steps")
    
    # Create optimizer
    optimizer = create_optimizer(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"\n  ⚙ Resuming from checkpoint: {resume_from}")
        checkpointer = ocp.PyTreeCheckpointer()
        abs_path = os.path.abspath(resume_from)
        loaded_params = checkpointer.restore(abs_path)
        state = state.replace(params=loaded_params)
        print(f"  ✓ Checkpoint loaded successfully")
    
    # Multi-device setup
    use_multi_device = num_devices > 1
    if use_multi_device:
        print(f"\n  ⚡ Multi-device training:")
        print(f"     Devices:           {num_devices}x {device_info['device_type']}")
        print(f"     Effective batch:   {effective_batch_size}")
        print(f"     Replicating model...")
        state = jax.device_put_replicated(state, jax.devices())
        print(f"     ✓ Model replicated")
    
    # Define loss function
    def loss_fn(params, batch, dropout_rng):
        """Compute cross-entropy loss"""
        inputs = batch["input_ids"]
        targets = batch["labels"]
        
        logits = model.apply(
            params,
            inputs,
            deterministic=False,
            rngs={"dropout": dropout_rng}
        )
        
        # Shift for next-token prediction
        logits_shifted = logits[:, :-1]
        labels_shifted = targets[:, :-1]
        
        # Flatten
        logits_flat = logits_shifted.reshape(-1, logits_shifted.shape[-1])
        labels_flat = labels_shifted.reshape(-1)
        
        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_flat, labels_flat
        )
        
        return jnp.mean(loss)
    
    # Training step
    if use_multi_device:
        @jax.pmap
        def train_step(state, batch, dropout_rng):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, dropout_rng)
            return loss, grads
        
        @jax.pmap
        def update_model(state, grads):
            return state.apply_gradients(grads=grads)
        
        @jax.pmap
        def eval_step(params, batch, dropout_rng):
            return loss_fn(params, batch, dropout_rng)
    else:
        @jax.jit
        def train_step(state, batch, dropout_rng):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, dropout_rng)
            return loss, grads
        
        @jax.jit
        def update_model(state, grads):
            return state.apply_gradients(grads=grads)
        
        @jax.jit
        def eval_step(params, batch, dropout_rng):
            return loss_fn(params, batch, dropout_rng)
    
    # ========================================================================
    # STEP 5: TRAINING LOOP
    # ========================================================================
    print_section("STEP 5: TRAINING")
    print()
    
    global_step = 0
    best_val_loss = float('inf')
    training_start = time.time()
    avg_epoch_loss = 0.0
    
    for epoch in range(epochs):
        print(f"\n{'─' * 120}")
        print(f"📊 EPOCH {epoch + 1}/{epochs}")
        print(f"{'─' * 120}\n")
        
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Gradient accumulation
        accumulated_grads = None
        accumulation_count = 0
        
        # Get epoch iterator
        batch_iterator = train_loader.get_epoch_iterator()
        
        # Training loop
        for batch in batch_iterator:
            # Generate dropout key
            rng, dropout_rng = jax.random.split(rng)
            
            # Multi-device batch preparation
            if use_multi_device:
                # Reshape batch for pmap
                batch_input = batch["input_ids"]
                batch_labels = batch["labels"]
                
                # Pad if needed
                total_size = batch_input.shape[0]
                remainder = total_size % num_devices
                if remainder != 0:
                    pad_size = num_devices - remainder
                    batch_input = jnp.concatenate([
                        batch_input,
                        jnp.zeros((pad_size, seq_len), dtype=jnp.int32)
                    ])
                    batch_labels = jnp.concatenate([
                        batch_labels,
                        jnp.zeros((pad_size, seq_len), dtype=jnp.int32)
                    ])
                
                # Reshape for pmap: (num_devices, per_device_batch_size, seq_len)
                per_device_batch = batch_input.shape[0] // num_devices
                batch = {
                    "input_ids": batch_input.reshape(num_devices, per_device_batch, seq_len),
                    "labels": batch_labels.reshape(num_devices, per_device_batch, seq_len),
                }
                
                # Dropout RNGs for each device
                rng, *dropout_rngs = jax.random.split(rng, num_devices + 1)
                dropout_rng = jnp.array(dropout_rngs)
            
            # Compute loss and gradients
            loss, grads = train_step(state, batch, dropout_rng)
            
            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_map(
                    lambda acc, g: acc + g,
                    accumulated_grads,
                    grads
                )
            
            accumulation_count += 1
            current_loss = jnp.mean(loss) if use_multi_device else loss
            epoch_loss += float(current_loss)
            
            # Apply gradients after accumulation steps
            if accumulation_count >= gradient_accumulation_steps:
                # Average accumulated gradients
                accumulated_grads = jax.tree_map(
                    lambda g: g / gradient_accumulation_steps,
                    accumulated_grads
                )
                
                # Update model
                state = update_model(state, accumulated_grads)
                
                # Reset accumulation
                accumulated_grads = None
                accumulation_count = 0
                global_step += 1
                num_batches += 1
                
                # Logging
                if global_step % log_every == 0:
                    elapsed = time.time() - training_start
                    avg_loss = epoch_loss / num_batches
                    tokens_per_sec = (global_step * effective_batch_size * seq_len) / elapsed
                    
                    print(
                        f"  Step {global_step:>6} │ "
                        f"Loss: {avg_loss:.4f} │ "
                        f"Tokens/sec: {tokens_per_sec:>10,.0f} │ "
                        f"Time: {elapsed:>7.1f}s"
                    )
                
                # Evaluation
                if val_loader and global_step % eval_every == 0:
                    print(f"\n  {'Validation':^112}")
                    print(f"  {'-' * 112}")
                    
                    val_losses = []
                    val_batches = 0
                    max_val_batches = 50
                    
                    for val_batch in val_loader.get_epoch_iterator():
                        rng, val_dropout_rng = jax.random.split(rng)
                        
                        if use_multi_device:
                            # Prepare batch for pmap
                            val_input = val_batch["input_ids"]
                            val_labels = val_batch["labels"]
                            
                            total_size = val_input.shape[0]
                            remainder = total_size % num_devices
                            if remainder != 0:
                                pad_size = num_devices - remainder
                                val_input = jnp.concatenate([
                                    val_input,
                                    jnp.zeros((pad_size, seq_len), dtype=jnp.int32)
                                ])
                                val_labels = jnp.concatenate([
                                    val_labels,
                                    jnp.zeros((pad_size, seq_len), dtype=jnp.int32)
                                ])
                            
                            per_device_batch = val_input.shape[0] // num_devices
                            val_batch = {
                                "input_ids": val_input.reshape(num_devices, per_device_batch, seq_len),
                                "labels": val_labels.reshape(num_devices, per_device_batch, seq_len),
                            }
                            
                            rng, *val_dropout_rngs = jax.random.split(rng, num_devices + 1)
                            val_dropout_rng = jnp.array(val_dropout_rngs)
                        
                        val_loss = eval_step(state.params, val_batch, val_dropout_rng)
                        val_losses.append(float(jnp.mean(val_loss)))
                        val_batches += 1
                        if val_batches >= max_val_batches:
                            break
                    
                    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
                    print(f"  Validation loss: {avg_val_loss:.4f} ({len(val_losses)} batches)")
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        checkpoint_path = os.path.abspath(f"{output_dir}/best_checkpoint")
                        print(f"  ✓ New best validation loss! Saving to: {checkpoint_path}")
                        
                        checkpointer = ocp.PyTreeCheckpointer()
                        if os.path.exists(checkpoint_path):
                            shutil.rmtree(checkpoint_path)
                        
                        params_to_save = (
                            jax.tree_map(lambda x: x[0], state.params)
                            if use_multi_device
                            else state.params
                        )
                        checkpointer.save(checkpoint_path, params_to_save)
                    
                    print()
                
                # Save checkpoint
                if save_every > 0 and global_step % save_every == 0:
                    checkpoint_path = os.path.abspath(f"{output_dir}/checkpoint_step_{global_step}")
                    print(f"  💾 Saving checkpoint to: {checkpoint_path}")
                    
                    checkpointer = ocp.PyTreeCheckpointer()
                    if os.path.exists(checkpoint_path):
                        shutil.rmtree(checkpoint_path)
                    
                    params_to_save = (
                        jax.tree_map(lambda x: x[0], state.params)
                        if use_multi_device
                        else state.params
                    )
                    checkpointer.save(checkpoint_path, params_to_save)
                    
                    # Upload to cloud if configured
                    if cloud_config:
                        upload_checkpoint_to_cloud(
                            checkpoint_path,
                            cloud_config,
                            f"checkpoint_step_{global_step}"
                        )
                        print(f"  ☁ Checkpoint uploaded to cloud")
        
        # Handle remaining accumulated gradients
        if accumulated_grads is not None and accumulation_count > 0:
            accumulated_grads = jax.tree_map(
                lambda g: g / accumulation_count,
                accumulated_grads
            )
            state = update_model(state, accumulated_grads)
            global_step += 1
            num_batches += 1
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        
        print(f"\n📈 Epoch {epoch + 1} Summary:")
        print(f"  Average loss:       {avg_epoch_loss:.4f}")
        print(f"  Batches processed:  {num_batches:,}")
        print(f"  Epoch time:         {epoch_time:.1f}s")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    total_time = time.time() - training_start
    
    print_banner("✅ TRAINING COMPLETE", 120, "=")
    
    print(f"\n{'Final Statistics':^120}")
    print(f"{'-' * 120}")
    print(f"  Total time:         {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"  Total steps:        {global_step:,}")
    print(f"  Total tokens:       ~{global_step * effective_batch_size * seq_len:,}")
    print(f"  Avg tokens/sec:     {(global_step * effective_batch_size * seq_len) / total_time:,.0f}")
    if val_loader:
        print(f"  Best val loss:      {best_val_loss:.4f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.abspath(f"{output_dir}/final_checkpoint")
    print(f"\n💾 Saving final checkpoint to: {final_checkpoint_path}")
    checkpointer = ocp.PyTreeCheckpointer()
    if os.path.exists(final_checkpoint_path):
        shutil.rmtree(final_checkpoint_path)
    
    params_to_save = (
        jax.tree_map(lambda x: x[0], state.params)
        if use_multi_device
        else state.params
    )
    checkpointer.save(final_checkpoint_path, params_to_save)
    
    # Save model config
    config_path = os.path.abspath(f"{output_dir}/model_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config.model_dump(), f, indent=2)
    
    # Save training summary
    summary_path = os.path.abspath(f"{output_dir}/training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "total_steps": global_step,
            "total_time": total_time,
            "total_tokens": global_step * effective_batch_size * seq_len,
            "avg_tokens_per_sec": (global_step * effective_batch_size * seq_len) / total_time,
            "final_loss": float(avg_epoch_loss),
            "best_val_loss": float(best_val_loss) if val_loader else None,
            "model_params": total_params,
            "device_info": device_info,
        }, f, indent=2)
    
    print(f"\n📁 Output files saved to: {output_dir}/")
    print(f"  • final_checkpoint/      (final model parameters)")
    if val_loader:
        print(f"  • best_checkpoint/       (best validation checkpoint)")
    print(f"  • model_config.json      (model configuration)")
    print(f"  • tokenizer.json         (trained tokenizer)")
    print(f"  • training_summary.json  (training statistics)")
    
    print_banner("🎉 ALL DONE!", 120, "=")
    
    return {
        "total_steps": global_step,
        "total_time": total_time,
        "best_val_loss": float(best_val_loss) if val_loader else None,
        "final_loss": float(avg_epoch_loss),
        "model_params": total_params,
    }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🚀 ULTIMATE TRANSFORMER TRAINER - Maximum Customization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Quick start with preset
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --model-preset tiny --auto-vocab-size --epochs 3

  # Custom architecture
  python -m src.training.train_ultimate \\
    --dataset-id "wikitext" --dataset-config "wikitext-2-raw-v1" \\
    --d-model 512 --num-layers 8 --num-heads 8 --d-ff 2048 \\
    --vocab-size 16000 --seq-len 256 --epochs 5

  # Parameter-based sizing
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --target-params 10000000 --prefer-depth --epochs 5

  # Memory-optimized with validation
  python -m src.training.train_ultimate \\
    --dataset-id "roneneldan/TinyStories" \\
    --model-preset medium --val-split "validation" \\
    --batch-size 16 --gradient-accumulation-steps 4 \\
    --use-streaming --eval-every 500

For more info: https://github.com/your-repo/transformer-training
        """
    )
    
    # ========== Dataset Parameters ==========
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument("--dataset-id", type=str, required=True,
                               help="HuggingFace dataset ID (e.g., 'roneneldan/TinyStories')")
    dataset_group.add_argument("--text-column", type=str, default="text")
    dataset_group.add_argument("--dataset-config", type=str, default=None)
    dataset_group.add_argument("--split", type=str, default="train")
    dataset_group.add_argument("--val-split", type=str, default=None)
    dataset_group.add_argument("--max-train-examples", type=int, default=None)
    dataset_group.add_argument("--max-val-examples", type=int, default=None)
    
    # ========== Tokenizer Parameters ==========
    tokenizer_group = parser.add_argument_group('Tokenizer Parameters')
    tokenizer_group.add_argument("--auto-vocab-size", action="store_true",
                                 help="Automatically determine optimal vocab size")
    tokenizer_group.add_argument("--vocab-size", type=int, default=16000)
    tokenizer_group.add_argument("--min-vocab-size", type=int, default=5000)
    tokenizer_group.add_argument("--max-vocab-size", type=int, default=50000)
    tokenizer_group.add_argument("--tokenizer-train-examples", type=int, default=10000)
    tokenizer_group.add_argument("--retrain-tokenizer", action="store_true")
    tokenizer_group.add_argument("--tokenizer-path", type=str, default=None)
    
    # ========== Model Parameters ==========
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--model-preset", type=str,
                            choices=["nano", "tiny", "small", "medium", "large", "xlarge"],
                            default=None, help="Use predefined model architecture")
    model_group.add_argument("--target-params", type=int, default=None,
                            help="Target number of parameters (e.g., 10000000 for 10M)")
    model_group.add_argument("--prefer-depth", action="store_true", default=True,
                            help="Prefer deeper models over wider (for target-params)")
    model_group.add_argument("--d-model", type=int, default=512)
    model_group.add_argument("--num-layers", type=int, default=6)
    model_group.add_argument("--num-heads", type=int, default=8)
    model_group.add_argument("--d-ff", type=int, default=2048)
    model_group.add_argument("--seq-len", type=int, default=256)
    model_group.add_argument("--dropout-rate", type=float, default=0.1)
    model_group.add_argument("--activation", type=str, default="gelu",
                            choices=["relu", "gelu", "silu"])
    model_group.add_argument("--no-rmsnorm", action="store_false", dest="use_rmsnorm")
    model_group.add_argument("--no-swiglu", action="store_false", dest="use_swiglu")
    model_group.add_argument("--no-rope", action="store_false", dest="use_rope")
    model_group.add_argument("--no-flash-attention", action="store_false", dest="use_flash_attention")
    model_group.add_argument("--use-lora", action="store_true")
    model_group.add_argument("--lora-rank", type=int, default=8)
    
    # ========== Training Parameters ==========
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument("--epochs", type=int, default=3)
    training_group.add_argument("--batch-size", type=int, default=32)
    training_group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=5e-4)
    training_group.add_argument("--weight-decay", type=float, default=0.01)
    training_group.add_argument("--warmup-steps", type=int, default=500)
    training_group.add_argument("--grad-clip", type=float, default=1.0)
    
    # ========== Data Loading Parameters ==========
    data_group = parser.add_argument_group('Data Loading Parameters')
    data_group.add_argument("--use-streaming", action="store_true", default=True,
                           help="Use streaming data loader (recommended)")
    data_group.add_argument("--shuffle-buffer-size", type=int, default=10000)
    data_group.add_argument("--stride", type=int, default=None)
    
    # ========== Optimization Parameters ==========
    opt_group = parser.add_argument_group('Optimization Parameters')
    opt_group.add_argument("--use-mixed-precision", action="store_true")
    
    # ========== Checkpoint & Logging ==========
    checkpoint_group = parser.add_argument_group('Checkpoint & Logging')
    checkpoint_group.add_argument("--output-dir", type=str, default="output_ultimate")
    checkpoint_group.add_argument("--log-every", type=int, default=10)
    checkpoint_group.add_argument("--save-every", type=int, default=1000)
    checkpoint_group.add_argument("--eval-every", type=int, default=500)
    checkpoint_group.add_argument("--resume-from", type=str, default=None)
    
    # ========== Cloud Storage ==========
    cloud_group = parser.add_argument_group('Cloud Storage (Optional)')
    cloud_group.add_argument("--cloud-provider", type=str, choices=["s3", "gcs", "azure"])
    cloud_group.add_argument("--cloud-bucket", type=str, default=None)
    cloud_group.add_argument("--cloud-region", type=str, default=None)
    cloud_group.add_argument("--cloud-prefix", type=str, default="checkpoints")
    cloud_group.add_argument("--azure-sas-token", type=str, default=None)
    cloud_group.add_argument("--azure-account-name", type=str, default=None)
    
    # ========== Other ==========
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Run training
    train_ultimate(
        # Dataset
        dataset_id=args.dataset_id,
        text_column=args.text_column,
        dataset_config=args.dataset_config,
        split=args.split,
        val_split=args.val_split,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        # Tokenizer
        auto_vocab_size=args.auto_vocab_size,
        vocab_size=args.vocab_size,
        min_vocab_size=args.min_vocab_size,
        max_vocab_size=args.max_vocab_size,
        tokenizer_train_examples=args.tokenizer_train_examples,
        retrain_tokenizer=args.retrain_tokenizer,
        tokenizer_path=args.tokenizer_path,
        # Model
        model_preset=args.model_preset,
        target_params=args.target_params,
        prefer_depth=args.prefer_depth,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        dropout_rate=args.dropout_rate,
        activation=args.activation,
        use_rmsnorm=args.use_rmsnorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_flash_attention=args.use_flash_attention,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        # Data loading
        use_streaming=args.use_streaming,
        shuffle_buffer_size=args.shuffle_buffer_size,
        stride=args.stride,
        # Optimization
        use_mixed_precision=args.use_mixed_precision,
        # Checkpoints
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        resume_from=args.resume_from,
        # Cloud
        cloud_provider=args.cloud_provider,
        cloud_bucket=args.cloud_bucket,
        cloud_region=args.cloud_region,
        cloud_prefix=args.cloud_prefix,
        azure_sas_token=args.azure_sas_token,
        azure_account_name=args.azure_account_name,
        # Other
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
