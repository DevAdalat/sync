# Text Generation Guide

Complete guide for generating text from your trained transformer models.

## Quick Start

### Basic Generation

```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --prompt "Once upon a time"
```

### Example Output

```
================================================================================
                              TEXT GENERATION                                   
================================================================================

Backend: GPU
Loading tokenizer from: output/tokenizer.json
Loading config from: output/model_config.json
Creating model...
Loading checkpoint from: output/best_checkpoint
✓ Model loaded successfully!
  Vocab size: 8986
  Max length: 128
  Model size: 64
  Layers: 92

================================================================================
GENERATING TEXT
================================================================================
Prompt: Once upon a time
Prompt tokens: 5
Max new tokens: 100
Temperature: 0.8
Top-k: 50
Top-p: disabled

────────────────────────────────────────────────────────────────────────────────
Generated text:
────────────────────────────────────────────────────────────────────────────────
Once upon a time, there was a little girl named Lily. She loved to play outside...
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--checkpoint` | Path to checkpoint directory | `output/best_checkpoint` |
| `--prompt` | Input text prompt | `"Once upon a time"` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tokenizer-path` | Auto-detected | Path to tokenizer.json file |
| `--max-length` | 100 | Maximum number of tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (0.1-2.0) |
| `--top-k` | 50 | Top-k sampling (0 = disabled) |
| `--top-p` | 1.0 | Nucleus sampling (0.0-1.0) |
| `--seed` | 42 | Random seed for reproducibility |

## Generation Parameters Explained

### Temperature

Controls randomness in generation:

- **Low (0.1-0.7)**: More focused, deterministic, coherent
  ```bash
  --temperature 0.5  # Conservative, safe completions
  ```

- **Medium (0.7-1.0)**: Balanced creativity and coherence
  ```bash
  --temperature 0.8  # Default, good balance
  ```

- **High (1.0-2.0)**: More creative, random, diverse
  ```bash
  --temperature 1.5  # Wild, creative completions
  ```

**Examples**:

```bash
# Very focused generation (good for facts/code)
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "The capital of France is" --temperature 0.3

# Balanced generation (good for stories)
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Once upon a time" --temperature 0.8

# Creative generation (good for brainstorming)
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Imagine a world where" --temperature 1.5
```

### Top-K Sampling

Limits selection to K most likely tokens:

- **Low (10-30)**: Very focused, limited vocabulary
- **Medium (40-60)**: Good balance (default: 50)
- **High (80-100)**: More diverse
- **Disabled (0)**: All tokens considered

```bash
# Focused with small vocabulary
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Hello" --top-k 20

# Default balanced
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Hello" --top-k 50

# Disable for maximum diversity
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Hello" --top-k 0
```

### Top-P (Nucleus Sampling)

Selects from smallest set of tokens whose cumulative probability >= p:

- **Low (0.5-0.7)**: Very focused
- **Medium (0.8-0.95)**: Balanced (recommended: 0.9)
- **Disabled (1.0)**: All tokens considered (default)

```bash
# Focused nucleus sampling
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "In conclusion" --top-p 0.7

# Balanced nucleus sampling
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "The story begins" --top-p 0.9

# Combine with temperature for fine control
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Once upon a time" --temperature 0.7 --top-p 0.9
```

### Max Length

Controls how many tokens to generate:

```bash
# Short completion
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Hello world" --max-length 20

# Medium completion
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Once upon a time" --max-length 100

# Long completion
python generate_text.py --checkpoint output/best_checkpoint \
    --prompt "Chapter 1:" --max-length 500
```

**Note**: Generation stops early if:
- End-of-sequence token is generated
- Model's maximum sequence length is reached

## Common Use Cases

### 1. Story Generation

```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --prompt "Once upon a time in a magical forest" \
    --max-length 200 \
    --temperature 0.9 \
    --top-k 50
```

### 2. Dialogue Completion

```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --prompt "Person A: How are you?\nPerson B:" \
    --max-length 50 \
    --temperature 0.7 \
    --top-p 0.9
```

### 3. Question Answering

```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --prompt "Question: What is the meaning of life?\nAnswer:" \
    --max-length 100 \
    --temperature 0.5 \
    --top-k 40
```

### 4. Code Completion

```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --prompt "def calculate_fibonacci(n):" \
    --max-length 150 \
    --temperature 0.3 \
    --top-k 30
```

### 5. Creative Writing

```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --prompt "In a world where magic is real" \
    --max-length 300 \
    --temperature 1.2 \
    --top-p 0.95
```

## Recommended Parameter Combinations

### Conservative (Facts, Definitions, Code)
```bash
--temperature 0.3 --top-k 30 --top-p 0.8
```

### Balanced (General Purpose)
```bash
--temperature 0.8 --top-k 50 --top-p 0.9
```

### Creative (Stories, Poetry, Brainstorming)
```bash
--temperature 1.2 --top-k 60 --top-p 0.95
```

### Experimental (Maximum Diversity)
```bash
--temperature 1.5 --top-k 0 --top-p 1.0
```

## Troubleshooting

### Error: "Could not find tokenizer.json"

**Solution**: Specify tokenizer path explicitly:
```bash
python generate_text.py \
    --checkpoint output/best_checkpoint \
    --tokenizer-path output/tokenizer.json \
    --prompt "Hello"
```

### Error: "Model config not found"

**Cause**: `model_config.json` must be in the same directory as checkpoint parent.

**Expected structure**:
```
output/
├── tokenizer.json
├── model_config.json
└── best_checkpoint/
    └── (checkpoint files)
```

**Solution**: Make sure you're pointing to the checkpoint directory, not the parent:
```bash
# ✓ Correct
python generate_text.py --checkpoint output/best_checkpoint

# ✗ Wrong
python generate_text.py --checkpoint output
```

### Generation is Repetitive

**Solutions**:
1. Increase temperature: `--temperature 1.0`
2. Enable top-p: `--top-p 0.9`
3. Increase top-k: `--top-k 60`
4. Try different random seed: `--seed 123`

### Generation is Incoherent

**Solutions**:
1. Decrease temperature: `--temperature 0.6`
2. Decrease top-k: `--top-k 30`
3. Enable nucleus sampling: `--top-p 0.8`

### Generation is Too Short

**Causes**:
- Model generates end-of-sequence token early
- Max length is too low

**Solutions**:
1. Increase max length: `--max-length 200`
2. Try different prompt
3. Adjust temperature: `--temperature 0.8`

### Out of Memory During Generation

**Solutions**:
1. Use CPU backend (slower but more memory):
   ```bash
   export JAX_PLATFORMS=cpu
   python generate_text.py ...
   ```

2. Generate shorter sequences: `--max-length 50`

3. Use smaller model (retrain with fewer parameters)

## Batch Generation Script

For generating multiple completions:

```bash
# Create batch_generate.sh
cat > batch_generate.sh << 'EOF'
#!/bin/bash

CHECKPOINT="output/best_checkpoint"
PROMPTS=(
    "Once upon a time"
    "In a galaxy far away"
    "The quick brown fox"
    "Hello world"
)

for prompt in "${PROMPTS[@]}"; do
    echo "Generating for prompt: $prompt"
    python generate_text.py \
        --checkpoint "$CHECKPOINT" \
        --prompt "$prompt" \
        --max-length 100 \
        --temperature 0.8
    echo ""
done
EOF

chmod +x batch_generate.sh
./batch_generate.sh
```

## Python API Usage

You can also use the generation functions in your own Python scripts:

```python
from generate_text import load_model_and_tokenizer, generate_text

# Load model
model, params, tokenizer, config = load_model_and_tokenizer(
    checkpoint_dir="output/best_checkpoint",
    tokenizer_path="output/tokenizer.json"
)

# Generate text
result = generate_text(
    model=model,
    params=params,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=1.0,
    rng_seed=42
)

print(result)
```

## Tips for Better Generation

1. **Prompt Engineering**: Be specific and clear in your prompts
   - ✓ Good: "Write a short story about a brave knight:"
   - ✗ Bad: "story"

2. **Experiment**: Try different parameter combinations for your use case

3. **Reproducibility**: Use `--seed` for consistent results

4. **Quality Control**: Lower temperature for factual content, higher for creative

5. **Context**: Include relevant context in your prompt for better results

6. **Testing**: Generate multiple times with different seeds to see variety

## Advanced: Interactive Generation

Create an interactive generation session:

```bash
# Create interactive_generate.sh
cat > interactive_generate.sh << 'EOF'
#!/bin/bash

CHECKPOINT="output/best_checkpoint"

while true; do
    echo ""
    read -p "Enter prompt (or 'quit' to exit): " prompt
    
    if [ "$prompt" = "quit" ]; then
        break
    fi
    
    python generate_text.py \
        --checkpoint "$CHECKPOINT" \
        --prompt "$prompt" \
        --max-length 100 \
        --temperature 0.8 \
        --top-k 50
done
EOF

chmod +x interactive_generate.sh
./interactive_generate.sh
```

## Conclusion

The `generate_text.py` script provides a flexible interface for text generation with your trained models. Experiment with different parameters to find what works best for your use case!

**Key Takeaways**:
- Start with default parameters (`temp=0.8, top-k=50`)
- Lower temperature for focused/factual content
- Higher temperature for creative content
- Use top-p for quality control
- Adjust max-length based on needs
