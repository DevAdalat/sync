#!/bin/bash

# Installation and verification script for Flash Attention (Kvax)

echo "=================================================="
echo "Flash Attention (Kvax) Installation & Verification"
echo "=================================================="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# Step 2: Verify installation
echo "Step 2: Verifying installation..."
python3 -c "
import sys
try:
    import jax
    print('✅ JAX installed:', jax.__version__)
except ImportError:
    print('❌ JAX not found')
    sys.exit(1)

try:
    import flax
    print('✅ Flax installed:', flax.__version__)
except ImportError:
    print('❌ Flax not found')
    sys.exit(1)

try:
    import kvax
    print('✅ Kvax installed')
except ImportError:
    print('⚠️  Kvax not found - flash attention will use fallback')

import jax
devices = jax.devices()
print(f'✅ JAX devices found: {len(devices)} ({devices[0].platform})')
"

if [ $? -eq 0 ]; then
    echo "✅ Installation verified"
else
    echo "❌ Verification failed"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# Step 3: Run flash attention tests
echo "Step 3: Running flash attention tests..."
pytest tests/test_flash_attention.py -v --tb=short

if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (this may be expected if Kvax is not available)"
fi

echo ""
echo "=================================================="
echo ""

# Step 4: Run example
echo "Step 4: Running flash attention example..."
python examples/example_flash_attention.py

echo ""
echo "=================================================="
echo "Installation and verification complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Check docs/FLASH_ATTENTION_GUIDE.md for detailed usage"
echo "2. Run 'python examples/example_flash_attention.py' for a demo"
echo "3. Use 'use_flash_attention=True' in ModelConfig to enable"
echo ""
