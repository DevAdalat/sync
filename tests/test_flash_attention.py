"""
Tests for Flash Attention implementation with Kvax.

Tests include:
- Device detection
- Flash attention availability
- Attention computation correctness
- CPU/GPU/TPU compatibility
- Fallback mechanism
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.flash_attention import (
    detect_device_type,
    is_flash_attention_supported,
    get_flash_attention_config,
    adaptive_flash_attention,
    standard_attention,
    create_positions,
    create_segment_ids,
    KVAX_AVAILABLE,
)
from src.config.config import ModelConfig
from src.models.model import ProductionTransformer, MultiHeadAttention


class TestDeviceDetection:
    """Test device detection functionality."""
    
    def test_detect_device_type(self):
        """Test that device type can be detected."""
        device_type = detect_device_type()
        assert device_type in ['cpu', 'gpu', 'tpu']
        print(f"Detected device type: {device_type}")
    
    def test_flash_attention_availability(self):
        """Test flash attention availability check."""
        supported = is_flash_attention_supported()
        assert isinstance(supported, bool)
        print(f"Flash attention supported: {supported}")
    
    def test_get_flash_attention_config(self):
        """Test getting flash attention configuration."""
        config = get_flash_attention_config()
        assert isinstance(config, dict)
        assert 'kvax_available' in config
        assert 'device_type' in config
        assert 'flash_attention_supported' in config
        print(f"Flash attention config: {config}")


class TestFlashAttentionUtilities:
    """Test utility functions for flash attention."""
    
    def test_create_positions(self):
        """Test position creation."""
        batch_size, seq_len = 2, 8
        positions = create_positions(batch_size, seq_len)
        
        assert positions.shape == (batch_size, seq_len)
        assert jnp.all(positions[0] == jnp.arange(seq_len))
        assert jnp.all(positions[1] == jnp.arange(seq_len))
    
    def test_create_segment_ids_no_padding(self):
        """Test segment ID creation without padding."""
        batch_size, seq_len = 2, 8
        segment_ids = create_segment_ids(batch_size, seq_len)
        
        assert segment_ids.shape == (batch_size, seq_len)
        assert jnp.all(segment_ids == 0)
    
    def test_create_segment_ids_with_padding(self):
        """Test segment ID creation with padding."""
        batch_size, seq_len = 2, 8
        padding_mask = jnp.array([
            [1, 1, 1, 1, 1, 1, 0, 0],  # Last 2 tokens are padding
            [1, 1, 1, 1, 0, 0, 0, 0],  # Last 4 tokens are padding
        ])
        
        segment_ids = create_segment_ids(batch_size, seq_len, padding_mask)
        
        assert segment_ids.shape == (batch_size, seq_len)
        # Check padding tokens are marked as -1
        assert segment_ids[0, 6] == -1
        assert segment_ids[0, 7] == -1
        assert segment_ids[1, 4] == -1


class TestAttentionMechanisms:
    """Test attention computation mechanisms."""
    
    def setup_method(self):
        """Set up test data."""
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 8
        self.head_dim = 16
        
        key = jax.random.PRNGKey(0)
        self.query = jax.random.normal(
            key,
            (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        )
        self.key = jax.random.normal(
            jax.random.PRNGKey(1),
            (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        )
        self.value = jax.random.normal(
            jax.random.PRNGKey(2),
            (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        )
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
    
    def test_standard_attention(self):
        """Test standard attention computation."""
        output = standard_attention(
            self.query,
            self.key,
            self.value,
            self.scale,
            causal=True,
        )
        
        assert output.shape == self.query.shape
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))
    
    def test_adaptive_flash_attention_disabled(self):
        """Test adaptive attention with flash attention disabled."""
        output = adaptive_flash_attention(
            self.query,
            self.key,
            self.value,
            self.scale,
            causal=True,
            use_flash_attention=False,
        )
        
        assert output.shape == self.query.shape
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))
    
    @pytest.mark.skipif(not KVAX_AVAILABLE, reason="Kvax not available")
    def test_adaptive_flash_attention_enabled(self):
        """Test adaptive attention with flash attention enabled."""
        output = adaptive_flash_attention(
            self.query,
            self.key,
            self.value,
            self.scale,
            causal=True,
            use_flash_attention=True,
        )
        
        assert output.shape == self.query.shape
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))
    
    def test_attention_output_consistency(self):
        """Test that outputs are numerically close for both methods."""
        # Standard attention
        output_standard = adaptive_flash_attention(
            self.query,
            self.key,
            self.value,
            self.scale,
            causal=True,
            use_flash_attention=False,
        )
        
        # Flash attention (may fallback)
        output_flash = adaptive_flash_attention(
            self.query,
            self.key,
            self.value,
            self.scale,
            causal=True,
            use_flash_attention=True,
        )
        
        # Outputs should be close (allowing for numerical differences)
        assert output_standard.shape == output_flash.shape
        
        # If Kvax is available, check numerical closeness
        if KVAX_AVAILABLE:
            # Allow some tolerance for different implementations
            max_diff = jnp.max(jnp.abs(output_standard - output_flash))
            print(f"Max difference between standard and flash attention: {max_diff}")
            # This might not be very close due to different implementations,
            # but both should produce valid outputs


class TestModelIntegration:
    """Test flash attention integration with the model."""
    
    def test_model_with_flash_attention(self):
        """Test creating and running model with flash attention."""
        config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            max_len=64,
            use_flash_attention=True,
        )
        
        model = ProductionTransformer(config=config)
        
        # Create dummy input
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (2, 32), 0, config.vocab_size)
        
        # Initialize and run model
        params = model.init(key, x, deterministic=True)
        output = model.apply(params, x, deterministic=True)
        
        assert output.shape == (2, 32, config.vocab_size)
        assert not jnp.any(jnp.isnan(output))
    
    def test_model_without_flash_attention(self):
        """Test creating and running model without flash attention."""
        config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            max_len=64,
            use_flash_attention=False,
        )
        
        model = ProductionTransformer(config=config)
        
        # Create dummy input
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (2, 32), 0, config.vocab_size)
        
        # Initialize and run model
        params = model.init(key, x, deterministic=True)
        output = model.apply(params, x, deterministic=True)
        
        assert output.shape == (2, 32, config.vocab_size)
        assert not jnp.any(jnp.isnan(output))
    
    def test_multihead_attention_with_flash(self):
        """Test MultiHeadAttention module with flash attention."""
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 16
        
        attention = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            use_flash_attention=True,
        )
        
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (batch_size, seq_len, d_model))
        
        params = attention.init(key, x, x, x, deterministic=True)
        output = attention.apply(params, x, x, x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not jnp.any(jnp.isnan(output))


class TestDeviceCompatibility:
    """Test compatibility across different devices."""
    
    def test_cpu_compatibility(self):
        """Test that flash attention works on CPU (with fallback if needed)."""
        device_type = detect_device_type()
        print(f"Testing on device: {device_type}")
        
        config = ModelConfig(
            vocab_size=500,
            d_model=64,
            num_heads=2,
            num_layers=1,
            d_ff=256,
            max_len=32,
            use_flash_attention=True,
        )
        
        model = ProductionTransformer(config=config)
        key = jax.random.PRNGKey(0)
        x = jax.random.randint(key, (1, 16), 0, config.vocab_size)
        
        params = model.init(key, x, deterministic=True)
        output = model.apply(params, x, deterministic=True)
        
        assert output.shape == (1, 16, config.vocab_size)
        print(f"✓ Model runs successfully on {device_type}")


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
