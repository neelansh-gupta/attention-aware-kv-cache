"""
Attention-Aware KV Cache Compression
Postman AI/ML Recruitment Task 3

Package layout (populated incrementally across stages):
    model_wrapper.py  -> model loading, KV cache inspection, attention instrumentation
    cache_manager.py  -> custom KV cache with eviction support and position tracking
    evictions.py       -> eviction policy implementations (sliding window, streaming, H2O)
"""

__version__ = "0.0.1"
