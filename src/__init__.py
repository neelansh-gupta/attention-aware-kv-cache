"""
Attention-Aware KV Cache Compression
Postman AI/ML Recruitment Task 3

    model_wrapper.py  -> loading, inspection, attention, compressed generation
    cache_manager.py  -> sliding, streaming, and H2O managers
    cache_utils.py    -> DynamicCache / legacy cache inspection
    evictions.py      -> eviction policy implementations
    position_utils.py -> absolute position ids after eviction
    evaluation.py     -> teacher-forced perplexity and NIH helpers
"""

__version__ = "1.0.0"
