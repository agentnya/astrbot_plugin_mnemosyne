# -*- coding: utf-8 -*-
"""
Mnemosyne Plugin constant definition
"""

# --- Milvus Related constants ---
DEFAULT_COLLECTION_NAME = "mnemosyne_default"  # Modified the default name for better recognition
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_MILVUS_TIMEOUT = 5  # Milvus Query timeout（Seconds）
VECTOR_FIELD_NAME = "embedding"
PRIMARY_FIELD_NAME = "memory_id"
DEFAULT_OUTPUT_FIELDS = [
    "content",
    "create_time",
    PRIMARY_FIELD_NAME,
]  # Default query return fields
# Maximum number of query memory entries
MAX_TOTAL_FETCH_RECORDS = 10000

# --- Conversation context related constants ---
DEFAULT_MAX_TURNS = 10  # Maximum number of short-term memory conversation rounds（Used for summarization）
DEFAULT_MAX_HISTORY = 20  # Maximum number of short-term memory historical messages（Used for summarization）

# --- RAG Related constants ---
DEFAULT_TOP_K = 5  # Default number of retrieved memories
DEFAULT_PERSONA_ON_NONE = "UNKNOWN_PERSONA"  # When personalityIDPlaceholder or default value used when empty

# --- Timer related ---
DEFAULT_SUMMARY_CHECK_INTERVAL_SECONDS = 60  # Default summary check interval Seconds
DEFAULT_SUMMARY_TIME_THRESHOLD_SECONDS = 3600  # Default time threshold
