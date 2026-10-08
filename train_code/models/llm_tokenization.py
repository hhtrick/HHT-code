"""Raw-text tokenization shared by training, caching and inference."""


# Retain the filenames of existing raw-text caches. Unsuffixed caches for these
# models may contain older chat-wrapped inputs and must not be reused.
RAW_CACHE_MODELS = frozenset({
    "qwen3_4b_instruct_2507", "qwen3_4b_thinking_2507",
    "chemdfm_v1_5_8b",
})


def llm_cache_key(model_key):
    """Use the established raw cache name; Base cache names stay unchanged."""
    return f"{model_key}_raw" if model_key in RAW_CACHE_MODELS else model_key


def tokenize_llm_texts(tokenizer, texts):
    """Encode the supplied text directly, without dialogue or role wrapping.

    Keep native tokenizer special-token/scientific-token rules and padding;
    callers extract the last decoder layer and remove padding when caching.
    """
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
