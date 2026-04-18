import os

def load_settings(settings):
    # APP
    if os.getenv("APP_ENV"):
        settings["app"]["env"] = os.getenv("APP_ENV")

    # EMBEDDING
    if os.getenv("EMBEDDING_MODEL"):
        settings["embedding"]["model"] = os.getenv("EMBEDDING_MODEL")

    if os.getenv("EMBEDDING_DEVICE"):
        settings["embedding"]["device"] = os.getenv("EMBEDDING_DEVICE")

    # VECTOR DB
    if os.getenv("VECTOR_DB_HOST"):
        settings["vector_database"]["host"] = os.getenv("VECTOR_DB_HOST")

    if os.getenv("VECTOR_DB_PORT"):
        settings["vector_database"]["port"] = int(os.getenv("VECTOR_DB_PORT"))

    if os.getenv("VECTOR_DB_COLLECTION"):
        settings["vector_database"]["collection_Name"] = os.getenv("VECTOR_DB_COLLECTION")

    # LLM
    if os.getenv("LLM_MODEL_NAME"):
        settings["llm"]["model_name"] = os.getenv("LLM_MODEL_NAME")

    if os.getenv("LLM_BASE_URL"):
        settings["llm"]["base_url"] = os.getenv("LLM_BASE_URL")

    if os.getenv("LLM_TEMPERATURE"):
        settings["llm"]["temperature"] = float(os.getenv("LLM_TEMPERATURE"))

    if os.getenv("LLM_MAX_TOKENS"):
        settings["llm"]["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))

    # RETRIEVAL
    if os.getenv("TOP_K"):
        settings["retrieval"]["top_k"] = int(os.getenv("TOP_K"))

    if os.getenv("SCORE_THRESHOLD"):
        settings["retrieval"]["score_threshold"] = float(os.getenv("SCORE_THRESHOLD"))

    return settings