from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TEMP_FOLDER: str = "/tmp"
    LLM_MODEL: str = "llama3.1"
    CHROMA_PATH: str = "/path/to/chroma"
    COLLECTION_NAME: str = "my_collection"
    TEXT_EMBEDDING_MODEL: str = "text-embedding-model"


settings = Settings()
