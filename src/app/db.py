from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from src.app import settings


CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
TEXT_EMBEDDING_MODEL = settings.TEXT_EMBEDDING_MODEL


def get_vector_db():
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL)
    vector = Chroma(embedding_function=embedding,
                    persist_directory=CHROMA_PATH,
                    collection_name=COLLECTION_NAME)
    return vector
