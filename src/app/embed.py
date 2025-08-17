from fastapi import File, HTTPException
from settings import settings
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.app import settings

TEMP_FOLDER = settings.TEMP_FOLDER


def allowed_file(file):
    ALLOWED_EXTENSIONS = {"pdf"}
    return "." in file and file.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_and_split_data(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks


def save_file(file: File):
    file_path = f"uploads/{secure_filename(file.filename)}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def embed(file):
    if file.filename == "" and not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")
    file_path = save_file(file)
    chunks = load_and_split_data(file_path)
    db = get_vector_db()
    return chunks
