from fastapi import FastAPI, Query, UploadFile, File
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

from src.app import embed, query


app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/embed")
def create_embedding(file: UploadFile = File(...)):
    embedded = embed(file.file)
    if embedded is None:
        return {"message": "Embedding failed"}
    return {"message": "Embedding created"}


@app.post("/query")
async def create_query(
    query_str: str = Query(str, min_length=1, max_length=100)
):

    return StreamingResponse(query(query_str), media_type="text/plain")
