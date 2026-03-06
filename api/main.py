from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Semantic Cache API"}