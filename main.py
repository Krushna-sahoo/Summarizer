from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import json
import os
import requests

load_dotenv()

app = FastAPI()
process_key = os.getenv("api_key")


async def get_vectors(namespaces: str):
    try:
        res = requests.get(f'https://embedder-routes-lawson.onrender.com/vectors/{namespaces}?api_key={process_key}')

        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching vectors: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vectors")

@app.get("/")
async def check():
    return {"message": "server active"}

@app.get("/vectors/{namespaces}")
async def get_vectors_route(namespaces: str, api_key: str):
    print(process_key)
    print(api_key)
    if process_key == api_key:
        vectors = await get_vectors(namespaces)
        return {"message": vectors}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized access - invalid API key")



