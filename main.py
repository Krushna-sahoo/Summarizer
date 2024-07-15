from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain.schema import Document
import os
import json
import requests
import numpy as np
from summarizer import determine_optimal_clusters, cluster_documents, summarize_documents, combine_summaries, formatted_summary


load_dotenv()
process_key = os.getenv("api_key")

app = FastAPI()

async def get_vectors(namespaces: str):
    try:
        res = requests.get(f'https://embedder-routes-lawson.onrender.com/vectors/{namespaces}?api_key={process_key}')
        res.raise_for_status()
        response_data = res.json()
        return response_data['message']['vectors']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching vectors: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vectors")

@app.get("/")
async def check():
    return {"message": "server active"}

@app.get("/summarize/{namespaces}")
async def get_vectors_route(namespaces: str, api_key: str):
    if process_key == api_key:
        vectors = await get_vectors(namespaces) 

        if vectors:
            try:
                # Convert vectors to numpy array
                vectors_array = np.array([vec['vector'] for vec in vectors])
                optimal_clusters_silhouette, _ = determine_optimal_clusters(vectors_array)
                num_clusters = optimal_clusters_silhouette
                sorted_indices = cluster_documents(vectors_array, num_clusters)

                # Extracting document contents from JSON structure
                docs = [Document(page_content=vec['text']) for vec in vectors]

                summaries = summarize_documents(docs, sorted_indices)
                combined_summary = combine_summaries(summaries)
                final_summary = formatted_summary(combined_summary)
                return {"message": final_summary}
            except Exception as e:
                print(f"Error during summarization: {e}")
                raise HTTPException(status_code=500, detail="Failed to summarize documents")
        else:
            raise HTTPException(status_code=400, detail="Vectors list is empty")
    else:
        raise HTTPException(status_code=401, detail="Unauthorized access - invalid API key")