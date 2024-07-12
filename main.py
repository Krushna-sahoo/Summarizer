from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain.schema import Document
import json
import os
import requests
import numpy as np 
from summarizer import determine_optimal_clusters,cluster_documents,summarize_documents,combine_summaries,formatted_summary

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
        vectors_array = np.array(vectors['vectors'])
        optimal_clusters_silhouette, _ = determine_optimal_clusters(vectors_array)
        num_clusters = optimal_clusters_silhouette
        sorted_indices = cluster_documents(vectors_array, num_clusters)
        
        # Assuming 'docs' can be generated or fetched as needed
        # Placeholder: List of dummy documents
        docs = [Document(page_content=f"Document content for index {i}") for i in range(len(vectors_array))]

        summaries = summarize_documents(docs, sorted_indices)
        combined_summary = combine_summaries(summaries)
        formatted_summary = formatted_summary(combined_summary)
        return {"message": formatted_summary}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized access - invalid API key")




