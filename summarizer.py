import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_together import ChatTogether
from langchain_together.embeddings import TogetherEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("TO_API_KEY")
pdf_path = os.environ.get("pdf_path")

llm = ChatTogether(
    api_key=api_key,
    model='mistralai/Mistral-7B-Instruct-v0.2'
)
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=api_key)

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text = "".join([page.page_content.replace('\t', ' ') for page in pages])
    return text

def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    return docs

def embed_documents(docs):
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    return np.array(vectors)

# Function to determine the optimal number of clusters using the Silhouette Score and Gap Statistic
def determine_optimal_clusters(vectors, max_clusters=10):
    silhouette_scores = []
    gap_statistics = []

    max_clusters = min(max_clusters, vectors.shape[0] - 1)
   # Calculate silhouette scores and gap statistics for different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=40).fit(vectors)
        cluster_labels = kmeans.labels_

        silhouette_avg = silhouette_score(vectors, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        reference_dis = np.random.random_sample(size=(vectors.shape[0], vectors.shape[1]))
        ref_kmeans = KMeans(n_clusters=n_clusters, random_state=40).fit(reference_dis)
        ref_inertia = ref_kmeans.inertia_
        gap_statistics.append(ref_inertia - kmeans.inertia_)

    optimal_clusters_silhouette = np.argmax(silhouette_scores) + 2
    optimal_clusters_gap = np.argmax(gap_statistics) + 2

    return optimal_clusters_silhouette, optimal_clusters_gap

def cluster_documents(vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=40).fit(vectors)
    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    sorted_indices = sorted(closest_indices)
    return sorted_indices

def summarize_documents(docs, indices):
    map_prompt = """
    You will be given a single passage of a judgement. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of the premise
    details. Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm,
                                     chain_type="stuff",
                                     prompt=map_prompt_template,
                                     verbose=True)

    summary_list = []
    for i in indices:
        chunk_summary = map_chain.run([docs[i]])
        summary_list.append(chunk_summary)
    summaries = "\n".join(summary_list)
    return Document(page_content=summaries)

def combine_summaries(summaries):
    combine_prompt = """
    You will be given a series of summaries from a judgement. The summaries will be enclosed in triple backticks (```)
    Your goal is to give a verbose summary of what happened in the story.
    The reader should be able to grasp the judgement in great detail.

    {text}

    VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm,
                                        chain_type='stuff',
                                        prompt=combine_prompt_template,
                                        verbose=True)

    return reduce_chain.run([summaries])

def format_summary(summary):
    formatted_summary_prompt = """
    You will be given a verbose summary of a judgment enclosed in triple backticks (```). Your task is to elaborate
    and expand this summary into a structured format with appropriate headings and subheadings. Your summary should
    be well-organized, clearly written, sufficiently detailed and long enough to provide a thorough understanding of
    the judgment. Use bullet points and long paragraphs to ensure clarity.

    {text}
    DETAILED FORMATTED SUMMARY:
    """
    formatted_summary_prompt_template = PromptTemplate(template=formatted_summary_prompt, input_variables=["text"])
    formatted_chain = load_summarize_chain(llm=llm,
                                           chain_type='stuff',
                                           prompt=formatted_summary_prompt_template,
                                           verbose=True)

    return formatted_chain.run([Document(page_content=summary)])

def process_pdf(file_path: str):
    text = load_pdf(file_path)
    docs = split_text(text)
    vectors = embed_documents(docs)
    optimal_clusters_silhouette, _ = determine_optimal_clusters(vectors)
    num_clusters = optimal_clusters_silhouette
    sorted_indices = cluster_documents(vectors, num_clusters)
    summaries = summarize_documents(docs, sorted_indices)
    combined_summary = combine_summaries(summaries)
    formatted_summary = format_summary(combined_summary)
    return formatted_summary

def main():
    file_path = pdf_path  # Replace with the actual PDF file path
    formatted_summary = process_pdf(file_path)
    print(formatted_summary)

if __name__ == "__main__":
    main()
