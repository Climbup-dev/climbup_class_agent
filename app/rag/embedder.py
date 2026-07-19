import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def get_embeddings_model():
    # Utilizing Gemini for embeddings given user's availability of Gemini
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=settings.GEMINI_API_KEY
    )

def get_vector_store(collection_name: str = "climbup_classrooms"):
    # Using Langchain's PGVector integration
    connection_string = settings.DATABASE_URL
    if connection_string.startswith("postgres://"):
        connection_string = connection_string.replace("postgres://", "postgresql+psycopg2://")
    elif connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+psycopg2://")
        
    embeddings = get_embeddings_model()
    return PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection_name,
        use_jsonb=True
    )

def process_and_store_documents(documents: list, classroom_id: int):
    if not documents:
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Inject classroom_id into every chunk's metadata for filtering during retrieval
    for chunk in chunks:
        chunk.metadata["classroom_id"] = classroom_id
        
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
def reformulate_query(query: str, chat_history: str = "") -> str:
    """Uses LLM to rewrite a vague student query into a powerful standalone search query."""
    if not chat_history.strip():
        return query
        
    prompt = f"""You are an AI assistant helping to optimize search queries. 
Given the chat history and the latest user question, reformulate the user question into a standalone, highly specific search query that can be used to search a vector database.
Do NOT answer the question. Just output the optimal search query string.

Chat History:
{chat_history}

Latest Question: {query}
Optimal Search Query:"""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        reformulated = response.content.strip()
        # Fallback if LLM output is too long or weird
        if len(reformulated) > 150 or not reformulated:
            return query
        return reformulated
    except Exception as e:
        print(f"Error reformulating query: {e}")
        return query

def rerank_documents(query: str, documents: list, top_k: int = 3) -> list:
    """Uses LLM-as-a-Judge to score and rerank the fetched vector chunks."""
    if not documents:
        return []
    if len(documents) <= top_k:
        return documents
        
    # We will ask the LLM to score each chunk from 0-10 on relevance.
    # To save tokens and time, we pass all chunks in one prompt.
    chunks_text = ""
    for i, doc in enumerate(documents):
        chunks_text += f"\n--- Chunk {i} ---\n{doc.page_content}\n" 
        
    prompt = f"""You are a relevance scoring engine.
Rate how relevant each chunk is to the user query on a scale of 0 to 10.
User Query: "{query}"

{chunks_text}

Respond STRICTLY in JSON format with a list of scores corresponding to each chunk index. Example: {{"scores": [9, 2, 0, 7, ...]}}"""
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        data = json.loads(content)
        scores = data.get("scores", [])
        
        # Zip documents with scores
        doc_scores = []
        for i, doc in enumerate(documents):
            score = scores[i] if i < len(scores) else 0
            doc_scores.append((score, doc))
            
        # Sort by score descending and take top_k
        doc_scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in doc_scores[:top_k]]
    except Exception as e:
        print(f"Error in reranking: {e}")
        # Fallback to original order
        return documents[:top_k]

def retrieve_context(query: str, classroom_id: int, chat_history: str = "", top_k: int = 3) -> str:
    vector_store = get_vector_store()
    
    # 1. Reformulate query
    standalone_query = reformulate_query(query, chat_history)
    print(f"Original Query: {query} | Reformulated: {standalone_query}")
    
    # 2. Fetch wider pool (k=10)
    fetch_k = min(top_k * 3, 10)
    results = vector_store.similarity_search(
        standalone_query, 
        k=fetch_k, 
        filter={"classroom_id": classroom_id}
    )
    
    # 3. Skip reranking (LLM can handle context window easily)
    best_results = results[:6]  # Just take top 6 to be safe, no data lost to small LLM judge
    
    # 4. Format context with citations
    context_parts = []
    for doc in best_results:
        page_num = doc.metadata.get("page_number", doc.metadata.get("slide_number", "Unknown"))
        context_parts.append(f"[Source: Page/Slide {page_num}]\n{doc.page_content}")
        
    context = "\n\n".join(context_parts)
    return context
