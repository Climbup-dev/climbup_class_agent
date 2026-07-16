import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector
from app.core.config import settings

def get_embeddings_model():
    # Utilizing Gemini for embeddings given user's availability of Gemini
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
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
    
def retrieve_context(query: str, classroom_id: int, top_k: int = 3) -> str:
    vector_store = get_vector_store()
    # Filter search by classroom_id
    results = vector_store.similarity_search(
        query, 
        k=top_k, 
        filter={"classroom_id": classroom_id}
    )
    
    context = "\n\n".join([doc.page_content for doc in results])
    return context
