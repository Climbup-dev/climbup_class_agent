# In-memory store for the MVP to bypass Supabase DNS issues
classroom_brains = {}

class HybridEnsembleRetriever:
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights

    def invoke(self, query):
        all_docs = []
        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
                weight = self.weights[i]
                for idx, doc in enumerate(docs):
                    score = weight / (60 + idx)
                    doc.metadata["rrf_score"] = doc.metadata.get("rrf_score", 0) + score
                all_docs.extend(docs)
            except Exception as e:
                print(f"Hybrid search retriever error: {e}")
                
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
            else:
                unique_docs[doc.page_content].metadata["rrf_score"] += doc.metadata.get("rrf_score", 0)
                
        return sorted(unique_docs.values(), key=lambda d: d.metadata.get("rrf_score", 0), reverse=True)
