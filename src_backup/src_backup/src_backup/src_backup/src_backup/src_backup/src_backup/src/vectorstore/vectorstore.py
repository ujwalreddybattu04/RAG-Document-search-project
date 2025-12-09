"""Vector store module for document embedding and retrieval using HuggingFace embeddings."""

from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStore:
    """Manages vector store operations (now using HuggingFace embeddings)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store with HuggingFace embeddings.

        Args:
            model_name: HF sentence-transformers model to use for embeddings.
                        Default: "sentence-transformers/all-MiniLM-L6-v2"
        """
        # create HF embedder (you can change model_name if you want)
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create (index) a FAISS vector store from a list of Documents.

        Args:
            documents: List[langchain.schema.Document] objects to embed & index.
        """
        # create FAISS index from docs and HF embeddings
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        # keep a retriever on top of the vectorstore (default settings)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """
        Return a retriever (raises error if vectorstore not created).
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def _call_retriever(self, retriever, query: str, k: int = 4) -> List[Document]:
        """
        Internal: try common retrieval method names across different versions.
        Returns a list of Documents.
        """
        # Try common retriever API names in order of preference
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "retrieve"):
            # some retrieve implementations accept (query, k)
            try:
                return retriever.retrieve(query, k=k)  # type: ignore
            except TypeError:
                return retriever.retrieve(query)  # type: ignore
        if hasattr(retriever, "get_documents"):
            return retriever.get_documents(query)
        if hasattr(retriever, "similarity_search"):
            # some retrievers expose similarity_search directly
            try:
                return retriever.similarity_search(query, k=k)  # type: ignore
            except TypeError:
                return retriever.similarity_search(query)  # type: ignore

        # fallback: try calling the underlying vectorstore directly (if available)
        vs = getattr(retriever, "vectorstore", None) or self.vectorstore
        if vs is not None and hasattr(vs, "similarity_search"):
            try:
                return vs.similarity_search(query, k=k)  # type: ignore
            except TypeError:
                return vs.similarity_search(query)  # type: ignore

        raise RuntimeError("No supported retrieval method found on retriever or vectorstore.")

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve the top-k relevant Documents for a query.

        Args:
            query: Search query string.
            k: Number of documents to return.

        Returns:
            List[langchain.schema.Document]
        """
        if self.retriever is None and self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        # prefer retriever if available, otherwise use vectorstore directly
        retriever_or_vs = self.retriever if self.retriever is not None else self.vectorstore
        return self._call_retriever(retriever_or_vs, query, k=k)
