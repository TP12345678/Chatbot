from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load your embedding model (same as used for indexing)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load your existing ChromaDB vector store (replace with your actual persist directory)
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Now you can query chroma_db with embedded vectors
