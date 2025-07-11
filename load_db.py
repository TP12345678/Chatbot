#load the db when running the chatbot
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
