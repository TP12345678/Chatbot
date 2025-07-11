from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load existing ChromaDB
chroma_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

print("IDC Chatbot is ready. Type 'exit' to quit.")

while True:
    query = input("You: ")

    if query.lower().strip() == "exit":
        print("Bye")
        break

    # Embed query and search ChromaDB (with score)
    results_with_scores = chroma_db.similarity_search_with_score(query, k=1)

    if results_with_scores:
        result, score = results_with_scores[0]
        print(f"Similarity Score: {score:.2f}") 
        
        if score > 0.9:
            print("AskIDC: Sorry, I don’t know that yet.")
        else:
            print("AskIDC:", result.page_content)
    else:
        print("No answer found.")
