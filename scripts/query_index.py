from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"


def build_embeddings_model():
    model_name = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def main():
    embeddings = build_embeddings_model()

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Пример запроса
    query = "Who is the leader of the Shellborne and what are his main traits?"
    print(f"Запрос: {query}\n")

    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"=== Результат {i} ===")
        print(f"source_path: {doc.metadata.get('source_path')}")
        print(f"chunk_id: {doc.metadata.get('chunk_id')}")
        print("--- content ---")
        print(doc.page_content[:500], "...")
        print("\n")

if __name__ == "__main__":
    main()
