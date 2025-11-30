import os
from pathlib import Path
import time

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"


def load_documents():
    docs = []
    for path in KB_DIR.glob("**/*"):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
            # добавим метаданные: путь к файлу
            for d in loaded:
                d.metadata["source_path"] = str(path.relative_to(BASE_DIR))
            docs.extend(loaded)
    return docs


def split_documents(documents):
    # 1000 символов ≈ 180–220 слов (очень грубо)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # можно увеличить до 1500–2000
        chunk_overlap=200,    # небольшой overlap для контекста
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_embeddings_model():
    # Модель эмбеддингов BGE-base
    model_name = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # при наличии GPU: "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def main():
    start_time = time.time()
    print(f"Base dir: {BASE_DIR}")
    print(f"Knowledge base dir: {KB_DIR}")

    print("1) Загружаем документы из knowledge_base/ ...")
    docs = load_documents()
    print(f"   Загружено документов: {len(docs)}")

    print("2) Разбиваем на чанки ...")
    chunks = split_documents(docs)
    print(f"   Количество чанков: {len(chunks)}")

    # Добавим id чанка
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i

    print("3) Создаём модель эмбеддингов ...")
    embeddings = build_embeddings_model()

    print("4) Строим и сохраняем Chroma-индекс ...")
    CHROMA_DIR.mkdir(exist_ok=True, parents=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vectorstore.persist()

    elapsed = time.time() - start_time
    print("Готово!")
    print(f"Всего чанков в индексе: {len(chunks)}")
    print(f"Время генерации: {elapsed:.2f} сек")


if __name__ == "__main__":
    main()

