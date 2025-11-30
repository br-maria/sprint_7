import os
from pathlib import Path

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Инициализация OpenAI-клиента (новый SDK v1.x)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def build_embeddings_model():
    """
    Та же модель эмбеддингов, что при индексации: BAAI/bge-base-en-v1.5.
    Важно: модель должна совпадать.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},          # при наличии GPU: "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def load_vectorstore(embeddings):
    """
    Загружаем уже построенный Chroma-индекс из папки chroma_db.
    """
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


# ----- Few-shot примеры из той же вселенной -----

FEW_SHOT_EXAMPLES = [
    {
        "question": "Кто является лидером народа Shellborne?",
        "answer": (
            "Лидером народа Shellborne считается Lyron. "
            "Он описывается как стратег, который сочетает дисциплину shadow adept "
            "с уважением к традициям Hamara Lineage."
        ),
    },
    {
        "question": "Что такое Xeno-Serum и какую роль оно играет во вселенной?",
        "answer": (
            "Xeno-Serum — это нестабильный экспериментальный состав, "
            "который превращает обычных существ в Fluxborn. "
            "Во многих документах именно Xeno-Serum является источником "
            "изменений и конфликтов, поскольку его применяют и корпорации, "
            "и Umbral Order."
        ),
    }
]


# ----- System-промпт с указанием на продуманное рассуждение (CoT в голове) -----

SYSTEM_PROMPT = """
Ты внутренняя помощница команды, которая работает в вымышленной вселенной
(народ Shellborne, Umbral Order, Xeno-Serum, Oblivion Engine и т. д.).
Ты отвечаешь только на основе предоставленного контекста.

Твои правила:
1. Сначала мысленно анализируй фрагменты контекста, находи нужные факты и делай выводы по шагам.
2. НЕ придумывай деталей, которых нет в контексте.
3. Если информации недостаточно, честно отвечай: "Я не знаю".
4. В ответе для пользователя давай только итог и краткое объяснение, без подробного перечисления внутренних шагов.
5. Отвечай на русском языке, понятно и по делу.
""".strip()


def build_user_prompt(question: str, docs) -> str:
    """
    Собираем промпт:
    - контекст из найденных чанков
    - few-shot примеры (Q/A)
    - текущий вопрос пользователя
    """

    context_blocks = []
    for i, d in enumerate(docs):
        source = d.metadata.get("source_path", "unknown")
        chunk_id = d.metadata.get("chunk_id", i)
        block = f"[DOC {i} | source={source} | chunk_id={chunk_id}]\n{d.page_content}"
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    # few-shot в виде Q/A
    few_shot_parts = []
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    few_shots_str = "\n\n".join(few_shot_parts)

    user_prompt = f"""
Ты отвечаешь на вопросы по внутренней базе знаний вымышленной вселенной.
Используй только информацию из контекста ниже.

КОНТЕКСТ:
{context_str}

ПРИМЕРЫ (few-shot):
{few_shots_str}

Теперь ответь на вопрос пользователя строго по контексту.

Вопрос: {question}

Если ответа в контексте нет или он неоднозначен, напиши: "Я не знаю".
Ответ:
""".strip()

    return user_prompt


def answer_question(question: str, k: int = 4) -> str:
    """
    Основной RAG-пайплайн:
    - берём эмбеддинги
    - загружаем Chroma
    - ищем ближайшие чанки
    - строим промпт (контекст + few-shot)
    - спрашиваем LLM (OpenAI)
    """

    embeddings = build_embeddings_model()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})


    docs = retriever.invoke(question)

    # Если вообще ничего не нашли — сразу "Я не знаю"
    if not docs:
        return "Я не знаю"

    user_prompt = build_user_prompt(question, docs)

    #  вызов openai v1.x
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer


def repl():
    """
    Простой консольный интерфейс (REPL).
    """
    print("RAG-бот по вымышленной TMNT-вселенной (Shellborne и др.).")
    print("Введи вопрос (или 'exit' для выхода).")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Пока!")
            break

        answer = answer_question(q)
        print("\nОтвет бота:")
        print(answer)


if __name__ == "__main__":
    repl()
