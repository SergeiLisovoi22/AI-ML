# rag_bot.py
from __future__ import annotations

import os
import re
import textwrap
from typing import List, Tuple

from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embeddings (сначала пробуем новый пакет, чтобы убрать warning; если нет — fallback)
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

INDEX_DIR = os.getenv("INDEX_DIR", os.path.join("..", "index"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Для FAISS в LangChain score часто = L2 distance (меньше лучше)
MAX_DISTANCE_FOR_ANSWER = float(os.getenv("MAX_DISTANCE_FOR_ANSWER", "0.60"))

MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "900"))


SYSTEM_PROMPT = """Ты — помощник по базе знаний.
Ты отвечаешь ТОЛЬКО на основе предоставленного контекста.

Правила:
- Всегда отвечай на РУССКОМ языке.
- Если в контексте нет ответа — честно скажи: "Я не знаю."
- В конце всегда укажи Sources: (файлы, из которых взята информация).
- Дополнительно выведи краткие Steps (2–4 пункта) как описание подхода, без лишних подробностей.
"""

USER_PROMPT = """Вопрос пользователя (RU):
{question_ru}

Вопрос для поиска (EN):
{question_en}

Контекст:
{context}

Ответь на русском языке.
"""


# -----------------------------
# Helpers
# -----------------------------
def is_russian(text: str) -> bool:
    # грубая эвристика: есть кириллица
    return bool(re.search(r"[А-Яа-яЁё]", text))


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY не задан. "
            "Задай переменную окружения (не хардкодь ключ в коде)."
        )
    return OpenAI(api_key=api_key)


def translate_ru_to_en(text: str) -> str:
    """
    Переводим RU -> EN только если есть кириллица.
    Важно: просим сохранять вымышленные имена как есть (Synth Flux и т.п.)
    """
    if not is_russian(text):
        return text

    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate user question to English for semantic search. "
                    "Do NOT change fictional proper nouns (e.g., Synth Flux, Photon Blade, Void Core, Astra Monks). "
                    "Return ONLY the translated text."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    out = (resp.choices[0].message.content or "").strip()
    return out if out else text


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        INDEX_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(vs: FAISS, query: str, k: int) -> Tuple[List[Document], List[float]]:
    docs_scores = vs.similarity_search_with_score(query, k=k)
    docs = [ds[0] for ds in docs_scores]
    scores = [float(ds[1]) for ds in docs_scores]
    return docs, scores


def should_say_idk(scores: List[float]) -> bool:
    if not scores:
        return True
    return min(scores) > MAX_DISTANCE_FOR_ANSWER


def format_context(docs: List[Document]) -> Tuple[str, List[str]]:
    chunks = []
    sources = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        title = d.metadata.get("title", "")
        chunk_id = d.metadata.get("chunk_id", "")
        sources.append(src)
        snippet = d.page_content[:MAX_CHARS_PER_CHUNK]
        header = f"[source={src} | title={title} | chunk_id={chunk_id}]"
        chunks.append(header + "\n" + snippet)
    return "\n\n---\n\n".join(chunks), sorted(set(sources))


def make_few_shot_examples(vs: FAISS) -> str:
    """
    Few-shot примеры: берём 1–2 “настоящих” примера из базы через retrieval.
    """
    example_questions = [
        "What is Synth Flux?",
        "Explain Photon Blade and who typically uses it.",
    ]

    blocks = []
    for q in example_questions:
        docs, scores = retrieve(vs, q, k=3)
        if not docs or should_say_idk(scores):
            continue
        context, sources = format_context(docs)

        # короткий пример-ответ: 1–2 предложения из первого документа
        first = docs[0].page_content.strip().split(".")
        short_answer = ".".join(first[:2]).strip()
        if short_answer and not short_answer.endswith("."):
            short_answer += "."

        blocks.append(
            f"Q: {q}\n"
            f"A: {short_answer}\n"
            f"Sources: {', '.join(sources)}"
        )

    if not blocks:
        return ""

    return "Few-shot examples:\n" + "\n\n".join(blocks) + "\n"


def call_llm(question_ru: str, question_en: str, context: str, few_shot: str) -> str:
    client = get_client()

    user = USER_PROMPT.format(
        question_ru=question_ru,
        question_en=question_en,
        context=context,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if few_shot:
        messages.append({"role": "user", "content": few_shot})

    messages.append({"role": "user", "content": user})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def answer_question(vs: FAISS, question_ru: str, few_shot: str) -> str:
    question_en = translate_ru_to_en(question_ru)

    docs, scores = retrieve(vs, question_en, k=TOP_K)

    if not docs or should_say_idk(scores):
        return (
            "Я не знаю.\n\n"
            "Steps:\n"
            "1) Выполнил поиск по векторному индексу.\n"
            "2) Найденные фрагменты недостаточно релевантны, чтобы ответить безопасно.\n\n"
            "Sources:\n"
            "- (нет надёжных источников)"
        )

    context, sources = format_context(docs)
    ans = call_llm(question_ru=question_ru, question_en=question_en, context=context, few_shot=few_shot)

    # страховка: если модель не вывела Sources/Steps — добавим
    if "Sources" not in ans:
        ans += "\n\nSources: " + ", ".join(sources)
    return ans


def main():
    print("🧠 RAG Bot (type 'exit' to quit)\n")
    vs = load_vectorstore()
    few_shot = make_few_shot_examples(vs)

    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "выход"}:
            break

        try:
            ans = answer_question(vs, q, few_shot)
            print("\n" + textwrap.fill(ans, width=120, replace_whitespace=False) + "\n")
        except Exception as e:
            print("\nError:\n" + str(e) + "\n")


if __name__ == "__main__":
    main()
