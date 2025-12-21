import re
import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("index")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_docs(kb_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in kb_dir.glob("**/*"):
        if path.suffix.lower() not in (".md", ".txt"):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        # Заголовок: если первая строка "# ..."
        title = path.stem
        first_line = text.splitlines()[0] if text else ""
        m = re.match(r"^\s*#\s+(.+)\s*$", first_line)
        if m:
            title = m.group(1).strip()

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.relative_to(kb_dir)).replace("\\", "/"),
                    "title": title,
                },
            )
        )
    return docs


def build_index():
    if not KB_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка {KB_DIR}/. Сначала выполни Задание 2 и создай knowledge_base/."
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_docs(KB_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: List[Document] = []
    for d in docs:
        parts = splitter.split_text(d.page_content)
        for i, part in enumerate(parts):
            chunks.append(
                Document(
                    page_content=part,
                    metadata={**d.metadata, "chunk_id": i},
                )
            )

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    t0 = time.time()
    vs = FAISS.from_documents(chunks, embedding=embed)
    vs.save_local(str(INDEX_DIR))
    elapsed = time.time() - t0

    print("✅ Index built")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Docs: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"Index artifacts: {INDEX_DIR}/faiss.index + {INDEX_DIR}/index.pkl")
    print(f"Time: {elapsed:.2f} sec")


if __name__ == "__main__":
    build_index()
