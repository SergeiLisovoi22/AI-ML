from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vs = FAISS.load_local(
    "index",
    embeddings=embed,
    allow_dangerous_deserialization=True,  # OK для локального проекта
)

queries = [
    "What is Synth Flux and who can use it?",
    "Explain Photon Blade and its typical users.",
    "What is Void Core and why is it important?",
]

for q in queries:
    print("\n==============================")
    print("QUERY:", q)
    docs = vs.similarity_search(q, k=5)

    for i, d in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print("source:", d.metadata.get("source"))
        print("title:", d.metadata.get("title"))
        print("chunk_id:", d.metadata.get("chunk_id"))
        print(d.page_content[:400].replace("\n", " ") + "...")
