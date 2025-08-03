import os
import pathlib
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

# Check for the secret file first, then fall back to environment variables
if os.path.exists("/run/secrets/openai_api_key"):
    with open("/run/secrets/openai_api_key", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

CATALOG_FILE = os.getenv("CATALOG_FILE", "./data/catalog.txt")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# LangChain & FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Local utils
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "app"))
from catalog_utils import normalize_name, is_placeholder


def load_catalog() -> List[str]:
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f.readlines()]
    # Clean, normalize display names (keep original), filter placeholders
    clean: List[str] = []
    seen_norm = set()
    for name in raw:
        if not name:
            continue
        if is_placeholder(name):
            continue
        norm = normalize_name(name)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        clean.append(name.strip())
    if not clean:
        raise RuntimeError("No valid products found in catalog.txt")
    return clean


def build_docs(names: List[str]) -> List[Document]:
    docs: List[Document] = []
    for n in names:
        norm = normalize_name(n)
        # Content for embedding: emphasize the product name and plausible query variants.
        page_content = f"PRODUCT_NAME: {n}\nALSO_KNOWN_AS: {norm}\nTYPE: ingredient or product\n"
        docs.append(Document(page_content=page_content, metadata={"display_name": n, "normalized_name": norm}))
    return docs


def main():
    names = load_catalog()
    print(f"Loaded {len(names)} unique catalog items.")
    docs = build_docs(names)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(docs, embeddings)

    pathlib.Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
