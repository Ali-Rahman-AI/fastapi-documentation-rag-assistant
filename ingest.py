"""
ingest.py

Loads FastAPI documentation from the local `data/` folder, splits it into
chunks, generates embeddings, and stores everything in a persistent
ChromaDB collection.

IMPORTANT: All documentation files are LOCAL files. We use `TextLoader`
for `.md` and `.txt`and `.py` files (not a web/URL loader). Using the wrong loader
for local files previously caused this error:

    Failed to load data\\dependencies.md: HTTP Error 403: Forbidden

That happened because an unsuitable loader treated local paths in a way
that triggered network-style behavior. `TextLoader` simply opens the file
from disk with `open(file_path, encoding="utf-8")`, so it never makes an
HTTP request and this error cannot happen again.


    python ingest.py
"""

import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config
from utils import get_logger

logger = get_logger(__name__)

# Maps a file extension to the LangChain loader that can read it.
# .md and .txt both use TextLoader because they are plain local text files.
# To support a new file type, just add one line here.
LOADER_MAP = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".py": TextLoader,
}

def load_documents(data_folder: str):
    """
    Recursively walk the data folder (including subdirectories) and load
    every supported LOCAL file into a list of LangChain Document objects.
    Unsupported or broken files are skipped with a warning instead of
    crashing the whole ingestion process.
    """
    data_path = Path(data_folder)

    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Data folder '{data_folder}' does not exist. "
            "Create it and add FastAPI documentation files (.md, .txt, .pdf , .py)."
        )

    documents = []
    loaded_files = 0
    failed_files = 0

    # rglob("*") scans the folder AND all subdirectories.
    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        loader_class = LOADER_MAP.get(extension)

        if loader_class is None:
            logger.warning("Skipping unsupported file: %s", file_path)
            continue

        try:
            # .md and .txt files are LOCAL text files -> TextLoader,

            if loader_class is TextLoader:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                loader = loader_class(str(file_path))

            loaded_docs = loader.load()

            # Attach useful metadata for later source citations.
            for doc in loaded_docs:
                relative_path = file_path.relative_to(data_path)
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_path"] = str(relative_path.as_posix())
                
            documents.extend(loaded_docs)
            loaded_files += 1
            logger.info("Loaded document: %s", file_path)
        except Exception as error:
            failed_files += 1
            logger.error("Failed to load %s: %s", file_path, error)

    logger.info("Loaded %d file(s), failed %d file(s)", loaded_files, failed_files)

    if loaded_files == 0:
        raise ValueError(
            f"No supported documents were found in '{data_folder}'. "
            "Add .md, .txt, or .pdf files and try again."
        )

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def split_documents(documents):
    """Split loaded documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add a chunk number per source file so citations can reference it.
    chunk_counters = {}
    for chunk in chunks:
        file_name = chunk.metadata.get("file_name", "unknown")
        chunk_counters[file_name] = chunk_counters.get(file_name, 0) + 1
        chunk.metadata["chunk_number"] = chunk_counters[file_name]

    logger.info(
        "Created %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    return chunks


def reset_vector_store(db_path: str):
    """
    Delete any existing ChromaDB folder before rebuilding.

    This is the simplest, easiest-to-explain way to avoid duplicate
    chunks: every time `python ingest.py` runs, it rebuilds the
    collection from scratch instead of appending to it.
    """
    path = Path(db_path)
    if path.exists():
        logger.info("Removing existing vector database at: %s", db_path)
        shutil.rmtree(path)


def build_vector_store(chunks):
    """Embed the chunks and store them in ChromaDB in batches."""

    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    logger.info(
        "Creating ChromaDB collection at: %s",
        config.CHROMA_DB_PATH,
    )

    vector_store = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_PATH,
    )

    batch_size = 500

    total_chunks = len(chunks)

    for start in range(0, total_chunks, batch_size):
        end = min(start + batch_size, total_chunks)

        batch = chunks[start:end]

        logger.info(
            "Adding chunks %d-%d of %d",
            start + 1,
            end,
            total_chunks,
        )

        vector_store.add_documents(batch)

    logger.info("Vector database created successfully.")

    return vector_store


def run_ingestion():
    """Full ingestion pipeline: reset -> load -> split -> embed -> store."""
    logger.info("Starting document ingestion pipeline...")

    reset_vector_store(config.CHROMA_DB_PATH)

    documents = load_documents(config.DATA_FOLDER)
    chunks = split_documents(documents)
    build_vector_store(chunks)

    logger.info("Ingestion complete. You can now run the app with: python app.py")


if __name__ == "__main__":
    run_ingestion()
