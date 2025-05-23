import os
import base64
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json
from dotenv import load_dotenv

load_dotenv(override=True)


def load_pdf(file_path: str) -> List[Dict]:
    """Load and extract text from a PDF file, returning a list of page contents."""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            base64_text = convert_to_base64(text)
            pages.append(
                {
                    "content": text,
                    "metadata": {
                        "page_number": page_num + 1,
                        "base64_content": base64_text,
                    },
                }
            )
    return pages


def convert_to_base64(text: str) -> str:
    """Convert text to base64 string."""
    return base64.b64encode(text.encode()).decode()


def process_pdfs(input_dir: str) -> List[Dict]:
    """Process all PDFs in the input directory and return processed pages."""
    documents = []

    # Process each PDF file
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")

            # Extract pages from PDF
            pages = load_pdf(file_path)

            # Add source filename to metadata for each page
            for page in pages:
                page["metadata"]["source"] = filename
                documents.append(page)

    return documents


def create_vector_store(documents: List[Dict], output_dir: str):
    """Create and save FAISS vector store."""
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create vector store
    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    vector_store = FAISS.from_texts(
        texts=texts, embedding=embeddings, metadatas=metadatas
    )

    # Save vector store
    vector_store.save_local(output_dir)
    print(f"Vector store saved to {output_dir}")


def main():
    # Create necessary directories if they don't exist
    os.makedirs("raw_files", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Process PDFs and create vector store
    documents = process_pdfs("raw_files")
    create_vector_store(documents, "vector_store")


if __name__ == "__main__":
    main()
