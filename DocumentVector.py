import os
import shutil
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai

class DocumentLoader:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_pages(self):
        # Load pages from the PDF file
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        return pages

class VectorDBBuilder:

    def __init__(self, documents):
        self.documents = documents

    def build(self):
        # Load OpenAI API key from environment variables
        load_dotenv()
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        # Initialize OpenAI embeddings and text splitter
        embedding = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        # Split documents into chunks for processing
        splits = text_splitter.split_documents(self.documents)
        persist_directory = 'docs/chroma/'
        # Uncomment the following block if you want to remove existing directory
        # try:
        #     shutil.rmtree('./' + persist_directory)
        # except:
        #     pass
        # Build vector database using Chroma
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        return vectordb

# def main():
#     # Path to the PDF file to be processed
#     pdf_path = "path/to/your/pdf_file.pdf"
#     # Load pages from the PDF file
#     document_loader = DocumentLoader(pdf_path)
#     pages = document_loader.load_pages()

#     # Build vector database from the loaded pages
#     vectordb_builder = VectorDBBuilder(pages)
#     vectordb = vectordb_builder.build()

#     # Use vectordb for further processing

# if __name__ == "__main__":
#     main()
