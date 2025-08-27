from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_all_pdfs(folder="docs"):
    """This function is used to simply load all the .pdf docs present inside a folder
    present in the current working directory"""
    all_pages = []
    pdf_files = Path(folder).glob("*.pdf")
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()      
        all_pages.extend(pages)
    return all_pages
