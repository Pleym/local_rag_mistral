from langchain_community.document_loaders import PyPDFLoader
import sys
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import argparse
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

PATH = "/Users/dorian/python/ML_rag/"

def load_pdf(pdf_name: str):
    try:
        document = PyPDFLoader(pdf_name)
        return document.load()
    except Exception as e:
        print("Cannot found the file.")
        return None


# Indexing
def indexing_document(document):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 200,
            length_function = len,
            is_separator_regex = False,
            )
    split_document = text_splitter.split_documents(document)
    return split_document 

#Now for search over them at runtime we need to embed them into a vector store
def create_embeddings():
    return OllamaEmbeddings(model="mistral")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help="Enter the name of your file")
    args = parser.parse_args()
    
    # Charger le document
    doc = load_pdf(args.filename)
    if doc is None:
        return
    
    # Créer les splitters pour les parents et enfants documents
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    
    # Initialiser le store pour les documents parents
    store = InMemoryStore()
    
    # Initialiser l'embedding
    embeddings = create_embeddings()
    
    # Créer le vectorstore
    vectorstore = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialiser le retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Ajouter les documents au retriever
    retriever.add_documents(doc)
    
    # Effectuer la recherche
    query = "De quelle forme sont les communications collectives?"
    retrieved_docs = retriever.get_relevant_documents(query)
    
    print("\nRésultats de la recherche:")
    context = "\nContexte extrait du document:\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n------- Résultat {i} -------")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print("Contenu:")
        print(doc.page_content.strip())
        print("-" * 50)
        
        # Ajouter au contexte
        context += f"Extrait {i} (Page {doc.metadata.get('page', 'N/A')}):\n"
        context += f"{doc.page_content.strip()}\n\n"
    # Retrival and generation

    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = RetrievalQA.from_chain_type(
        llm = Ollama(model="Mistral"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("\nContexte complet pour le LLM:")
    print(context)
    result = rag_chain.invoke({"query": query})
    print(result["result"])

if __name__ == "__main__":
    main()