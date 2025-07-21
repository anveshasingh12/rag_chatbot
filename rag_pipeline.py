from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv
import os

load_dotenv()

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

