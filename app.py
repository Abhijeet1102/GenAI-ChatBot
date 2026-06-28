from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone   

app = Flask(__name__)

load_dotenv()

# API Keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

#  NEW Pinecone init (v3)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings
embeddings = download_hugging_face_embeddings()

# Index name
index_name = "presi-bot"

#  Connect to index
index = pc.Index(index_name)

#  LangChain wrapper
docsearch = PineconeStore(
    index,
    embeddings,
    "text"
)

# Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM
chatModel = ChatOpenAI(model="gpt-4o")

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)