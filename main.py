from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import pymongo
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Allow all CORS (for frontend React access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class QueryRequest(BaseModel):
    question: str

# Load local LLM and Embeddings
pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sample knowledge base
docs = [
    "Client A has ₹85,00,000 in equity portfolio",
    "Client B has ₹73,00,000 in mutual funds",
    "Client C has ₹66,00,000 in bonds"
]
docsearch = FAISS.from_texts(docs, embedding_model)
docsearch.save_local("vector_index")
retriever = FAISS.load_local("vector_index", embedding_model, allow_dangerous_deserialization=True).as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["wealthdb"]
mongo_collection = mongo_db["clients"]

# Connect to MySQL
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Prc@2003#18",
    database="wealth_data"
)

# === API Routes ===

@app.get("/api/status")
def status():
    return {"status": "running"}

@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        result = rag_chain.run(req.question)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/top-portfolios")
def top_portfolios(n: int = 5):
    try:
        cursor = mysql_conn.cursor()
        cursor.execute("""
            SELECT client_id, SUM(value) AS total_value
            FROM transactions
            GROUP BY client_id
            ORDER BY total_value DESC
            LIMIT %s
        """, (n,))
        rows = cursor.fetchall()
        cursor.close()
        return {
            "top_portfolios": [{"client_id": row[0], "total_value": row[1]} for row in rows]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/rm-breakup")
def rm_breakup():
    try:
        cursor = mysql_conn.cursor()
        cursor.execute("""
            SELECT relationship_manager, SUM(value) AS total_value
            FROM transactions
            GROUP BY relationship_manager
        """)
        rows = cursor.fetchall()
        cursor.close()
        return {
            "rm_portfolios": [{"manager_id": row[0], "total": row[1]} for row in rows]
        }
    except Exception as e:
        return {"error": str(e)}
