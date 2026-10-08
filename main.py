from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Тохиргоо ачааллах
load_dotenv()
app = FastAPI(title="AI Business & HR Manager API")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Supabase Client үүсгэх
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 3. AI Моделуудыг тохируулах (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="business_knowledge",
    query_name="match_business_knowledge" 
)

# 4. Хүсэлтийн загварууд
class KnowledgeItem(BaseModel):
    content: str
    category: str = "general"

class ChatRequest(BaseModel):
    question: str

# 5. API Endpoints
@app.post("/api/add_knowledge")
def add_knowledge(item: KnowledgeItem):
    try:
        vector_store.add_texts(
            texts=[item.content],
            metadatas=[{"category": item.category}]
        )
        return {"status": "success", "message": "Мэдээлэл хадгалагдлаа."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_bot(request: ChatRequest):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        prompt = PromptTemplate.from_template(
            """Чи бол энэ байгууллагын ухаалаг туслах юм. 
            Доорх олдсон мэдээлэл дээр үндэслэн хэрэглэгчийн асуултад монгол хэлээр, эелдгээр хариул. 
            Хэрэглэгчийн асуулт: {question}
            Олдсон мэдээлэл: {context}
            Хариулт:"""
        )
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
