from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, recommend, products

app = FastAPI(title="TailTalk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(recommend.router, prefix="/api/recommend", tags=["recommend"])
app.include_router(products.router, prefix="/api/products", tags=["products"])


@app.get("/health")
def health():
    return {"status": "ok"}
