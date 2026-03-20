import os

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, Fusion, FusionQuery,
    MatchAny, MatchValue, Prefetch, SparseVector, Range,
)

from pipeline.state import ChatState

# ── 클라이언트 ──────────────────────────────────────────────────────────────────
llm       = OpenAI()
LLM_MODEL = "gpt-4o-mini"

QDRANT_URL = f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
qdrant     = QdrantClient(url=QDRANT_URL)

# ── 임베딩 모델 (lazy loading) ──────────────────────────────────────────────────
_dense_model  = None
_sparse_model = None


def get_models():
    global _dense_model, _sparse_model
    if _dense_model is None:
        from fastembed import TextEmbedding, SparseTextEmbedding
        print("Dense 모델 로드 중...")
        _dense_model  = TextEmbedding("intfloat/multilingual-e5-large")
        print("Sparse 모델 로드 중...")
        _sparse_model = SparseTextEmbedding("Qdrant/bm25")
        print("모델 로드 완료")
    return _dense_model, _sparse_model


# ── Hybrid Search ───────────────────────────────────────────────────────────────

def embed(query: str):
    dense_model, sparse_model = get_models()
    dv = list(dense_model.embed([f"query: {query}"]))[0].tolist()
    sv = list(sparse_model.embed([query]))[0]
    return dv, SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def hybrid_search(collection: str, query: str, top_k: int = 10, qdrant_filter=None):
    dv, sv = embed(query)
    return qdrant.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dv, using="dense", limit=top_k * 3, filter=qdrant_filter),
            Prefetch(query=sv, using="sparse", limit=top_k * 3, filter=qdrant_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points


# ── 공통 헬퍼 ───────────────────────────────────────────────────────────────────

DOMAIN_INTENT_TO_CATEGORY = {
    "health_disease":      "건강 및 질병",
    "care_management":     "사육 및 관리",
    "nutrition_diet":      "영양 및 식단",
    "behavior_psychology": "행동 및 심리",
    "travel":              "여행 및 이동",
}


def build_pet_context(state: ChatState) -> str:
    p = state.get("pet_profile") or {}
    parts = []
    if p.get("species"):
        parts.append(f"종: {'강아지' if p['species'] == 'dog' else '고양이'}")
    if p.get("breed"):   parts.append(f"품종: {p['breed']}")
    if p.get("age"):     parts.append(f"나이: {p['age']}")
    if state.get("health_concerns"):
        parts.append(f"건강관심사: {', '.join(state['health_concerns'])}")
    if state.get("allergies"):
        parts.append(f"알레르기: {', '.join(state['allergies'])}")
    return " / ".join(parts) if parts else "펫 프로필 없음"
