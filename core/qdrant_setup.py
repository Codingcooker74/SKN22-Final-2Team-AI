"""
Qdrant 컬렉션 초기화 스크립트

컬렉션:
  - products   : 상품 Hybrid Search (Dense + Sparse BM25 + RRF)
  - domain_qna : 반려동물 도메인 QnA RAG
  - breed_meta : 품종별 수의 영양학 메타데이터 RAG

실행:
  python -m core.qdrant_setup
  python -m core.qdrant_setup --recreate   # 기존 컬렉션 삭제 후 재생성
"""

import argparse
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY  = os.getenv("QDRANT_API_KEY", None)

# Dense 벡터 차원 — multilingual-e5-large: 1024, bge-m3: 1024
DENSE_DIM = 1024

COLLECTIONS = {
    "products": {
        "description": "상품 Hybrid Search",
        "dense_dim": DENSE_DIM,
    },
    "domain_qna": {
        "description": "반려동물 도메인 QnA RAG",
        "dense_dim": DENSE_DIM,
    },
    "breed_meta": {
        "description": "품종별 수의 영양학 메타데이터 RAG",
        "dense_dim": DENSE_DIM,
    },
}


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)


def create_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            )
        },
    )


def setup_collections(recreate: bool = False) -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    for name, cfg in COLLECTIONS.items():
        if name in existing:
            if recreate:
                print(f"  [{name}] 삭제 후 재생성...")
                client.delete_collection(name)
            else:
                print(f"  [{name}] 이미 존재 — 건너뜀 (--recreate 로 강제 재생성 가능)")
                continue

        create_collection(client, name, cfg["dense_dim"])
        print(f"  [{name}] 생성 완료 — {cfg['description']}")

    print("\n컬렉션 목록:")
    for c in client.get_collections().collections:
        info = client.get_collection(c.name)
        print(f"  {c.name}: dense={info.config.params.vectors['dense'].size}d, "
              f"points={info.points_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant 컬렉션 초기화")
    parser.add_argument("--recreate", action="store_true", help="기존 컬렉션 삭제 후 재생성")
    args = parser.parse_args()

    print(f"Qdrant 연결: {QDRANT_URL}")
    setup_collections(recreate=args.recreate)
