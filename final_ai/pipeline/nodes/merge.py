from decimal import Decimal

from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState


def _serialize_number(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return value


@traceable(name="merge_node", run_type="chain")
def merge_node(state: ChatState) -> dict:
    """
    domain_qa + recommend 결과 병합.
    어느 한 쪽만 있어도 정상 동작.
    product 테이블 컬럼명 기준으로 product_cards 구성.
    """
    domain_contexts  = state.get("domain_contexts")  or []
    reranked_results = state.get("reranked_results") or []

    if domain_contexts and reranked_results:
        mode = "combined"
    elif domain_contexts:
        mode = "domain_qa"
    elif reranked_results:
        mode = "recommend"
    else:
        mode = "empty"

    print(f"[MERGE] mode={mode} | contexts={len(domain_contexts)}, products={len(reranked_results)}")

    # product 테이블의 실제 컬럼명에 맞게 구성
    product_cards = [
        {
            "goods_id":      p.get("goods_id"),
            "product_name":  p.get("goods_name"),     # product 테이블 컬럼: goods_name
            "brand_name":    p.get("brand_name"),
            "price":         _serialize_number(p.get("price")),
            "discount_price":_serialize_number(p.get("discount_price")),
            "rating":        _serialize_number(p.get("rating")),
            "reviews":       _serialize_number(p.get("review_count")),
            "thumbnail_url": p.get("thumbnail_url"),
            "product_url":   p.get("product_url"),
        }
        for p in reranked_results
    ]

    return {"product_cards": product_cards}
