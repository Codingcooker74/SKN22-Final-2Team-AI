def build_general_query_prompt(pet_context: str, user_input: str) -> str:
    return (
        "다음 질문을 반려동물 정보를 반영해 검색에 최적화된 한 문장으로 재작성하세요.\n"
        f"펫 정보: {pet_context}\n"
        f"질문: {user_input}"
    )
