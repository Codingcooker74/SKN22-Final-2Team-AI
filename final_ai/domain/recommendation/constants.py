STRICT_SUBCATEGORIES = {
    "주식캔",
    "주식파우치",
    "간식캔",
    "간식파우치",
    "습식사료",
    "캔/파우치",
    "동결건조/에어드라이",
    "동결/건조간식",
    "화식",
    "소프트사료",
}

RECOMMENDATION_TOP_K = 5
MIN_RECOMMENDATION_RESULTS = 5
MAX_FILTER_RELAXATION_COUNT = 4

SAMPLE_BLACKLIST_WORDS = {"샘플", "맛보기", "체험팩"}

PURE_CAN_SUBS = {"주식캔", "간식캔"}
PURE_POUCH_SUBS = {"주식파우치", "간식파우치"}
MIXED_SUBS = {"캔/파우치"}

AGE_EXCLUDE_KEYWORDS = {
    "키튼": ["어덜트", "시니어", "노령"],
    "퍼피": ["어덜트", "시니어", "노령"],
    "어덜트": ["키튼", "퍼피"],
    "시니어": ["키튼", "퍼피"],
}

AGE_MANDATORY_KEYWORDS = {
    "키튼": ["키튼", "전연령"],
    "퍼피": ["퍼피", "전연령"],
}

FEED_CATEGORIES = ["사료", "습식관"]

CORE_ANIMAL_PLANTS = {"닭", "소", "양", "말", "굴", "게", "꿀", "오리", "연어", "참치", "돼지"}
ALLERGY_TERM_ALIASES = {
    "닭": {"닭", "닭고기", "계육", "chicken"},
    "소": {"소", "소고기", "beef"},
    "양": {"양", "양고기", "lamb"},
    "말": {"말", "말고기", "horse"},
    "굴": {"굴", "oyster"},
    "게": {"게", "crab"},
    "꿀": {"꿀", "honey"},
    "오리": {"오리", "duck"},
    "연어": {"연어", "salmon"},
    "참치": {"참치", "tuna"},
    "돼지": {"돼지", "돼지고기", "pork"},
}
ALLERGY_STOP_NOUNS = {"고기", "가루", "분말", "생물", "제품", "성분", "첨가물", "함유", "용", "포함"}
ALLERGY_SAFE_WORDS = {"말티즈", "소프트", "소화", "소형", "소형견", "소프", "말티", "소중형", "말랑"}

HEALTH_TRAIT_KEYWORDS = ["슬개골", "기관허탈", "눈물", "피부", "관절", "체중", "소화", "신장", "심장"]
