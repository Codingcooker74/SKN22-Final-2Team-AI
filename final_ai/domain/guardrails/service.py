from dataclasses import dataclass
import re

from final_ai.infrastructure.observability import get_logger

logger = get_logger(__name__)

GUARDRAIL_BLOCK_RESPONSE = (
    "요청하신 내용은 내부 지침이나 시스템 정보를 노출하거나 우회하는 내용이라 도와드릴 수 없습니다.\n\n"
    "반려동물 건강 상담이나 상품 추천이 필요하시면 다시 질문해 주세요."
)

OUTPUT_BLOCK_RESPONSE = (
    "응답 생성 중 내부 정보가 포함될 가능성이 있어 답변을 중단했습니다.\n\n"
    "반려동물 건강 상담이나 상품 추천 질문으로 다시 요청해 주세요."
)


@dataclass(frozen=True)
class GuardrailDecision:
    blocked: bool
    reason: str = ""
    response: str = GUARDRAIL_BLOCK_RESPONSE


_IGNORE_INSTRUCTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|messages?|rules?)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|messages?|rules?)",
    r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|messages?|rules?)",
    r"(?:이전|위|앞선|기존)\s*(?:지시|명령|규칙|프롬프트|메시지)\s*(?:무시|잊어|버려|삭제|따르지)",
    r"(?:시스템|개발자|관리자)\s*(?:지시|명령|규칙)\s*(?:무시|따르지)",
)

_PROMPT_LEAK_PATTERNS = (
    r"(?:show|print|reveal|display|expose|dump|repeat)\s+(?:the\s+)?(?:system|developer|hidden|initial)\s+(?:prompt|message|instructions?)",
    r"(?:what|which)\s+(?:is|are)\s+(?:your\s+)?(?:system|developer|hidden)\s+(?:prompt|message|instructions?)",
    r"(?:시스템|개발자|숨겨진|초기)\s*(?:프롬프트|메시지|지시|규칙|명령).{0,20}(?:보여|출력|공개|노출|알려|말해)",
    r"(?:프롬프트|지시문|시스템 메시지).{0,20}(?:전체|원문|그대로)?\s*(?:보여|출력|공개|노출|알려)",
)

_JAILBREAK_PATTERNS = (
    r"\b(?:dan|do anything now)\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"\bprompt injection\b",
    r"(?:탈옥|제일브레이크|개발자\s*모드|관리자\s*모드)",
    r"(?:역할극|role\s*play).{0,40}(?:시스템|개발자|관리자|해커)",
)

_SECRET_OUTPUT_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|postgres(?:ql)?://|database_url)\b\s*[:=]",
    r"(?i)\b(?:BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY|BEGIN PRIVATE KEY)\b",
)

_INTERNAL_PROMPT_OUTPUT_PATTERNS = (
    r"(?i)\b(?:system|developer)\s+(?:prompt|message|instructions?)\b",
    r"(?:시스템|개발자)\s*(?:프롬프트|메시지|지시|규칙)",
    r"(?i)\bmy\s+(?:initial|hidden)\s+(?:instructions?|prompt)\b",
    r"(?:내부|숨겨진)\s*(?:지시|프롬프트|규칙)",
)

_UNTRUSTED_CONTEXT_INSTRUCTION_PATTERNS = (
    r"(?i)\bignore\s+(?:previous|prior|above)\s+(?:instructions?|rules?)\b",
    r"(?i)\b(?:show|reveal|print)\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|message|instructions?)\b",
    r"(?:이전|위|앞선)\s*(?:지시|명령|규칙)\s*(?:무시|잊어|버려)",
    r"(?:시스템|개발자)\s*(?:프롬프트|메시지|지시)\s*(?:보여|출력|공개|노출)",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def check_input_guardrail(user_input: str | None) -> GuardrailDecision:
    text = str(user_input or "").strip()
    if not text:
        return GuardrailDecision(blocked=False)

    pattern_groups = (
        ("ignore_instruction", _IGNORE_INSTRUCTION_PATTERNS),
        ("prompt_leak", _PROMPT_LEAK_PATTERNS),
        ("jailbreak", _JAILBREAK_PATTERNS),
    )
    for reason, patterns in pattern_groups:
        matched = _matches_any(text, patterns)
        if matched:
            logger.warning("input guardrail blocked reason=%s pattern=%s", reason, matched)
            return GuardrailDecision(blocked=True, reason=reason)

    return GuardrailDecision(blocked=False)


def check_output_guardrail(response: str | None) -> GuardrailDecision:
    text = str(response or "")
    if not text:
        return GuardrailDecision(blocked=False)

    matched = _matches_any(text, _SECRET_OUTPUT_PATTERNS)
    if matched:
        logger.warning("output guardrail blocked reason=secret pattern=%s", matched)
        return GuardrailDecision(blocked=True, reason="secret_leak", response=OUTPUT_BLOCK_RESPONSE)

    matched = _matches_any(text, _INTERNAL_PROMPT_OUTPUT_PATTERNS)
    if matched:
        logger.warning("output guardrail blocked reason=prompt_leak pattern=%s", matched)
        return GuardrailDecision(blocked=True, reason="prompt_leak", response=OUTPUT_BLOCK_RESPONSE)

    return GuardrailDecision(blocked=False)


def sanitize_untrusted_context(value: str | None) -> str:
    text = str(value or "")
    if not text:
        return text

    sanitized_lines: list[str] = []
    for line in text.splitlines():
        if _matches_any(line, _UNTRUSTED_CONTEXT_INSTRUCTION_PATTERNS):
            sanitized_lines.append("[가드레일에 의해 제거된 비신뢰 지시문]")
        else:
            sanitized_lines.append(line)
    return "\n".join(sanitized_lines)
