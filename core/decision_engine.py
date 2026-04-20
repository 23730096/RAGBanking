from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re


SUGGESTIONS = [
    "Mở tài khoản thanh toán cần giấy tờ gì?",
    "Biểu phí chuyển tiền của Vietcombank là bao nhiêu?",
    "Điều kiện vay tiêu dùng theo quy định hiện hành là gì?",
]


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    return re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)


def query_interpretability(question: str) -> Tuple[bool, str]:
    q = _normalize(question)
    if not q:
        return False, "Câu hỏi đang để trống."
    tokens = _tokenize(q)
    if len(tokens) == 0:
        return False, "Câu hỏi hiện chưa đủ rõ nghĩa để hệ thống xử lý."
    if len(q) < 3:
        return False, "Câu hỏi quá ngắn để hệ thống hiểu được ý định."
    # Chỉ chặn các input hoàn toàn vô nghĩa theo hình thức, không chặn theo từ cụ thể
    if len(tokens) == 1:
        tok = tokens[0]
        if tok.isdigit():
            return False, "Câu hỏi hiện chưa đủ rõ nghĩa để hệ thống xử lý."
        if len(tok) <= 2:
            return False, "Câu hỏi quá ngắn hoặc chưa thể hiện rõ nhu cầu tra cứu."
    return True, ""


def evidence_strength(retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not retrieved_docs:
        return {
            "has_evidence": False,
            "top_score": 0.0,
            "avg_score": 0.0,
            "keyword_hits": 0,
            "metadata_hits": 0,
            "reason": "Không truy xuất được tài liệu liên quan."
        }

    scores: List[float] = []
    keyword_hits = 0
    metadata_hits = 0

    banking_terms = ["tài khoản", "ngân hàng", "phí", "lãi suất", "vay", "thẻ", "giao dịch", "sms banking", "mobile banking", "internet banking"]

    for d in retrieved_docs:
        score = d.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

        text = _normalize(d.get("text") or "")
        if any(term in text for term in banking_terms):
            keyword_hits += 1

        metadata = d.get("metadata") or {}
        meta_blob = _normalize(" ".join(str(v) for v in metadata.values()))
        if any(term in meta_blob for term in banking_terms):
            metadata_hits += 1

    top_score = max(scores) if scores else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    has_evidence = (
        top_score >= 0.50
        or avg_score >= 0.35
        or keyword_hits >= 2
        or metadata_hits >= 1
    )

    return {
        "has_evidence": has_evidence,
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "keyword_hits": keyword_hits,
        "metadata_hits": metadata_hits,
        "reason": "" if has_evidence else "Bằng chứng truy xuất chưa đủ mạnh để trả lời đáng tin cậy."
    }


def query_specificity(question: str) -> Dict[str, Any]:
    q = _normalize(question)
    tokens = _tokenize(q)

    # Không dùng keyword domain để quyết định in/out scope.
    # Chỉ đánh giá mức độ cụ thể của câu hỏi.
    vague = (
        len(tokens) <= 2
        or (len(tokens) <= 4 and not any(t in q for t in ["là gì", "bao nhiêu", "như thế nào", "thế nào", "cần gì", "điều kiện", "thủ tục", "đăng ký", "mở", "cách", "hướng dẫn"]))
    )

    return {
        "is_vague": vague,
        "token_count": len(tokens),
    }


def decide(question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok, reason = query_interpretability(question)
    if not ok:
        return {
            "status": "needs_clarification",
            "answer": reason + " Bạn có thể hỏi rõ hơn theo một trong các gợi ý.",
            "suggestions": SUGGESTIONS,
            "reason": reason,
        }

    evidence = evidence_strength(retrieved_docs)
    specificity = query_specificity(question)

    # Nếu không có tín hiệu retrieval nào đáng kể, lúc đó mới coi là ngoài phạm vi
    # if evidence["top_score"] < 0.15 and evidence["keyword_hits"] == 0 and evidence["metadata_hits"] == 0:
    #     return {
    #         "status": "out_of_scope",
    #         "answer": "Câu hỏi này có vẻ nằm ngoài phạm vi tri thức nghiệp vụ ngân hàng mà hệ thống đang hỗ trợ.",
    #         "suggestions": SUGGESTIONS,
    #         "reason": "Không có tín hiệu truy xuất liên quan đáng kể từ kho tri thức.",
    #         "evidence": evidence,
    #         "specificity": specificity,
    #     }

    # Nếu có vẻ liên quan nhưng câu hỏi còn mơ hồ -> hỏi lại
    if specificity["is_vague"] and evidence["top_score"] < 0.60:
        return {
            "status": "needs_clarification",
            "answer": "Câu hỏi của bạn có vẻ liên quan đến nghiệp vụ ngân hàng nhưng vẫn còn hơi chung chung. Bạn có thể nói rõ hơn loại dịch vụ, sản phẩm hoặc tình huống cần tra cứu.",
            "suggestions": SUGGESTIONS,
            "reason": "Câu hỏi còn mơ hồ, cần làm rõ thêm trước khi trả lời.",
            "evidence": evidence,
            "specificity": specificity,
        }

    # if not evidence["has_evidence"]:
    #     return {
    #         "status": "insufficient_evidence",
    #         "answer": "Tôi chưa tìm thấy đủ bằng chứng trong dữ liệu để trả lời câu hỏi này một cách đáng tin cậy.",
    #         "suggestions": SUGGESTIONS,
    #         "reason": evidence["reason"],
    #         "evidence": evidence,
    #         "specificity": specificity,
    #     }

    return {
        "status": "answer",
        "answer": "",
        "suggestions": [],
        "reason": "Câu hỏi đủ rõ và có bằng chứng phù hợp.",
        "evidence": evidence,
        "specificity": specificity,
    }
