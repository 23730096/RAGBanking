from typing import List, Dict, Any

# ===== IMPORT HÀM CỦA BẠN =====
# sửa path cho đúng project
from core.decision_engine import decide


# ===== MOCK DATA =====
def mock_doc(text, score, keyword_hits=0, metadata_hits=0):
    return {
        "text": text,
        "score": score,
        "keyword_hits": keyword_hits,
        "metadata_hits": metadata_hits,
    }


# ===== TEST CASES =====
TEST_CASES = [
    {
        "name": "GOOD - SMS Banking",
        "question": "Tôi có thể đăng ký SMS Banking ở đâu?",
        "docs": [
            mock_doc("Khách hàng có thể đăng ký SMS Banking tại quầy giao dịch...", 0.82, 2, 1),
            mock_doc("Dịch vụ SMS Banking hỗ trợ thông báo biến động số dư...", 0.75, 1, 0),
        ],
    },
    {
        "name": "WEAK - có nhưng không rõ",
        "question": "Phí SMS Banking thế nào?",
        "docs": [
            mock_doc("Dịch vụ SMS Banking có nhiều tiện ích...", 0.45, 0, 0),
            mock_doc("Thông báo giao dịch qua SMS...", 0.42, 0, 0),
        ],
    },
    {
        "name": "NO EVIDENCE",
        "question": "SMS Banking có đăng ký online không?",
        "docs": [],
    },
    {
        "name": "OUT OF SCOPE",
        "question": "Thời tiết hôm nay thế nào?",
        "docs": [],
    },
    {
        "name": "LOW SCORE",
        "question": "Mở tài khoản cần gì?",
        "docs": [
            mock_doc("Một số thông tin chung về dịch vụ...", 0.2, 0, 0),
        ],
    },
]


# ===== RUN TEST =====
def run_tests():
    for case in TEST_CASES:
        print("\n" + "=" * 50)
        print(f"TEST: {case['name']}")
        print(f"Q: {case['question']}")
        print(f"Docs: {len(case['docs'])}")

        result = decide(case["question"], case["docs"])

        print("\n👉 RESULT:")
        for k, v in result.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    run_tests()