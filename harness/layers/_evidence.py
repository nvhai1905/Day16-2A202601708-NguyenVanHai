"""Các phép thử bằng chứng dùng chung — MỘT định nghĩa, vì hai sẽ lệch nhau.

`critic` (§2) và `citation_checker` (§11) cùng phải trả lời đúng câu hỏi mà
bộ chấm đóng băng trả lời, và phải trả lời GIỐNG HỆT nó:

    câu này có phải là trích dẫn nguyên văn của MỘT DÒNG tài liệu không?

`arena.scorer._supports` chính là phép thử đó, và nó chạy trên văn bản đã
`_norm`: NFC, casefold, gộp khoảng trắng. Một phép thử `in` thô thì CHẶT HƠN,
và khoảng cách đó không phải giả định ở vòng chấm thật: một endpoint thật có
thể trả tiếng Việt ở dạng NFD, hoặc gộp một khoảng trắng đôi ngay bên trong
câu mà nó đang chép lại đúng từng chữ. Mọi claim kiểu đó là SUPPORTED dưới
mắt scorer nhưng vô hình với phép thử thô — nên một `critic` xây trên `in`
sẽ xoá đúng những claim sắp được tính điểm.

Chiều ngược lại còn đắt hơn: quan sát chứa NHIỀU thứ không phải dòng thân
tài liệu (tiêu đề trong kết quả search, khung JSON, đoạn vắt qua hai dòng).
Với scorer, mọi thứ đó là `HALLUCINATED` — mất trọn 15 điểm honesty trên MỌI
brief. `text in ctx.observed_text` giữ chúng lại; `supports` thì không.

Nhập thẳng từ `arena.scorer` là có chủ ý: đó là bộ chấm thật, đóng băng và
băm hash, nên nhập nó bảo đảm hai lớp và người chấm dùng CÙNG một định nghĩa
thay vì một bản sao chép tay sẽ trôi đi. `harness/agent.py` đã nhập
`_canonicalise_output` theo đúng cách này và vì đúng lý do này.

KHÔNG có gì ở đây VIẾT LẠI một claim: chuẩn hoá chỉ dùng để QUYẾT ĐỊNH, còn
thứ nằm lại trong report vẫn là từng byte của chính mô hình (README §8.2).
"""

from __future__ import annotations

import re

from arena.scorer import (
    MAX_CLAIM_CHARS,
    _norm as norm,
    _normalised_bodies as doc_lines,
    _supports as supports,
)

#: Mã tài liệu trong một quan sát. Kết quả `search` liệt kê `doc_id` của
#: từng hit, nên tập này xấp xỉ `_RunFacts.retrieved` mà scorer dựng lại
#: bằng cách phát lại truy vấn — và xấp xỉ về phía AN TOÀN: một lần search
#: bị cắt làm mất vài mã khỏi đây, nhưng scorer phát lại truy vấn nên vẫn
#: coi chúng là đã truy xuất. Thiếu thì mất một cơ hội gắn lại; thừa thì
#: bị chấm `UNRETRIEVED`.
_DOC_ID_RE = re.compile(r"doc-\d{4}")

__all__ = [
    "MAX_CLAIM_CHARS",
    "doc_lines",
    "evidence_view",
    "norm",
    "quotes_any_document",
    "supports",
]


class evidence_view:
    """Những gì lượt chạy CHỨNG MINH được là đã đọc, tính đúng một lần.

    `ctx.observed_text` nối lại toàn bộ quan sát ở MỖI lần gọi, và cả hai
    lớp đều hỏi nó một lần cho mỗi claim nhân mỗi tài liệu. Dựng sẵn ở đây
    biến việc đó thành một lần nối và hai lần quét corpus cho cả `after_agent`.
    """

    def __init__(self, ctx) -> None:
        observed = ctx.observed_text
        corpus = ctx.corpus
        self.lines = doc_lines(corpus) if corpus is not None else {}
        #: Tài liệu về NGUYÊN VĂN từ một lần fetch sạch — bằng chứng mạnh nhất.
        self.fetched = (
            [doc.doc_id for doc in corpus.docs if doc.body and doc.body in observed]
            if corpus is not None
            else []
        )
        #: Tài liệu chỉ mới lộ mã trong một kết quả search. Yếu hơn, nhưng
        #: scorer vẫn tính là đã truy xuất, nên vẫn gắn lại được.
        self.mentioned = frozenset(_DOC_ID_RE.findall(observed))

    def cites_safely(self, doc_id: str) -> bool:
        """Trích tài liệu này có tránh được `UNRETRIEVED` không?"""
        return doc_id in self.mentioned or doc_id in self.fetched

    def source_for(self, normalised_claim: str, doc_ids) -> str:
        """Mã tài liệu ĐẦU TIÊN trong `doc_ids` có chứa câu này như một dòng."""
        for doc_id in doc_ids:
            if supports(self.lines.get(doc_id, ()), normalised_claim):
                return doc_id
        return ""


def quotes_any_document(lines_by_doc: dict, normalised_claim: str) -> bool:
    """Có tài liệu NÀO trong kho chứa câu này như một dòng không?

    Đúng phép thử `HALLUCINATED` của scorer, viết ở dạng khẳng định: sai
    nghĩa là claim này sẽ bị chấm bịa đặt, và một claim bịa đặt duy nhất
    lấy đi trọn 15 điểm honesty trên MỌI brief.
    """
    return any(supports(lines, normalised_claim) for lines in lines_by_doc.values())
