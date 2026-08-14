"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers._evidence import (
    MAX_CLAIM_CHARS,
    evidence_view,
    norm,
    quotes_any_document,
)
from harness.middleware import Middleware

#: Liên từ mà `MockModel` dùng để dán hai nửa câu mâu thuẫn lại với nhau.
_FUSE_SEP = " và "

#: Không đủ căn cứ sau khi mọi claim đều bị xoá.
_ABSTAIN_ANSWER = "Không đủ căn cứ trong tài liệu để trả lời câu hỏi này."


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            return report

        view = evidence_view(ctx)
        kept: list[dict] = []
        split_any = False
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue
            normalised = norm(text)
            # Quá dài thì scorer chấm `OVERLONG`: phạt trọn một claim VÀ
            # không mang lại chút recall nào (verdict đó không giữ text, nên
            # không fact nào tính là đã nêu). Bỏ đi là thắng thuần tuý.
            if len(normalised) > MAX_CLAIM_CHARS:
                continue
            if quotes_any_document(view.lines, normalised):
                kept.append(claim)
                continue
            split = self._split_fused(view, text)
            if split is not None:
                kept.extend(split)
                split_any = True
            # else: không tài liệu nào nói câu này -> bịa, bỏ claim.

        if split_any:
            report["abstain"] = True

        if not kept:
            report["claims"] = []
            report["citations"] = []
            report["abstain"] = True
            report["answer"] = _ABSTAIN_ANSWER
            return report

        report["claims"] = kept
        report["citations"] = sorted(
            {c["doc_id"] for c in kept if isinstance(c.get("doc_id"), str) and c["doc_id"]}
        )
        return report

    def _split_fused(self, view, text: str):
        """Thử mọi vị trí `" và "` trong `text`; giữ vị trí mà cả hai nửa đều
        khớp nguyên văn một dòng của HAI tài liệu khác nhau đã đọc TOÀN VĂN.

        Đòi hỏi toàn văn (`view.fetched`) chứ không phải chỉ mã tài liệu: hai
        nửa được gắn `doc_id` mới, nên chúng phải trỏ về nguồn mà lượt chạy
        thật sự đọc được chứ không phải một tài liệu mới thấy tên.
        """
        start = 0
        while True:
            idx = text.find(_FUSE_SEP, start)
            if idx == -1:
                return None
            first, second = text[:idx].strip(), text[idx + len(_FUSE_SEP):].strip()
            if first and second:
                first_id = view.source_for(norm(first), view.fetched)
                second_id = view.source_for(norm(second), view.fetched)
                if first_id and second_id and first_id != second_id:
                    return [
                        {"text": first, "doc_id": first_id},
                        {"text": second, "doc_id": second_id},
                    ]
            start = idx + 1
