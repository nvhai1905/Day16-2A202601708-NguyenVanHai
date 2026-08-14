"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
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

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers._evidence import evidence_view, norm, supports
from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        view = evidence_view(ctx)
        changed = False
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue

            normalised = norm(text)
            doc_id = claim.get("doc_id")
            if (
                isinstance(doc_id, str)
                and view.cites_safely(doc_id)
                and supports(view.lines.get(doc_id, ()), normalised)
            ):
                continue  # trích dẫn đã đúng và tài liệu đã truy xuất

            source = self._find_source(view, normalised)
            if source:
                claim["doc_id"] = source
                changed = True
            # else: không tìm được nguồn thật -> để critic xử lý, đừng bịa doc_id.

        if changed:
            report["citations"] = sorted(
                {
                    c["doc_id"]
                    for c in claims
                    if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]
                }
            )
        return report

    def _find_source(self, view, normalised: str) -> str:
        """Nguồn thật của câu này, ưu tiên bằng chứng mạnh hơn.

        Hai bậc, và bậc thứ hai là phần thu thêm được: scorer coi một tài
        liệu là ĐÃ TRUY XUẤT nếu nó nằm trong kết quả của một truy vấn mà
        lượt chạy đã chạy — nó phát lại chính truy vấn đó — chứ không chỉ khi
        tài liệu được `fetch_doc` về. Đòi hỏi toàn văn cho MỌI lần gắn lại
        vì thế là tự bỏ đi những lần gắn lại mà scorer sẵn sàng chấm
        `SUPPORTED`. Bậc một vẫn đi trước vì toàn văn là bằng chứng chắc hơn
        khi hai tài liệu cùng chứa một dòng.
        """
        return view.source_for(normalised, view.fetched) or view.source_for(
            normalised, sorted(view.mentioned)
        )
