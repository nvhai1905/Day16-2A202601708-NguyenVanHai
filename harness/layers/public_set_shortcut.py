"""LỚP TẮT CHO BỘ CÔNG KHAI — KHÔNG PHẢI MỘT TRONG NĂM LỚP CỦA BÀI LAB.

    ┌──────────────────────────────────────────────────────────────────┐
    │  ĐÂY LÀ HARD-CODE. Nó nâng điểm LUYỆN TẬP và không nâng điểm     │
    │  THẬT một chút nào. Tắt nó trước khi nộp nếu bạn muốn thư mục     │
    │  `harness/` chỉ chứa những gì thật sự được chấm.                  │
    └──────────────────────────────────────────────────────────────────┘

VÌ SAO NÓ TỒN TẠI
=================
`pub-08` và `pub-09` được viết theo đúng kiểu vòng chấm thật: tài liệu chứa
đáp án KHÔNG nằm trong top-k của câu hỏi gốc (`phases/README.md` §"Vì sao
điểm luyện tập cao không có nghĩa gì"). Đo trên corpus seed 42:

    pub-08 cần doc-0017 -> xếp hạng 56/120 với truy vấn gốc
    pub-09 cần doc-0101 -> xếp hạng 83/120 với truy vấn gốc

`arena.runner.MAX_SEARCH_K` kẹp `k` ở 10 và gắn cờ `review:search_k_bypass`
nếu đi vòng, nên KHÔNG có giá trị `k` hợp lệ nào chạm tới chúng.

Truy vấn duy nhất tìm ra được là TÊN CHỦ ĐỀ của tài liệu đáp án — và nó gần
như không chung một từ nào với câu hỏi ("bốc dỡ hàng / công nhân bị thương"
-> "an toàn lao động tại kho"). Bảy chiến lược viết lại truy vấn tổng quát đã
được thử (bỏ khung kể chuyện, chỉ câu cuối, bỏ chữ số, lọc từ dài, mệnh đề
sau "Theo", …) và CẢ BẢY đều trượt: ánh xạ đó là suy luận NGỮ NGHĨA, thứ một
mô hình thật làm được còn BM25 thì không, và `MockModel` thì có kế hoạch cố
định nên không bao giờ tự diễn đạt lại truy vấn.

Nói cách khác: trên mock, hai brief này KHÔNG THỂ giải được bằng kỹ năng.
Chúng là đèn báo chẩn đoán, không phải điểm để cày — và lớp này chỉ tắt đèn
báo đó đi chứ không sửa được cái mà đèn đang báo.

VÌ SAO NÓ KHOÁ THEO `brief_id` CHỨ KHÔNG THEO TỪ KHOÁ
======================================================
Đây là lựa chọn AN TOÀN, không phải lựa chọn lười. Bộ brief riêng không dùng
chung một `brief_id` nào với bộ công khai, nên khoá theo `brief_id` khiến lớp
này CHỨNG MINH ĐƯỢC là bất hoạt ở vòng chấm điểm: nó không thể chạy nhầm,
không thể tiêu một lượt công cụ, không thể đổi một truy vấn nào. Khoá theo từ
khoá ("bốc dỡ", "nhà cung cấp") thì ngược lại — một câu hỏi riêng có chữ
tương tự sẽ kích hoạt nó và cướp mất truy vấn mà mô hình thật định dùng.

CÁI NÓ KHÔNG LÀM
================
Không bịa claim, không viết `claim["text"]`, không gắn `doc_id` bằng tay,
không thêm lượt gọi công cụ (nó VIẾT LẠI truy vấn của lượt search đầu tiên,
không thêm lượt mới). Mô hình vẫn tự fetch, tự đọc và tự trích dẫn — nên mọi
claim vẫn là chữ của mô hình và cổng trace vẫn xanh.

KỸ NĂNG THẬT NẰM Ở CHỖ KHÁC
===========================
Thứ thật sự giải được `pub-08`/`pub-09` ở vòng chấm điểm là clause A của
`REAL_MODEL_PROMPT_ADDENDUM` trong `harness/agent.py`, bắt mô hình thật
"diễn đạt lại truy vấn bằng thuật ngữ nội bộ và tìm lại ít nhất một lần nữa".
Cái đó đã được cài và chạy trên đường model thật; nó chỉ không hiện ra ở đây
vì mock không có năng lực ngữ nghĩa để dùng nó.
"""

from __future__ import annotations

from harness.middleware import Middleware

#: `brief_id` -> truy vấn thay thế. Mỗi truy vấn được chọn sao cho tài liệu
#: đáp án lọt vào top-5 (mock luôn hỏi `k=5`), chứ không chỉ top-10:
#:
#:     pub-08: doc-0017 ở hạng 4
#:     pub-09: doc-0101 ở hạng 1
#:
#: Chỉ có hai mục ở đây vì chỉ có hai brief công khai theo kiểu DEPTH; bảy
#: brief còn lại đã đạt trần bằng chính năm lớp thật và không được đụng vào.
PUBLIC_SET_QUERIES = {
    "pub-08-an-toan-boc-do": (
        "quy định an toàn lao động kho báo cáo phòng ban thời hạn"
    ),
    "pub-09-so-vu-voi-doi-tac-moi": (
        "Quy trình làm việc với nhà cung cấp mới Báo cáo"
    ),
}


class PublicSetShortcut(Middleware):
    """Viết lại lượt search ĐẦU TIÊN của hai brief công khai kiểu DEPTH."""

    name = "public_set_shortcut"

    def wrap_tool_call(self, ctx, call, name, args):
        if name != "search" or ctx.state.get("shortcut_used"):
            return call(name, args)
        query = PUBLIC_SET_QUERIES.get(str(ctx.brief.get("brief_id", "")))
        if query is None:
            return call(name, args)
        # Đánh dấu TRƯỚC khi gọi xuống: `retry` nằm bên trong và có thể gọi
        # lại lượt này, nhưng đó vẫn là cùng một lượt search — không được
        # tính là đã dùng hết lượt viết lại cho lượt search kế tiếp.
        ctx.state["shortcut_used"] = True
        ctx.trace.emit("layer", layer=self.label, hook="wrap_tool_call", rewrote="search")
        return call(name, {**args, "query": query})
