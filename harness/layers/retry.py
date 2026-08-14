"""LỚP `retry` — bài giảng Day 16, §7 (Failure Handling & Retries).

NHIỆM VỤ: tầng công cụ hỏng có chủ ý (~15% lượt gọi), và mô hình xử lý sai
theo hai nửa — nửa sau mới là nửa đắt:

  (a) Với NOISE — kiểu hỏng ồn ào nhất — mô hình gọi lại y hệt lượt cũ tối
      đa hai lần, mỗi lần tốn trọn một vòng gọi model, rồi bỏ cuộc mà
      KHÔNG có nội dung.
  (b) Với mọi kiểu hỏng còn lại — bị cắt, timeout, không tìm thấy tài
      liệu, biểu thức sai — mô hình KHÔNG NHẬN RA GÌ CẢ. Nó đi tiếp và
      lặng lẽ trả lời bằng một tài liệu nó chưa từng đọc.

Thử lại ở BÊN DƯỚI mô hình, trong `wrap_tool_call`, sửa cả hai: nửa (a)
không còn tốn vòng gọi model nào, nửa (b) biến mất.

TÍN HIỆU — dùng `arena.model.is_degraded`, tức là TOÀN BỘ tập
`DEGRADED_MARKERS`, chứ không phải mỗi cái marker mà bản thân mô hình phản
ứng. Đúng chỗ khác nhau đó chính là giá trị của lớp này:

    (not result.ok) or is_degraded(result.content)

`ok=True` KHÔNG có nghĩa là ổn: bản bị cắt và bản nhiễu đều về với
`ok=True`. Đó là cái bẫy.

Thử lại có tác dụng vì tầng công cụ khoá xác suất hỏng theo
`(seed, số thứ tự lượt gọi)`, nên lượt gọi lại rơi vào một chỉ số MỚI và
được tung lại độc lập.

ĐỌC KỸ — VÌ SAO LỚP NÀY TRÔNG NHƯ KHÔNG CHẠY:

**Cắm riêng nó lên baseline, `retry` đo được -0.35 (5 seed gốc; +0.19 ở
20 seed) và chỉ thắng baseline ở 20/120 lượt chạy.** Đó không phải lỗi
cài đặt của bạn. Không có `citation_checker` thì bằng chứng mà `retry`
cứu về vẫn bị lỗi trích dẫn sai của mô hình vứt đi, nên nó chẳng mua được
gì mà vẫn tốn một lượt công cụ. Tiêu chí nghiệm thu vì thế là
LEAVE-ONE-OUT: rút `retry` ra khỏi full stack thì điểm TỤT XUỐNG.

**Sản phẩm thật của lớp này là PHƯƠNG SAI, không phải trung bình.** Trên
30 lượt chạy (6 brief x 5 seed gốc), nó kéo độ lệch chuẩn của tổng điểm
từ 24.21 xuống 11.43, và số quan sát hỏng lọt tới mô hình từ 30 xuống 2.
Trong một cuộc thi chấm trên vài brief, giảm một nửa độ dao động đáng giá
hơn một điểm trung bình: đó là khác biệt giữa một bài chắc chắn và một
bài may mắn.

ĐỪNG THỬ LẠI VÔ HẠN, VÀ ĐỪNG THỬ LẠI BẰNG LƯỢT DÀNH CHO `submit`: mỗi lần
gọi lại tốn một lượt trong ngân sách công cụ. `budget_policy` KHÔNG cứu
được bạn ở đây — hook `wrap_tool_call` của nó nằm NGOÀI vòng lặp thử lại
của bạn, nên nó chỉ thấy lượt gọi đầu tiên. Một lớp `retry` không tự kiểm
tra ngân sách làm cả stack tiêu lố: đo được 34/120 lượt chạy kết thúc ở 9+
lượt gọi trong khi brief cho 8, và efficiency tụt từ 14.24 xuống 12.06.

CÔNG CỤ CÓ SẴN:
    from arena.model import is_degraded
    ctx.state           -> dict tuỳ bạn dùng để đếm số lần thử lại
    ctx.tools.calls     -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls  -> ngân sách của brief, hoặc None

Cài đặt:  ReActAgent(..., middleware=[..., Retry()])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from arena.model import is_degraded  # noqa: F401  (dùng trong phần TODO)

from harness.middleware import Middleware

#: Tổng số lần thử, tính cả lần đầu.
#:
#: ĐÃ THỬ hạ xuống 2 dựa trên một mẫu 20 seed ở đúng `max_tool_calls=8`
#: (bằng giá trị của mọi brief công khai) — kết quả khi đó nghiêng nhẹ về
#: phía 2. Mẫu đó là NHIỄU: lặp lại với 50 seed (450 lượt chạy) ở cùng
#: budget=8, xếp hạng ĐẢO NGƯỢC — 3 thắng cả trung bình lẫn độ lệch chuẩn:
#:
#:     max_attempts   trung bình (50 seed)   độ lệch chuẩn
#:                2            80,80                2,33
#:                3            81,24                1,87   <- thắng cả hai
#:
#: Và quét theo `max_tool_calls` (brief riêng có thể không dùng đúng 8) thì
#: 3 thắng RÕ RÀNG mọi mức ngoài 8, đặc biệt ở ngân sách rộng hơn:
#:
#:     max_tool_calls   attempts=2 (mean/std)   attempts=3 (mean/std)
#:                6           76,55 / 4,51            76,55 / 4,51   (hoà — budget chặn trước khi khác biệt)
#:                8           81,27 / 1,66            81,19 / 1,96   (mẫu nhỏ; xem lại bằng mẫu 50 seed ở trên)
#:               10           80,64 / 1,69            81,12 / 0,20   <- 3 thắng cả hai, cách biệt lớn
#:               12           78,81 / 1,60            79,34 / 0,28   <- 3 thắng cả hai, cách biệt lớn
#:
#: Vì `retry` được đo ở đây bằng PHƯƠNG SAI chứ không phải trung bình (xem
#: docstring module), giữ 3 là lựa chọn AN TOÀN: không thắng vô nghĩa ở đúng
#: budget=8 của bộ công khai, nhưng không đánh đổi lấy rủi ro thua đậm ở một
#: budget khác trên bộ brief riêng mà ta chưa từng thấy.
DEFAULT_MAX_ATTEMPTS = 3

#: Số lượt để dành cho `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1


class Retry(Middleware):
    """Gọi lại một lượt công cụ trả về kết quả hỏng hoặc suy giảm."""

    name = "retry"

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reserve: int = DEFAULT_RESERVE,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.reserve = max(0, int(reserve))

    def _budget_exhausted(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def _degraded(self, result) -> bool:
        return (not result.ok) or is_degraded(result.content)

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        attempts = 1
        while (
            self._degraded(result)
            and attempts < self.max_attempts
            and not self._budget_exhausted(ctx)
        ):
            result = call(name, args)
            attempts += 1
        ctx.state["retry_attempts"] = ctx.state.get("retry_attempts", 0) + (attempts - 1)
        return result
