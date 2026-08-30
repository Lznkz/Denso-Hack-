# `seul_code`

Branch orphan tối thiểu để đưa C-MAPSS FD002–FD004 vào Smart Conv1D AE.

- Model/train lấy từ `origin/smart_AE:src/phase1`.
- Train data lấy từ `origin/bminh_modify_code:demo/data`.
- Loader mới ghép dataset, tạo engine ID riêng, tách `X/W`, chuẩn hóa bằng train stats.

## Chạy

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest tests.test_smoke
.venv/bin/python -m src.phase1.main --smoke
.venv/bin/python -m src.phase1.main
```

## Test checklist

| Bước | Lệnh | Pass khi |
|---|---|---|
| 1. Môi trường | `.venv/bin/pip check` | Không có dependency conflict |
| 2. Data + split | `.venv/bin/python -m unittest tests.test_smoke` | `OK`; 609 engine; ID không trùng; train/val không giao nhau |
| 3. Model shape | Cùng test bước 2 | `(batch,20,9)` trả đúng `x_hat`, `w_hat(3)`, `z(4)` |
| 4. Smoke train | `.venv/bin/python -m src.phase1.main --smoke` | Chạy 1 epoch; tạo `smart_ae_smoke.pt`; loss hữu hạn |
| 5. Full train | `.venv/bin/python -m src.phase1.main` | Early stopping hoặc đủ 50 epoch; tạo `smart_ae.pt` |

Data test/RUL chưa cần cho training. Thêm lại khi đánh giá RUL hoặc mô phỏng realtime.
