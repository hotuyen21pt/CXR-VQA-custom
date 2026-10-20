# Dữ liệu phát hành thực sự chứa gì

Xem thêm: [PIPELINE.md](PIPELINE.md) — vết của một bản ghi qua toàn bộ pipeline ·
[EVALUATION.md](EVALUATION.md) — các độ đo giải thích từ đầu

Mọi con số dưới đây đều tái lập được bằng:

```bash
python -m vindr_cxr_vqa.build_dataset     # ghi ra data/processed/
python -m vindr_cxr_vqa.stats             # ghi ra outputs/dataset_report.md
```

`outputs/dataset_report.md` là bản do máy sinh ra của chính trang này: nó tính
lại mọi thống kê từ `data/processed/*.jsonl` và đánh dấu từng mục `match` hay
`MISMATCH` so với paper. **25 trên 30 mục khớp.** Trang này giải thích 5 mục
không khớp, và repo xử lý chúng ra sao.

---

## Những gì khớp

| Paper / dataset card công bố | Dựng lại được |
| --- | --- |
| 4.394 ảnh, 17.597 QA pair | khớp tuyệt đối |
| 4,0 QA pair mỗi ảnh | khớp tuyệt đối |
| 14 lớp bệnh lý | khớp tuyệt đối |
| 6 loại câu hỏi, cân bằng 16,4–17,0% | khớp (16,4–17,0%) |
| Tỉ lệ từng lớp, Cardiomegaly 28,0% → Atelectasis 0,1% | cả 14 lớp lệch ≤ 0,1 điểm % |
| Chia 3.735 / 659 / 300 ảnh | khớp tuyệt đối |
| ~8,3 box ground truth mỗi ảnh validation | 8,44 |
| Mọi `image_id` đều tìm được file PNG tương ứng | 4.394 / 4.394 |

Con số 5.471 box ground truth của Mục 3.2 ở đây ra **5.561**. Paper không công
bố seed chia tập nên không thể khớp tuyệt đối; chênh 1,6% là nhiễu do chia
ngẫu nhiên, và nó xác nhận số box được đếm là box **thô** của từng bác sĩ chứ
không phải box đồng thuận đã gộp.

> **Lưu ý về nguồn ảnh.** Repo HuggingFace `Dangindev/VinDR-CXR-VQA` chỉ có 3
> file — `.gitattributes`, `README.md`, `data_v1.json` — **không có tấm ảnh
> nào**. Ảnh phải lấy từ bản VinDr-CXR trên máy bạn. Đã kiểm chứng: toàn bộ
> 4.394 ảnh và 17.597 QA pair của HF đều map được, 0 mẫu rớt, 0 file PNG
> thiếu. Trong 18.000 ảnh local thì 13.606 ảnh **không được dùng** — paper chỉ
> chọn ảnh có ít nhất một tổn thương được khoanh.

---

## Sai lệch 1 — `gt_location` thường thuộc về một bệnh lý khác

**Đây là phát hiện nghiêm trọng nhất.** Mục 2.2 khẳng định việc đối chiếu chéo
toàn bộ 4.394 ảnh với VinDr-CXR "xác nhận không có sai khác nào […] đảm bảo
bảo toàn 100% chú thích của chuyên gia".

Dựng lại mối nối cho ra bức tranh khác:

| | QA pair | Tỉ lệ |
| --- | --- | --- |
| `gt_location` là box của **đúng** `gt_finding` | 7.043 | **40,0%** |
| `gt_location` là box của **bệnh lý khác** | 10.530 | **59,8%** |
| `gt_location` không khớp box nào | 24 | 0,1% |

Bản thân các box đều **thật** — 99,9% là chú thích VinDr có thật trên đúng ảnh
đó — nhưng cứ ba lần thì hai lần box bị ghép với **sai nhãn**. Các cặp nhầm
phổ biến nhất:

| `gt_finding` ghi là | box thực tế là | Số lượng |
| --- | --- | --- |
| Cardiomegaly | Aortic enlargement | 1.906 |
| Cardiomegaly | Pleural thickening | 398 |
| Other lesion | Aortic enlargement | 353 |
| Lung Opacity | Aortic enlargement | 352 |
| Pulmonary fibrosis | Pleural thickening | 351 |

Dạng lỗi này khớp với giả thuyết: pipeline sinh dữ liệu đã lấy danh sách bệnh
lý và danh sách box của một ảnh **một cách độc lập** thay vì nối chúng lại.

Ví dụ cụ thể — ảnh `9a5094b2563a1ef3ff50dc5c7ff71345`, cả ba QA pair đều mang
`<loc_691_1375_1653_1831>`:

```
R10  Cardiomegaly         691 1376 1654 1832   <- box được công bố, ở đây ĐÚNG
R8   Cardiomegaly         689 1313 1667 1763
R9   Cardiomegaly         693 1376 1657 1800
R9   Aortic enlargement  1052  715 1299  967
R9   Pleural effusion    1789 1729 1875 1992
```

Và ca **sai** — ảnh `051132a778e61a86eb147c7c6f564dfe`, `gt_finding` ghi
"Cardiomegaly" nhưng `gt_location = <loc_1264_743_1611_1019>` chính là box
**Aortic enlargement** của ảnh đó (IoU 0,99), trong khi IoU với cả ba box
Cardiomegaly thật đều bằng **0,00**.

### Vì sao chính các độ đo của paper không phát hiện được

Mục 3.2 tính một dự đoán là đúng khi nó chồng lên **bất kỳ** box ground truth
nào với IoU ≥ 0,3 — tức là **không phân biệt lớp bệnh**. Một hình chữ nhật vẽ
trên động mạch chủ nhưng gán nhãn "Cardiomegaly" vẫn chồng lên một tổn thương
có thật, nên vẫn được tính đúng. Sai lệch này **không tốn gì lúc đánh giá**;
nó tốn lúc **huấn luyện**, nơi nó dạy mô hình chỉ sai cơ quan ở 60% số mẫu
giám sát.

### Repo này xử lý thế nào

Mỗi bản ghi mang **cả hai**, nên không mất gì:

| Trường | Ý nghĩa |
| --- | --- |
| `gt_location`, `bbox_orig` | box được công bố, nguyên xi |
| `box_label_consistent` | box công bố có đúng là của `gt_finding` không |
| `gt_location_fixed`, `bbox_fixed_orig` | box đồng thuận **thực sự** thuộc `gt_finding` |
| `repair_method` | `published` / `best_iou` / `consensus` / `unavailable` |

`data.box_source` trong `configs/default.yaml` chọn dùng cái nào để huấn
luyện. Mặc định là `fixed`. Đặt `published` để tái lập đúng thiết lập của
paper.

Cách sửa: gộp hai hoặc ba lần đọc độc lập của các bác sĩ cho cùng một tổn
thương (phân cụm tham lam theo IoU ở ngưỡng 0,3, rồi lấy trung bình) và chọn
box đồng thuận của đúng lớp — ưu tiên box chồng lên box công bố nếu có, nếu
không thì lấy box được đồng thuận cao nhất của lớp đó.

---

## Sai lệch 2 — phần lập luận dài ~18 từ, không phải 150–250

Bảng 2 và dataset card đều ghi "đoạn lập luận lâm sàng chi tiết (150–250 từ)".
Trường `reason` thực tế:

| | Số từ |
| --- | --- |
| nhỏ nhất | 4 |
| trung vị | 20 |
| trung bình | 17,8 |
| lớn nhất | 40 |
| **số mẫu nằm trong khoảng 150–250 từ** | **0 / 17.597** |

Một giá trị điển hình chỉ là một câu:

> "Cardiomegaly is a nonspecific finding often indicative of chronic cardiac
> stress, such as chronic hypertension, valvular disease, or cardiomyopathy."

Và ngắn nhất thì suy biến thành `"Visible at specified location."`.

Trường này vẫn hữu ích làm giám sát cho tính giải thích được, chỉ là ngắn hơn
nhiều so với công bố. `data.include_reason` (mặc định `true`) nối nó vào chuỗi
đích; chính vì nó ngắn nên `data.max_seq_length` 1024 là thoải mái.

---

## Sai lệch 3 — không có mẫu âm tính nào

Phần tóm tắt của paper nêu "phân bố cân bằng 41,7% mẫu dương và 58,3% mẫu âm,
giúp giảm ảo giác ở các ca bình thường".

Trong `data_v1.json`:

- cả 17.597 QA pair đều có `gt_finding` **và** `gt_location` không rỗng;
- không mẫu nào có `gt_finding = "No finding"`;
- không câu trả lời nào bắt đầu bằng "No";
- chỉ 32 câu trả lời chứa từ phủ định ở bất kỳ đâu.

Cả 4.394 ảnh đều được chọn vì "chứa ít nhất một tổn thương được chuyên gia
khoanh" (Mục 2.1), nên không có tập ảnh bình thường nào để rút mẫu âm. File
phát hành **100% là mẫu dương**.

**Hệ quả.** Mô hình fine-tune trên dữ liệu này **không bao giờ thấy ví dụ về
việc từ chối báo một tổn thương** — đúng cái hành vi mà sự cân bằng kia định
tạo ra. Hãy chờ đợi đúng dạng recall cao / precision thấp mà paper báo (recall
0,926, precision 0,471 sau fine-tune). Muốn tạo mẫu âm thật thì phải kéo thêm
~10.600 ảnh VinDr không có tổn thương — nằm ngoài phạm vi repo này, nhưng
`vindr.load_images` đã sẵn sàng cho bạn truy cập chúng.

---

## Sai lệch 4 — độ phức tạp đa tổn thương

Bảng 3 ghi "54,8% [số ảnh validation] có ≥2 bệnh lý". Tính lại trên bản phát
hành, **92,7%** số ảnh validation mang box của hai lớp bệnh lý khác nhau trở
lên, và con số trên toàn bộ 4.394 ảnh là 93,0%.

Con số đi kèm trong cùng dòng đó — 8,3 box mỗi ảnh validation — tái lập sạch
ở mức 8,44, nên số lượng box là đúng và chỉ riêng tỉ lệ ≥2 là sai. Chúng tôi
không tìm được định nghĩa "bệnh lý" nào cho ra 54,8%.

Điều này ảnh hưởng tới cách đọc kết luận ở Mục 4: recall box thấp (0,090) được
paper quy cho "thiếu ví dụ huấn luyện đa tổn thương", nhưng ảnh huấn luyện dày
đặc tổn thương **y hệt** ảnh validation. Nguyên nhân khả dĩ hơn: mỗi ảnh chỉ
có ~4 QA pair nên nhiều nhất ~4 box được giám sát, trong khi lúc đánh giá có
~8,4 box ground truth — tức recall đã bị chặn trần ở khoảng 0,48 trước khi mô
hình phạm bất kỳ lỗi nào.

---

## Sai lệch 5 — tập huấn luyện lớn hơn cả dataset

Mục 3.1 ghi huấn luyện "3 epoch trên tập huấn luyện **20.880 mẫu**".

Toàn bộ bản phát hành chỉ có 17.597 QA pair. Lấy 85% làm tập huấn luyện thì
được **14.960** — đúng con số repo này dựng ra. 20.880 không đơn thuần là một
tỉ lệ chia khác; nó **vượt quá tổng số mẫu tồn tại**.

Có ba cách giải thích và paper không phân định: tập huấn luyện đã được tăng
cường (văn bản không nói gì), một phiên bản nội bộ lớn hơn đã được dùng rồi
chỉ phát hành một phần, hoặc con số đơn giản là sai. Điều này quan trọng vì nó
quyết định ngân sách bước: với batch hiệu dụng 16, 3 epoch là 3.915 bước theo
con số của paper nhưng chỉ 2.805 bước theo dữ liệu thực. `scripts/reproduce.py`
và notebook Kaggle dùng dữ liệu thực, nên `--max-steps -1` nghĩa là 2.805 bước.

---

## Các hệ toạ độ

`gt_location` nằm trong **hệ pixel DICOM gốc**, trong khi các file PNG ở
`dataset/train_data/train/` đã được resize sẵn về 224×224. Trộn hai hệ sẽ tạo
ra box sai vị trí mà **không có lỗi nào báo ra**, nên mỗi bản ghi mang sẵn cả
ba hệ một cách tường minh:

| Trường | Hệ toạ độ |
| --- | --- |
| `bbox_orig` | pixel DICOM gốc (chính là thứ `gt_location` mã hoá) |
| `bbox_resized` | 224×224, quy đổi qua `resize_factors.csv` |
| `bbox_norm` / `loc_norm` | số nguyên chuẩn hoá 0–999 |

Huấn luyện mặc định dùng hệ chuẩn hoá (`data.loc_format: norm1000`): nó độc
lập với độ phân giải, và độ rộng cố định 3 chữ số là thứ cho phép hàm mất mát
soft-IoU định vị được các token chữ số. `evaluate.py` quy đổi dự đoán ngược về
pixel gốc trước khi tính bất kỳ IoU nào.
