# Đánh giá hoạt động thế nào — giải thích từ đầu

Viết cho người **chưa từng làm computer vision**. Tài liệu giải thích: mô hình
được hỏi gì, nó trả lời ra sao, những con số nào được tính từ câu trả lời đó,
và đọc chúng thế nào. Mọi ví dụ đều là bản ghi thật trong dataset này.

*(English version: [EVALUATION.md](EVALUATION.md))*

---

## 1. Năm thuật ngữ, nói một lần

| Thuật ngữ | Ở đây nghĩa là |
| --- | --- |
| **Ảnh X-quang ngực** | Một ảnh xám. Có 4.394 ảnh. |
| **Tổn thương / bệnh lý** | Bất thường mà bác sĩ nhìn thấy, vd *Cardiomegaly* (tim to). Có 14 loại. |
| **Bounding box** | Hình chữ nhật khoanh vùng tổn thương, viết dạng `<loc_xmin_ymin_xmax_ymax>`. |
| **Ground truth** | Thứ bác sĩ thực sự vẽ và gán nhãn. Đây là **đáp án**. |
| **Prediction** | Thứ mô hình viết ra. Đây là **bài làm** cần chấm. |

Mô hình làm **hai việc cùng lúc**: nói *thấy gì* (văn bản) và nói *ở đâu* (hình
chữ nhật). Ba bảng kết quả chấm hai việc đó riêng biệt.

---

## 2. Mô hình được hỏi gì

Mỗi ảnh có 3–5 câu hỏi, thuộc sáu loại. Ví dụ thật lấy từ
`data/processed/val.jsonl`:

| Loại | Câu hỏi | Đáp án tham chiếu |
| --- | --- | --- |
| `Yes_No` | Is there Pulmonary fibrosis? | Yes, Pulmonary fibrosis `<loc_1024_681_1362_969>` |
| `Is_there` | Does this X-ray show Lung Opacity? | Yes, Lung Opacity is listed as a primary finding… `<loc_1309_816_1658_1138>` |
| `What` | What abnormality is visible? | Other lesion `<loc_1787_215_1972_257>` |
| `Where` | Where is the Pulmonary fibrosis? | …localized to the area defined by `<loc_1024_681_1362_969>` |
| `How_many` | How many findings are there? | Pulmonary fibrosis `<loc_1024_681_1362_969>` |
| `Which` | Which side shows the abnormality? | …localized to the patient's left side `<loc_1309_816_1658_1138>` |

### Điểm bất ngờ

**Các độ đo KHÔNG chấm xem câu Yes/No có trả lời "yes" hay "no" đúng không.**

Hai lý do. Thứ nhất, **dataset không có mẫu âm tính** — mọi câu hỏi trong bản
phát hành đều hỏi về một tổn thương thực sự có mặt, nên đáp án đúng của mọi câu
`Yes_No` và `Is_there` đều là "yes". Một mô hình cứ trả lời "yes" sẽ đạt 100%,
tức là đo được **con số không**.

Thứ hai, chính các đáp án tham chiếu cũng không tôn trọng loại câu hỏi của
mình. Nhìn dòng `How_many` ở trên: câu hỏi đòi một con số, đáp án lại nêu tên
một bệnh lý. Dòng `What` thì bắt đầu bằng "Yes,". Chấm khớp chuỗi với những
đáp án đó chỉ đo được sự thiếu nhất quán của pipeline sinh dữ liệu, không đo
được mô hình.

Vì vậy, việc đánh giá **bỏ qua loại câu hỏi** và hỏi hai điều trên mỗi **ảnh**:

1. **Mô hình nhắc tới những bệnh lý nào, ở bất kỳ đâu trong các câu trả lời?**
   → Bảng 4
2. **Nó đặt hình chữ nhật ở đâu?** → Bảng 5 và 6

---

## 3. Ba bước

```
bước 1  infer      mô hình + ảnh   ->  preds_val_base.jsonl
bước 2  evaluate   prediction      ->  eval_val_base.json   (ba bảng)
bước 3  results    hai file eval   ->  results_val.md       (đặt cạnh nhau)
```

### Bước 1 — sinh prediction

```bash
python -m vindr_cxr_vqa.infer --split val --out outputs/preds_val_base.jsonl
```

Đọc từng ảnh, hỏi từng câu, ghi lại mô hình nói gì:

```json
{"sample_id": "0005e8e3..._0", "image_id": "0005e8e3...",
 "question": "Where is the Lung Opacity?",
 "prediction": "The lung opacity is in the right upper zone <loc_706_332_816_411>."}
```

Chạy **hai lần**: một lần không có adapter (dòng **Pretrained** — mô hình gốc
vừa tải về) và một lần có `--adapter` (dòng **Fine-tuned** — sau khi huấn
luyện).

### Bước 2 — chấm điểm

```bash
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_base.jsonl --split val
```

### Bước 3 — đặt hai kết quả cạnh nhau

```bash
python -m vindr_cxr_vqa.results --base outputs/eval_val_base.json \
    --ft outputs/eval_val_ft.json --split val
```

---

## 4. Bảng 4 — gọi đúng tên bệnh lý không?

### Câu hỏi thực sự được đặt ra

Với một ảnh, so sánh **hai tập tên bệnh lý**:

- **Đáp án**: những gì bác sĩ đã khoanh trên ảnh đó.
- **Bài làm**: những tên bệnh lý xuất hiện trong câu trả lời của mô hình.

Ảnh thật `9a5094b2563a1ef3ff50dc5c7ff71345` có bốn tổn thương: *Aortic
enlargement, Cardiomegaly, Pleural effusion, Pleural thickening*.

Giả sử mô hình nhắc tới *Cardiomegaly, Pleural effusion, Pneumothorax*. Đặt 14
bệnh lý cạnh nhau:

| Bệnh lý | Có thật? | Mô hình nói? | Kết luận |
| --- | --- | --- | --- |
| Cardiomegaly | có | có | **TP** — true positive, tìm đúng |
| Pleural effusion | có | có | **TP** |
| Aortic enlargement | có | không | **FN** — false negative, **bỏ sót** |
| Pleural thickening | có | không | **FN** |
| Pneumothorax | không | có | **FP** — false positive, **bịa ra** |
| 9 cái còn lại | không | không | **TN** — true negative, im lặng đúng |

Mỗi ảnh đóng góp 14 kết luận như vậy. Cộng dồn trên cả 659 ảnh validation, rồi:

### Ba con số

**Precision (độ chính xác) — khi mô hình lên tiếng, bao nhiêu lần nó đúng?**

```
            TP              2
P  =  ───────────  =  ─────────  =  0,67
         TP + FP        2 + 1
```
Nêu ra 3 bệnh lý thì 2 cái có thật. Precision thấp = **mô hình bịa bệnh**.

**Recall (độ bao phủ) — trong những gì thực sự có, nó bắt được bao nhiêu?**

```
            TP              2
R  =  ───────────  =  ─────────  =  0,50
         TP + FN        2 + 2
```
Có 4 bệnh lý thật thì tìm được 2. Recall thấp = **mô hình bỏ sót bệnh**.

**F1 — một con số cân bằng cả hai.**

```
          2 · P · R        2 · 0,67 · 0,50
F1  =  ───────────────  =  ───────────────  =  0,57
           P + R             0,67 + 0,50
```

F1 là trung bình **điều hoà**, không phải trung bình cộng. Điều đó quan trọng:
nó trừng phạt sự lệch lạc. Mô hình có P = 1,00 và R = 0,02 thì trung bình cộng
là 0,51, nhưng F1 = 0,04 — và đó mới là đánh giá trung thực: một mô hình chỉ
tìm ra 2% bệnh thì vô dụng, dù khi nó lên tiếng có cẩn thận đến đâu.

**Đây là con số chính của paper**: F1 từ 0,558 lên 0,624 sau khi fine-tune.

### Vì sao Accuracy được in ra nhưng không đáng tin

```
             TP + TN            2 + 9
Acc  =  ─────────────────  =  ─────────  =  0,79
         TP + TN + FP + FN       14
```

79% nghe có vẻ tốt, nhưng nó bị thổi phồng bởi 9 **TN** — những bệnh lý không
có mặt và mô hình cũng không nhắc tới. Trung bình mỗi ảnh validation chỉ có
**3,59** trong số 14 bệnh lý, nên **một mô hình không nói gì cả vẫn đạt
accuracy 0,743**.

Accuracy thưởng cho sự im lặng; precision và recall thì không. Đó là lý do
**F1 mới là con số cần đọc**.

> **Lưu ý về Bảng 4 trong paper.** Paper in **cùng một giá trị** ở cột Accuracy
> và cột F1 (0,558 hai lần, rồi 0,624 hai lần). Đây là hai đại lượng khác nhau
> và nói chung không bằng nhau — hơn nữa F1 in ra đúng bằng `2PR/(P+R)` của
> precision và recall in ra. Nhiều khả năng cột Accuracy chỉ là bản sao của F1.
> Repo này báo cả hai: `f1` là con số để so sánh, `accuracy` là
> `(TP+TN)/tổng` thật.

### "Có nhắc tới Cardiomegaly không?" được quyết định thế nào

Tìm chuỗi văn bản, có **xử lý phủ định** (`evaluate.py:mentioned`). Mỗi bệnh lý
có nhiều cách diễn đạt — *Cardiomegaly* còn khớp với "enlarged heart", "cardiac
enlargement". Sau đó:

```
"There is cardiomegaly."              ->  TÍNH là có nhắc
"There is no pleural effusion."       ->  KHÔNG tính
"No pneumothorax. Cardiomegaly is present."
                                      ->  Cardiomegaly tính, Pneumothorax không
```

Phủ định chỉ có hiệu lực lùi về tới ranh giới mệnh đề gần nhất, nên ví dụ thứ
ba được xử lý đúng.

---

## 5. Bảng 5 — đặt box đúng chỗ không?

### IoU — ý tưởng duy nhất bạn cần hiểu

**Intersection over Union** đo độ chồng lấp của hai hình chữ nhật. Bằng diện
tích phần giao chia cho diện tích phần hợp:

```
   ┌─────────────┐                    phần giao (∩)
   │ prediction  │                IoU = ──────────────
   │      ┌──────┼───────┐              phần hợp (∪)
   │      │ ████ │       │
   └──────┼──────┘       │         0,0 = không chồng chút nào
          │ ground truth │         0,3 = lỏng nhưng dùng được trên lâm sàng
          └──────────────┘         0,5 = tốt
                                   1,0 = trùng khít từng pixel
```

Cụ thể: prediction 10×10 và ground truth 10×10 lệch nhau 5 pixel:

```
phần giao = 5 × 10          = 50
phần hợp  = 100 + 100 - 50  = 150
IoU       = 50 / 150        = 0,33
```

### Chấm các hình chữ nhật

Mỗi ảnh có các box ground truth (trung bình **8,4** cái — ba bác sĩ mỗi người
tự vẽ) và một số box do mô hình sinh ra.

1. Tính IoU cho **mọi cặp** prediction × ground truth.
2. Ghép **tham lam**: IoU cao nhất trước, mỗi box chỉ dùng một lần.
3. Cặp ghép có **IoU ≥ 0,3** là một **TP**.
4. Prediction thừa là **FP** (chỉ vào chỗ sai).
5. Ground truth thừa là **FN** (tổn thương bị bỏ sót hoàn toàn).

Rồi áp dụng đúng công thức precision / recall / F1 như Bảng 4.

**Vì sao phải ghép 1-1?** Nếu không, một hình chữ nhật may mắn có thể nhận công
của ba ground truth chồng lên nhau, và số FN trở nên vô nghĩa. Xem
`boxes.py:greedy_match`.

**Vì sao ngưỡng 0,3 chứ không phải 0,5?** Paper gọi 0,3 là "độ chồng lấp tối
thiểu có ý nghĩa lâm sàng" — đủ để chỉ cho bác sĩ đúng vùng cần nhìn, tức mục
đích lâm sàng, kể cả khi hình chữ nhật vẽ cẩu thả.

### mIoU — chỉ tính trên các TP

Là IoU trung bình của những cặp ghép **thành công**. Nó trả lời một câu hỏi
khác: *khi mô hình tìm ra được tổn thương, nó khoanh sát đến mức nào?*

Đây là lý do paper có thể báo recall 0,090 nhưng mIoU 0,615 — mô hình tìm được
ít tổn thương, nhưng cái nào tìm được thì khoanh khá sát. Hai thất bại khác
nhau, được nhìn thấy riêng.

---

## 6. Bảng 6 — kết quả có dùng được trên từng ảnh không?

Bảng 4 và 5 gộp tất cả thành một con số. Bảng 6 hỏi theo kiểu của bác sĩ:
**trên bao nhiêu phần trăm số ảnh, mô hình cho ra được ít nhất một hình chữ
nhật mà bác sĩ dùng được?**

| Độ đo | Ý nghĩa |
| --- | --- |
| `IoU≥0.5 (%)` | tỉ lệ ảnh có ít nhất một box "tốt" |
| `IoU≥0.3 (%)` | tỉ lệ ảnh có ít nhất một box "chấp nhận được" |
| `Mean IoU` | trung bình trên **mọi** box dự đoán, kể cả box sai |

Mô hình fine-tuned của paper: 22,8% tốt, 48,6% chấp nhận được. Tức khoảng một
nửa số ảnh nó chỉ vào chỗ có ích, và một phần năm thì thực sự chính xác.

Lưu ý `Mean IoU` ở đây (0,133) thấp hơn nhiều so với `mIoU(TP)` ở Bảng 5
(0,615), vì bảng này tính cả những lần trượt, còn bảng kia chỉ trung bình trên
những lần trúng.

---

## 7. Đọc kết quả in ra

```
=== preds_val_ft.jsonl  (split=val, space=norm1000, match=greedy) ===

Table 4 — VQA (text-based lesion classification)
  Accuracy 0.624   Precision 0.471   Recall 0.926   F1 0.624
                                     ↑ tìm được gần hết...
                                                  ↑ ...nhưng nửa số khẳng định là sai

Table 5 — Bounding-box detection (IoU>=0.3, gt=raw)
  Precision 0.120   Recall 0.090   F1 0.103   mIoU(TP) 0.615
                                              ↑ khi trúng thì khoanh sát
  4171 predicted vs 5561 ground-truth boxes

Table 6 — Localization quality (per image, n=659)
  iou_ge_0.5_pct         22.8     ← 1/5 số ảnh có box tốt
  iou_ge_0.3_pct         48.6     ← ~1/2 có box dùng được
  mean IoU (all preds)   0.133

Output format: 2140/2637 predictions contain a <loc_...> span (81.2%)
```

**Dòng cuối quan trọng nhất khi có sự cố.** Nếu nó ghi **0.0%**, mô hình không
sinh ra hình chữ nhật nào cả, và Bảng 5 với Bảng 6 là **vô nghĩa chứ không
phải kém** — chúng đang đo một lỗi định dạng, không phải khả năng định vị. Khi
đó bộ chấm in thêm cảnh báo và một prediction mẫu.

Mô hình zero-shot (chưa fine-tune) thường rơi vào 0% ở đây: nó chưa từng thấy
quy ước `<loc_…>`. **Học định dạng này chính là một phần việc mà fine-tuning
làm.**

---

## 8. Tự kiểm tra bộ chấm

Trước khi tin bất kỳ con số nào của mô hình, hãy xác nhận **cái cân có chuẩn
không**. Phát lại chính đáp án, giả vờ đó là bài làm của mô hình:

```bash
python -m vindr_cxr_vqa.evaluate --oracle --split val
```

Kết quả mong đợi, và cũng là kết quả repo này in ra thật:

```
Table 4   Precision 1.000      mọi bệnh lý được nêu đều thực sự có mặt
Table 6   iou_ge_0.5_pct 100.0 mọi box đều rơi trúng tổn thương thật
```

Nếu precision không phải 1.000 → phần so khớp văn bản hỏng. Nếu localization
không phải 100% → phần chuyển đổi toạ độ hỏng. **Không cần GPU**, chạy được
trên laptop bất kỳ.

Recall ở đó **không** phải 1,000 (khoảng 0,29), và điều đó là **đúng**: dataset
chỉ hỏi về ~4 trong số ~8,4 tổn thương mỗi ảnh, nên dù trả lời hoàn hảo vẫn còn
phần lớn box ground truth không được ghép.

Đây cũng là **trần** cho mô hình thật: 2.637 câu hỏi trên 5.561 box ground
truth → recall box **không thể vượt quá 0,474** với mức giám sát này. Nên con
số 0,090 của paper là khoảng **một phần năm** của mức khả thi, chứ không phải
một phần hai mươi.

---

## 9. Bốn cách đọc sai các con số này

1. **"Accuracy cao nghĩa là mô hình tốt."** Không — xem §4. Hãy đọc F1.
2. **"Các độ đo box có phân biệt lớp bệnh."** Không. Một hình chữ nhật vẽ trên
   động mạch chủ nhưng gán nhãn "Cardiomegaly" **vẫn tính là TP** nếu nó chồng
   lên bất kỳ box ground truth nào. Đây cũng là lý do lỗi lệch nhãn/box mô tả
   trong [DATA_NOTES.md](DATA_NOTES.md) **không tốn gì lúc đánh giá** nhưng
   **làm hỏng quá trình huấn luyện**.
3. **"Validation và test độc lập nhau."** Không — test là tập con 300 ảnh của
   659 ảnh validation, vì chỉ có cách hiểu đó thì các con số trong Bảng 3 của
   paper mới cộng lại khớp. Số liệu **validation** mới là số so sánh được với
   paper.
4. **"Recall box thấp nghĩa là mô hình định vị kém."** Một phần thôi. §8 cho
   thấy 0,474 là trần cấu trúc, nên 0,090 là khoảng một phần năm mức khả thi.

---

## 10. Mỗi con số được tính ở đâu

| Việc | File |
| --- | --- |
| Câu trả lời có nhắc bệnh lý X không | `evaluate.py:mentioned` |
| Xử lý phủ định | `evaluate.py:_negated` |
| Bảng 4 | `evaluate.py:eval_vqa` |
| IoU | `boxes.py:iou` |
| Ghép 1-1 | `boxes.py:greedy_match` |
| Bảng 5 và 6 | `evaluate.py:eval_boxes` |
| Chẩn đoán định dạng | `evaluate.py:diagnose_predictions` |
| Tự kiểm tra bộ chấm | `evaluate.py:oracle_predictions` |
| Dựng bảng so sánh | `results.py:render` |
