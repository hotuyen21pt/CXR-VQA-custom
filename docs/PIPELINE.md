# Một record đi qua toàn bộ pipeline

Tài liệu bám theo **đúng một bản ghi thật** từ lúc là dữ liệu thô cho tới lúc
thành một con số trong bảng kết quả. Mọi giá trị dưới đây đều được in ra từ
code của repo này, không phải ví dụ bịa.

Bản ghi được chọn: ảnh `9a5094b2563a1ef3ff50dc5c7ff71345`, câu hỏi đầu tiên.

> **Bản có hình:** [PIPELINE_VISUAL.pdf](PIPELINE_VISUAL.pdf) — cùng vết này
> nhưng có ảnh X-quang thật với box của bác sĩ vẽ đè lên đúng tỉ lệ, sơ đồ từng
> chặng, và phần "cuối cùng ra output gì". Dựng lại bằng
> `python scripts/make_pipeline_pdf.py`.

```
         data_v1.json (HuggingFace, CHỈ CÓ TEXT)
                    +                                 ┐
         dataset/annotations_train.csv                │ chặng 0
         dataset/train_data/train/*.png (ảnh của BẠN) ┘
                    │
                    ▼  chặng 1-2   build_dataset.py
         data/processed/val.jsonl  ── 1 dòng JSON
                    │
                    ▼  chặng 3     prompts.py
         prompt + chuỗi đích (target)
                    │
                    ▼  chặng 4     data.py
         token id + mask nhãn + vị trí 12 chữ số toạ độ
                    │
                    ▼  chặng 5     losses.py + train.py
         L = L_VQA + 0,5 · L_bbox   →  adapter LoRA
                    │
                    ▼  chặng 6     infer.py
         câu trả lời sinh ra (text có <loc_…>)
                    │
                    ▼  chặng 7     evaluate.py → results.py
         TP/FP/FN  →  Bảng 4, 5, 6
```

---

## Chặng 0 — Nguyên liệu: **hai nguồn tách rời**

### Nguồn A — HuggingFace (chỉ có chữ, **không có ảnh nào**)

Repo `Dangindev/VinDR-CXR-VQA` chứa đúng 3 file:

```
.gitattributes
README.md
data_v1.json      ← 9,3 MB, toàn bộ là text
```

Không có một tấm ảnh nào. Đây là điều **hay bị hiểu nhầm nhất**: tải dataset từ
HF về là chỉ có phần hỏi–đáp, chưa chạy được gì cả.

Bản ghi của ta trong `data_v1.json`:

```json
{
  "question": "Where is the Cardiomegaly?",
  "answer": "The Cardiomegaly finding refers to an enlargement of the cardiac
             silhouette, centrally located in the chest <loc_691_1375_1653_1831>.",
  "reason": "Cardiomegaly is a nonspecific finding often indicative of chronic
             cardiac stress, such as chronic hypertension, valvular disease...",
  "type": "Where",
  "difficulty": "Medium",
  "gt_finding": "Cardiomegaly",
  "gt_location": "<loc_691_1375_1653_1831>"
}
```

### Nguồn B — bản VinDr-CXR trên máy bạn (có ảnh)

`dataset/annotations_train.csv`, các dòng của **đúng ảnh này**:

```
R10  Cardiomegaly           691 1376 1654 1832
R8   Cardiomegaly           689 1313 1667 1763
R9   Cardiomegaly           693 1376 1657 1800
R9   Aortic enlargement    1052  715 1299  967
R9   Pleural thickening    1789 1729 1875 1992
R9   Pleural effusion      1789 1729 1875 1992
```

Ba bác sĩ (R8, R9, R10) mỗi người tự vẽ tim to của riêng mình — nên có **3 box
Cardiomegaly gần trùng nhau**, không phải 3 tổn thương khác nhau.

Và ảnh: `dataset/train_data/train/9a5094b2563a1ef3ff50dc5c7ff71345.png`

### Hai nguồn được nối bằng gì

Bằng `image_id` — chuỗi hash 32 ký tự, **không phải theo thứ tự dòng**. Cùng
một khoá đó xuất hiện ở cả 5 nơi: `data_v1.json`, tên file PNG,
`annotations_train.csv`, `image_labels_train.csv`, `resize_factors.csv`.

Nếu thiếu ảnh, `build_dataset.py` **dừng hẳn** với thông báo lỗi chứ không map
nhầm im lặng.

---

## Chặng 1 — Gộp box của ba bác sĩ

`vindr.fuse_boxes()` gom các box **cùng lớp** chồng lên nhau (IoU ≥ 0,3) thành
một cụm rồi lấy trung bình:

```
6 box thô  ->  4 box đồng thuận

  Cardiomegaly        [691 1355 1659 1798]   votes=3  rads=[R8, R9, R10]  ← gộp 3
  Aortic enlargement  [1052  715 1299  967]  votes=1  rads=[R9]
  Pleural thickening  [1789 1729 1875 1992]  votes=1  rads=[R9]
  Pleural effusion    [1789 1729 1875 1992]  votes=1  rads=[R9]
```

Đồng thời đọc kích thước ảnh từ `resize_factors.csv`:

```
gốc 2080 × 2336  ->  đã resize 224 × 224
fx = 0,107692    fy = 0,095890
```

> **Vì sao phải quan tâm?** Box trong `gt_location` ở **hệ toạ độ pixel gốc**
> (2080×2336), nhưng file PNG trên đĩa đã resize về **224×224**. Dùng lẫn hai
> hệ là box rơi sai chỗ mà không có lỗi nào báo ra.

---

## Chặng 2 — Thành một dòng JSONL

`build_dataset.py` ghi ra `data/processed/val.jsonl`. Dòng tương ứng:

| Trường | Giá trị | Từ đâu ra |
| --- | --- | --- |
| `sample_id` | `9a5094b2…_0` | `image_id` + số thứ tự câu hỏi |
| `image_path` | `train_data/train/9a5094b2….png` | tra theo `image_id` |
| `split` | `val` | chia phân tầng 85/15 |
| `question` | Where is the Cardiomegaly? | HF |
| `gt_finding` | Cardiomegaly | HF |
| `gt_location` | `<loc_691_1375_1653_1831>` | HF, **giữ nguyên** |
| `bbox_orig` | `[691, 1375, 1653, 1831]` | parse từ trên |
| `bbox_resized` | `[74.42, 131.85, 178.02, 175.58]` | × fx, fy |
| `bbox_norm` | `[332, 588, 794, 783]` | chuẩn hoá về 0–999 |
| `loc_norm` | `<loc_332_588_794_783>` | **dùng để huấn luyện** |
| `box_label_consistent` | `True` | box này đúng là của Cardiomegaly |
| `repair_method` | `published` | không cần sửa |
| `gt_location_fixed` | `<loc_691_1375_1653_1831>` | giống bản gốc |
| `image_width/height` | 2080 / 2336 | `resize_factors.csv` |

Một box, **ba hệ toạ độ**, lưu sẵn cả ba để không bao giờ phải đoán:

```
pixel gốc        691, 1375, 1653, 1831      (hệ của gt_location)
224×224          74,4  131,9  178,0  175,6  (hệ của file PNG)
chuẩn hoá 0-999  332, 588, 794, 783         (hệ dùng để train)
```

### Mỗi hệ toạ độ dùng để làm gì

Ảnh này gốc **2080 × 2336**, file PNG trên đĩa **224 × 224**.

**Hệ 1 — pixel gốc.** Hệ của ảnh DICOM nguyên bản, cũng là hệ mà HF công bố
`gt_location` và là hệ của `annotations_train.csv`. Đây là *sự thật gốc*: mọi
phép so với ground truth cuối cùng đều quy về đây. Nhược điểm là mỗi ảnh một
kích thước, nên `691` ở ảnh này và `691` ở ảnh khác không cùng nghĩa.

**Hệ 2 — 224×224.** Hệ của file PNG thật sự nằm trên đĩa. `resize_factors.csv`
cho hệ số:

```
fx = 0,107692 = 224 / 2080          691  × 0,107692 = 74,42
fy = 0,095890 = 224 / 2336          1375 × 0,095890 = 131,85
```

Dùng khi cần vẽ box lên đúng file ảnh (`visualize.py`), hoặc khi cần biết tổn
thương nằm ở đâu trên bức ảnh mô hình thực sự nhìn thấy. Chú ý `fx ≠ fy`: ảnh
gốc không vuông mà bị bóp về vuông, nên phải nhân hai hệ số khác nhau — dùng
một tỉ lệ chung cho cả hai trục là box méo.

**Hệ 3 — chuẩn hoá 0–999.** Hệ dùng để huấn luyện:

```
691  × 999/2080 = 331,88 → 332      1653 × 999/2080 = 794,05 → 794
1375 × 999/2336 = 588,02 → 588      1831 × 999/2336 = 783,03 → 783
```

Hai lý do bắt buộc phải có hệ này:

- **Độ rộng cố định 3 chữ số.** `<loc_332_588_794_783>` luôn đúng 4 nhóm × 3
  chữ số. `SoftBoxGIoULoss` (chặng 5) phải biết *chính xác* chữ số thứ mấy
  trong chuỗi là toạ độ nào để bốc logits ra tính IoU khả vi. Pixel gốc thì lúc
  3 chữ số lúc 4 (`691` so với `1375`), vị trí token xô lệch theo từng mẫu, và
  soft-IoU không gióng hàng được nữa.
- **Độc lập kích thước ảnh.** `332` nghĩa là "33% chiều ngang lồng ngực" ở mọi
  bức ảnh. Mô hình học vị trí giải phẫu, không học thuộc pixel của từng tấm.

### Vì sao lưu sẵn cả ba thay vì tính lúc cần

Mỗi chặng dùng một hệ khác nhau: train dùng hệ 3, vẽ hình dùng hệ 2, chấm điểm
so với ground truth dùng hệ 1. Chuyển qua lại cần `width`/`height`/`fx`/`fy`
**của đúng ảnh đó**. Tính lại ở chặng sau mà vớ nhầm kích thước của ảnh khác
thì box lệch vài trăm pixel *mà không có lỗi nào bắn ra* — IoU tụt và không ai
biết tại sao. Lưu sẵn cả ba là mua sự chắc chắn bằng vài chục byte mỗi dòng.

### Chặng 2b — khi nhãn và box **không khớp** (60% dataset)

Bản ghi trên là ca **tốt**. Đây là ca xấu, ảnh `051132a778e61a86eb147c7c6f564dfe`:

```
question              Is there Cardiomegaly?
gt_finding            Cardiomegaly
gt_location           <loc_1264_743_1611_1019>      ← box được công bố
```

Đối chiếu với **toàn bộ 7 box thô** của ảnh đó — ảnh này cũng được ba bác sĩ
đọc độc lập:

```
  Cardiomegaly        R10  [ 952 1318 2034 1724]   IoU với box công bố = 0,00
  Cardiomegaly        R8   [ 948 1267 2063 1672]   IoU = 0,00
  Cardiomegaly        R9   [ 962 1331 2035 1625]   IoU = 0,00
  Aortic enlargement  R10  [1265  744 1611 1019]   IoU = 0,99  ←── ĐÂY MỚI LÀ NÓ
  Aortic enlargement  R8   [1306  731 1614 1066]   IoU = 0,73
  Aortic enlargement  R9   [1276  730 1613 1067]   IoU = 0,79
  Pleural thickening  R9   [1593  452 1804  604]   IoU = 0,00
```

Nhãn ghi **Cardiomegaly** nhưng box được công bố chính là box **Aortic
enlargement của bác sĩ R10** — IoU 0,99, chênh 0,01 chỉ do làm tròn. IoU với
mọi box Cardiomegaly thật đều bằng **0,00**: hai vùng này cách xa nhau thật
sự, tim to nằm ở giữa–dưới lồng ngực (y ≈ 1300–1700) còn quai động mạch chủ ở
trên (y ≈ 730–1050), không chồng lấn một pixel nào.

Nếu train thẳng, mô hình học chỉ vào động mạch chủ mỗi khi nghe "tim to" — sai
về y khoa, và sai một cách *nhất quán* nên nó học rất nhanh.

#### Vì sao hỏng: pipeline ghép nhầm hai danh sách

VinDr-CXR-VQA được sinh tự động. Khi tạo câu hỏi cho một ảnh, pipeline lấy
**danh sách bệnh lý** và **danh sách box** của ảnh đó rồi ghép lại — nhưng ghép
*độc lập*, không kiểm tra box có thuộc đúng bệnh lý không. Nên `gt_location`
gần như luôn là một box thật của bác sĩ trên đúng ảnh đó, chỉ là **của lớp
khác**. Tỉ lệ 60% lấy từ `data/processed/build_summary.json`:

```
box_label_consistent    7 043
box_label_mismatch     10 554      ← 10.554 / 17.597 = 60,0%
```

#### `repair_box()` nối lại — bốn bước

**1. Box công bố có đúng là box của `gt_finding` không?** So với box **thô**
(chưa gộp), vì bản release copy từ đó ra. Ở đây box công bố trùng một box
*Aortic enlargement*, không trùng box *Cardiomegaly* nào → không nhất quán.

**2. Gộp các lần đọc trùng nhau** (`fuse_boxes`, giống chặng 1):

```
  Cardiomegaly        votes=3  [ 954 1305 2044 1674]   IoU = 0,00
  Aortic enlargement  votes=3  [1282  735 1613 1051]   IoU = 0,83
  Pleural thickening  votes=1  [1593  452 1804  604]   IoU = 0,00
```

Box Cardiomegaly đồng thuận là **trung bình cộng** ba lần đọc:

```
x_min: (952,06 + 948,38 + 962,09) / 3 = 954,18
y_min: (1317,80 + 1267,42 + 1331,01) / 3 = 1305,41
x_max: (2033,72 + 2063,35 + 2035,49) / 3 = 2044,19
y_max: (1723,94 + 1672,18 + 1624,95) / 3 = 1673,69
```

Trung bình chứ không phải **hợp** (union): hợp sẽ phình box ra bằng phần tử
lớn nhất của cả ba, còn trung bình cho ước lượng tốt nhất về ranh giới thật.
`votes` cho biết mấy bác sĩ đồng ý — đó là cơ sở để tin box nào.

**3. Chọn ứng viên chồng lấn nhiều nhất với box công bố.** Ở đây IoU = 0,00,
không có gì để bám → `method = "consensus"` (bỏ hẳn box công bố, lấy box đồng
thuận mạnh nhất của `gt_finding`). Nếu box công bố có chồng lấn *một phần* —
lệch nhưng vẫn trúng vùng tim — thì `method = "best_iou"`, tức tin box công bố
đủ để nó định hướng chọn đúng tổn thương.

**4. Ghi kết quả:**

```
box_label_consistent   False
repair_method          consensus
gt_location_fixed      <loc_954_1305_2044_1674>   ← box Cardiomegaly đồng thuận, 3/3 phiếu
```

#### Bốn kiểu `repair_method` và tần suất thật

| method | nghĩa | số bản ghi |
| --- | --- | ---: |
| `published` | box công bố vốn đã đúng | 7.043 |
| `consensus` | không chồng lấn → lấy box đồng thuận | 8.008 |
| `best_iou` | có chồng lấn → giữ box cùng lớp trùng nhất | 2.546 |
| `unavailable` | ảnh không có box nào của `gt_finding` | 0 |

Cộng lại đúng 17.597 cặp QA.

#### Giữ cả hai bản

Record lưu song song `gt_location` (bản công bố) và `gt_location_fixed` (bản đã
nối lại), chọn bằng `data.box_source` trong `configs/default.yaml`:

- `published` — tái lập đúng paper, kể cả phần sai
- `fixed` — **mặc định của repo này**, dùng để huấn luyện thật

Giữ cả hai nghĩa là so sánh được "số của paper" và "số đúng" trên cùng một lần
chạy, thay vì phải chọn một trong hai ngay từ lúc build dataset. Chi tiết:
[DATA_NOTES.md](DATA_NOTES.md).

---

## Chặng 3 — Dựng prompt và chuỗi đích

`prompts.py` biến record thành thứ mô hình đọc được:

**SYSTEM**
```
You are an expert radiologist analysing a chest X-ray. Answer the question,
cite the location of each finding you mention as a
<loc_xmin_ymin_xmax_ymax> span, and then explain your reasoning.
```

**USER** (ảnh + câu hỏi)
```
Where is the Cardiomegaly? Give every location as <loc_xmin_ymin_xmax_ymax>
with coordinates normalised to 0-999.
```

**TARGET** — đáp án mô hình phải học viết ra
```
The Cardiomegaly finding refers to an enlargement of the cardiac silhouette,
centrally located in the chest <loc_332_588_794_783>. Reasoning: Cardiomegaly
is a nonspecific finding often indicative of chronic cardiac stress, such as
chronic hypertension, valvular disease, or cardiomyopathy.
```

Ba điều đã xảy ra ở đây:

1. Box pixel gốc `<loc_691_1375_1653_1831>` được **thay** bằng bản chuẩn hoá
   `<loc_332_588_794_783>`. Chuỗi đích được **dựng lại** quanh box mà config
   yêu cầu, nên toạ độ giám sát và hệ toạ độ không bao giờ lệch nhau.
2. `reason` được nối vào sau `Reasoning:` — đây là phần giải thích được, thứ
   làm nên chữ "explainable" của paper.
3. Hàm trả về **vị trí ký tự** của span: `[108:129]`. Chặng sau cần nó.

Mỗi toạ độ đúng **3 chữ số** (`332`, `588`, `794`, `783`). Đây không phải tình
cờ — xem chặng 5.

---

## Chặng 4 — Token hoá và che nhãn

`data.MedGemmaCollator` chuẩn bị tensor cho mô hình:

```
input_ids = [ prompt (ảnh 256 token + câu hỏi) ][ target ]
labels    = [ -100  -100  -100  …  -100       ][ target ]
                   ↑ che đi: không tính loss trên phần đề bài
```

Che prompt là bắt buộc — nếu không, mô hình sẽ được thưởng vì học thuộc câu
hỏi thay vì học trả lời. Token ảnh cũng bị che.

Sau đó collator tìm **12 token chữ số** của toạ độ, dùng `offset_mapping` để
ánh xạ từ vị trí ký tự `[108:129]` sang chỉ số token:

```
'<' 'loc' '_' '3' '3' '2' '_' '5' '8' '8' '_' '7' '9' '4' '_' '7' '8' '3' '>'
             └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
               x_min       y_min       x_max       y_max
```

12 chữ số thật của bản ghi này: `3 3 2 | 5 8 8 | 7 9 4 | 7 8 3`

Nếu không tìm đúng 12 token (bản ghi hỏng, hoặc dùng `loc_format: pixel` vì
toạ độ pixel không cố định độ dài), mẫu đó **bị bỏ qua ở phần loss box** chứ
không bị giám sát sai vị trí.

---

## Chặng 5 — Hàm mất mát đa nhiệm

Paper (Mục 3.1): `L_total = L_VQA + λ · L_bbox`, với `λ = 0,5`.

**`L_VQA`** — cross-entropy thông thường trên token đáp án. Dạy mô hình viết.

**`L_bbox`** — đây là phần thú vị. Paper mô tả "IoU-based regression" trên toạ
độ "dự đoán qua special token", **không thêm đầu hồi quy nào**. Cách hiện thực:

```
Tại mỗi vị trí chữ số, mô hình xuất phân phối trên 10 token '0'..'9'.
   Chữ số kỳ vọng  =  Σ  p(d) · d          ← khả vi theo logit
   Toạ độ          =  100·e₀ + 10·e₁ + e₂
   4 toạ độ        →  một "soft box"
   loss            =  1 − GIoU(soft box, box đúng)
```

Với bản ghi này, đích so sánh là:

```
box đích, chuẩn hoá 0-999 : [332, 588, 794, 783]
cùng box, thang [0,1]     : [0.3323, 0.5886, 0.7948, 0.7838]
```

> **Vì sao cần 3 chữ số cố định?** Vì bộ giải mã phải biết chắc vị trí nào là
> hàng trăm, hàng chục, hàng đơn vị. Toạ độ pixel (`691`, `1375` — 3 và 4 chữ
> số) thì không làm được. Đó là lý do `loc_format: norm1000` là mặc định; đặt
> `pixel` thì phần loss box tự tắt.

> **Vì sao GIoU chứ không phải IoU?** IoU thường bằng 0 khi hai box **không
> chồng nhau chút nào** — tức là phẳng lì, không có gradient, đúng vào giai
> đoạn đầu huấn luyện khi mô hình đoán lung tung. GIoU vẫn cho gradient trong
> trường hợp đó.

Kết quả chặng này: thư mục adapter LoRA (~8,9M tham số, 0,21% mô hình).

---

## Chặng 6 — Sinh dự đoán

`infer.py` cho mô hình xem ảnh + câu hỏi, rồi ghi lại nó viết gì:

```json
{"sample_id": "9a5094b2…_0",
 "image_id":  "9a5094b2…",
 "question":  "Where is the Cardiomegaly?",
 "prediction": "Yes, there is cardiomegaly centrally in the chest
                <loc_332_580_797_770>."}
```

Chạy hai lần: không adapter (dòng **Pretrained**) và có adapter (dòng
**Fine-tuned**).

---

## Chặng 7 — Chấm điểm

### 7a. Phần chữ → Bảng 4

```
văn bản dự đoán  →  mentioned()  →  {'Cardiomegaly'}
```

Đối chiếu với tập bệnh lý thật của ảnh — ở đây là `{Cardiomegaly, Aortic
enlargement, Pleural thickening, Pleural effusion}`:

```
Cardiomegaly         có thật ✓  có nói ✓  →  TP
Aortic enlargement   có thật ✓  không    →  FN
Pleural thickening   có thật ✓  không    →  FN
Pleural effusion     có thật ✓  không    →  FN
10 bệnh lý còn lại   không      không    →  TN
```

### 7b. Phần box → Bảng 5 và 6

```
<loc_332_580_797_770>  →  parse  →  [332, 580, 797, 770]   (0-999)
                       →  đổi hệ →  [691, 1356, 1659, 1801] (pixel gốc)
```

Rồi tính IoU với **từng** box thật của ảnh:

```
  Cardiomegaly        IoU 0,950   ←── TP (≥ 0,3)
  Cardiomegaly        IoU 0,889
  Cardiomegaly        IoU 0,828
  Aortic enlargement  IoU 0,000
  Pleural thickening  IoU 0,000
  Pleural effusion    IoU 0,000
```

Ghép **tham lam 1-1**: box dự đoán này nhận cặp ghép tốt nhất (IoU 0,950) với
một box Cardiomegaly. Hai box Cardiomegaly còn lại và 3 box khác trở thành
**FN** — mô hình chỉ vẽ một hình chữ nhật trong khi ảnh có 6.

> Đây chính là **trần** nói tới trong [EVALUATION.md](EVALUATION.md): dataset
> chỉ hỏi ~4 câu cho ~8,4 box mỗi ảnh, nên recall box không thể vượt 0,474 dù
> mô hình có hoàn hảo.

### 7c. Gộp thành bảng

Cộng dồn TP/FP/FN trên 659 ảnh validation → `results.py` dựng Bảng 4, 5, 6 và
đặt cạnh số của paper.

Cách đọc từng con số: [EVALUATION.md](EVALUATION.md).

---

## Tóm tắt: record biến đổi ra sao

| Chặng | Bản ghi tồn tại dưới dạng | File |
| --- | --- | --- |
| 0 | JSON từ HF + dòng CSV + file PNG | `data_v1.json`, `annotations_train.csv` |
| 1 | `ImageMeta` + box đồng thuận | `vindr.py` |
| 2 | 1 dòng JSONL, 3 hệ toạ độ, box gốc + box sửa | `build_dataset.py` |
| 3 | prompt + chuỗi đích + vị trí ký tự của span | `prompts.py` |
| 4 | `input_ids`, `labels` (đã che), 12 vị trí chữ số | `data.py` |
| 5 | 2 số loss → cập nhật trọng số LoRA | `losses.py`, `train.py` |
| 6 | chuỗi văn bản mô hình sinh ra | `infer.py` |
| 7 | TP/FP/FN + IoU → một hàng trong bảng | `evaluate.py`, `results.py` |

## Tự chạy lại vết này

```bash
# chặng 0-2: dựng dataset (không cần GPU)
python -m vindr_cxr_vqa.build_dataset
python -m vindr_cxr_vqa.stats

# chặng 7: kiểm tra bộ chấm bằng chính đáp án (không cần GPU)
python -m vindr_cxr_vqa.evaluate --oracle --split val

# chặng 3-6: cần GPU
python -m vindr_cxr_vqa.train
python -m vindr_cxr_vqa.infer --split val --adapter outputs/medgemma-lora \
    --out outputs/preds_val_ft.jsonl
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_ft.jsonl --split val
```

Xem một bản ghi bất kỳ sau chặng 2:

```bash
head -1 data/processed/val.jsonl
```
