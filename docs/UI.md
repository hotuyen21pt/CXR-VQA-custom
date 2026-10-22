# UI thử inference

Một trang React để hỏi model từng ảnh một, xem nó trả lời gì, vẽ box nó đoán
đè lên box của bác sĩ, và chấm IoU ngay tại chỗ.

Hai phần rời nhau:

| Phần | Ở đâu | Cần gì |
| --- | --- | --- |
| `vindr_cxr_vqa/serve.py` | **máy có GPU** | torch, transformers, peft, Pillow — và adapter |
| `ui/` (React + Vite) | máy nào cũng được | node 18+ |

React không đụng tới torch; nó chỉ gọi HTTP. Nên GPU có thể là một session
Kaggle, một runtime Colab hay một con máy ở xa, còn trang web chạy trên laptop.

---

## Gắn output vào đâu

Training để lại bốn thứ. Đây là chỗ cắm từng thứ:

| Output | Nó là gì | Gắn vào |
| --- | --- | --- |
| `medgemma-lora/` hoặc `medgemma-lora.zip` | **adapter LoRA** — thứ duy nhất mà training thật sự sinh ra | `--adapter` |
| MedGemma base weights (~8 GB) | trọng số gốc mà LoRA đắp lên | `--model`, hoặc để nguyên và tải từ HuggingFace |
| `dataset/` | ảnh X-quang PNG, để hiện lên màn hình | `--dataset-dir` |
| `data/processed/` | `train/val/test.jsonl` + `images.json` — danh sách mẫu và box ground truth để chấm IoU | `--processed-dir` |

Ba cái sau có mặc định đúng trong `configs/default.yaml`, nên nếu anh chạy ngay
trong repo thì **chỉ cần `--adapter`**:

```bash
python -m vindr_cxr_vqa.serve --adapter outputs/medgemma-lora
```

Cũng nhận được cả `.env`:

```bash
export VINDR_ADAPTER=outputs/medgemma-lora
export VINDR_MODEL=/path/to/medgemma-4b-it     # khi chạy offline
export VINDR_DATASET_DIR=/path/to/dataset
export VINDR_PROCESSED_DIR=/path/to/data/processed
```

### `--adapter` nhận ba hình dạng

Training kết thúc theo nhiều kiểu khác nhau, nên cả ba đều dùng được:

```bash
--adapter outputs/medgemma-lora          # chạy xong trọn vẹn
--adapter outputs/medgemma-lora.zip      # file Kaggle notebook lưu ra, tự giải nén
--adapter outputs/medgemma-lora/checkpoint-600   # một checkpoint cụ thể
```

Nếu đưa vào một thư mục chỉ có `checkpoint-N` bên trong (run bị cắt giữa chừng),
nó tự lấy checkpoint mới nhất và in ra cái nó chọn.

**Bỏ hẳn `--adapter`** thì chạy MedGemma gốc chưa fine-tune — chính là dòng
"Pretrained" trong Bảng 4–6 của paper. Mở hai server ở hai cổng khác nhau là so
sánh được trực tiếp.

### Lấy adapter từ Kaggle về

Cell cuối của `notebooks/kaggle_train.ipynb` ghi ra
`/kaggle/working/medgemma-lora.zip` (vài chục MB — LoRA nhỏ, không phải 8 GB).
Tải từ tab **Output** của notebook, rồi trỏ `--adapter` thẳng vào file zip,
không cần giải nén tay.

---

## Chạy

### Cách 1 — tất cả trên một máy có GPU

```bash
cd ui && npm install && npm run build && cd ..
python -m vindr_cxr_vqa.serve --adapter outputs/medgemma-lora
```

Mở `http://127.0.0.1:8000`. `serve.py` tự phục vụ luôn `ui/dist`, không cần
node chạy nền.

### Cách 2 — GPU ở xa, UI ở laptop

Trên máy GPU:

```bash
python -m vindr_cxr_vqa.serve --adapter outputs/medgemma-lora \
    --host 0.0.0.0 --port 8000
```

Trên laptop:

```bash
cd ui && npm install && npm run dev
```

Mở `http://localhost:5173`, dán địa chỉ máy GPU vào ô **Backend** trên đầu
trang. Nó được nhớ trong `localStorage`, lần sau khỏi gõ lại.

### Cách 3 — chỉ xem giao diện, không có GPU

```bash
python -m vindr_cxr_vqa.serve --mock
```

Trả lời dựng sẵn, không nạp model, không cần torch. Mỗi câu trả lời được đánh
dấu `MOCK` trên màn hình. Dùng để kiểm tra giao diện và luồng dữ liệu —
**không phản ánh model nói gì**.

### Cách 4 — Docker (không phải lo python trên máy)

Đây là cách gọn nhất khi python trên máy không cài được torch. Image tự build
React bằng node rồi chạy backend bằng python 3.11 Linux; model, ảnh và adapter
đều mount vào, không nướng vào image.

```bash
cp docker/.env.example docker/.env     # rồi điền HF_TOKEN
docker compose -f docker/compose.yaml build

# tải trọng số một lần (~8 GB) về ./outputs/models/
docker compose -f docker/compose.yaml run --rm fetch-model

# chạy
docker compose -f docker/compose.yaml up serve        # CPU
docker compose -f docker/compose.yaml up serve-gpu    # máy có GPU
docker compose -f docker/compose.yaml up mock         # chỉ giao diện
```

Mở `http://localhost:8000`.

Bốn thứ cần gắn, giờ nằm trong `docker/.env`:

| Biến | Là gì |
| --- | --- |
| `HF_TOKEN` | chỉ `fetch-model` đọc; các service khác chạy với `HF_HUB_OFFLINE=1` |
| `MEDGEMMA_REPO` | repo trên HuggingFace, mặc định `google/medgemma-1.5-4b-it` |
| `MODEL_DIR` | thư mục dưới `outputs/models/` mà trọng số nằm vào |
| `ADAPTER` | **đường dẫn trong container**. `outputs/` mount ở `/app/outputs`, nên `./outputs/medgemma-lora.zip` ở host là `/app/outputs/medgemma-lora.zip` |

**Chọn base model cho đúng.** Adapter ghi lại nó được train trên model nào,
trong `adapter_config.json` → `base_model_name_or_path`. Nạp LoRA lên một base
khác thì nó vẫn load được nhưng trả lời rác. Đọc ra bằng:

```bash
python - <<'EOF'
import json, zipfile
print(json.loads(zipfile.ZipFile('outputs/medgemma-lora.zip')
      .read('adapter_config.json'))['base_model_name_or_path'])
EOF
```

**Chạy CPU thì chậm.** Một câu trả lời mất khoảng 2–5 phút, vì mỗi token là
một lượt forward qua 4 tỉ tham số. Đủ để xem hệ thống chạy đúng, không đủ để
chấm cả split — cái đó vẫn nên đẩy lên Kaggle. Service `serve` đã đặt sẵn
`--dtype bfloat16` (torch gần như không có kernel fp16 cho CPU),
`--max-new-tokens 160` để cắt bớt phần lý giải, và `--preload` để thời gian
nạp 8 GB nằm trong log `compose up` chứ không nằm trong câu hỏi đầu tiên.

Docker Desktop cấp bao nhiêu RAM cho container là giới hạn thật:
`docker info --format '{{.MemTotal}}'`. Model bf16 cần ~9 GB nên dưới 12 GB sẽ
bị OOM-kill — chỉnh trong Settings → Resources.

### Trên Kaggle

Kaggle không mở cổng ra ngoài, nên cần một đường hầm. Trong một cell:

```python
!pip install -q pyngrok
import subprocess, os
from pyngrok import ngrok
os.environ["VINDR_ADAPTER"] = ADAPTER        # cell train in ra biến này
subprocess.Popen(["python", "-m", "vindr_cxr_vqa.serve",
                  "--host", "0.0.0.0", "--port", "8000"], cwd=WORK)
print(ngrok.connect(8000).public_url)
```

Dán URL ngrok in ra vào ô **Backend**. Bản `dist/` đã build sẵn trong repo
bundle thì mở thẳng URL đó cũng được.

---

## Màn hình có gì

**Cột trái — chọn ảnh.** Ba tab `val` / `test` / `train` theo đúng split đã
build. Ô tìm kiếm lọc theo câu hỏi, `image_id` hay tên bệnh lý; dropdown lọc
theo 6 loại câu hỏi. Chip **"box vá"** nghĩa là mẫu đó nằm trong 60% mà box
công bố thuộc bệnh lý khác với nhãn — xem `docs/DATA_NOTES.md`.

Hoặc kéo-thả ảnh X-quang của anh vào ô dưới. Ảnh ngoài dataset thì không có
ground truth, nên UI nói thẳng là không chấm IoU được, thay vì hiện một con số
rỗng.

**Cột giữa — ảnh và câu hỏi.** Khung xanh lá là box bác sĩ (`boxes_fused`,
hợp nhất từ ba người đọc độc lập), khung đỏ nét đứt là model đoán, kèm IoU ngay
trên khung — cùng bảng màu với Hình 2 trong paper. Tắt/bật từng lớp bằng hai
checkbox.

Câu hỏi mặc định là câu của mẫu, nhưng sửa được. `Ctrl+Enter` để chạy. Mấy nút
gợi ý bên dưới là 5 dạng câu hỏi thường dùng, tự điền tên bệnh lý của mẫu đang
chọn vào.

**Cột phải — kết quả.** Câu trả lời, phần `Reasoning:` tách riêng, danh sách
bệnh lý model nhắc tới (qua đúng hàm `evaluate.mentioned()` dùng để chấm Bảng
4, nên nó cũng bỏ qua các câu phủ định như "no pleural effusion"), và bảng box:
model viết ra gì, quy ra pixel gốc là bao nhiêu, IoU bao nhiêu. Xanh khi
IoU ≥ 0.5, vàng khi ≥ 0.3, đỏ khi thấp hơn — đúng ngưỡng Mục 3.2.

Chuỗi sinh thô nằm ở `<details>` dưới cùng. Mọi con số phía trên đều suy ra từ
chuỗi đó, nên khi thấy số lạ thì văn bản sinh ra nó ở ngay đấy.

---

## Những chỗ hay vướng

**Model nạp lâu.** Lần hỏi đầu tiên phải nạp ~8 GB trọng số, mất vài phút. Badge
trên đầu trang chuyển `đang nạp model…` rồi `model đã nạp`. Muốn nạp ngay lúc
khởi động thì thêm `--preload`.

**Baseline không sinh ra `<loc_...>`.** Bình thường — MedGemma gốc chưa được dạy
cú pháp đó. Đấy chính là cái mà fine-tune thêm vào, và là lý do Bảng 5 của
paper có hai dòng.

**Ảnh không hiện.** Badge sẽ báo `thiếu thư mục ảnh`. `paths.dataset_dir` đang
trỏ sai — sửa trong config hoặc truyền `--dataset-dir`.

**Danh sách mẫu trống.** Chưa build dataset:
`python -m vindr_cxr_vqa.build_dataset`.

**IoU luôn bằng 0 dù box trông đúng chỗ.** `data.loc_format` lúc serve phải
khớp với lúc train. Server lấy `pred_space` từ config đang dùng và hiển thị nó
cạnh tiêu đề bảng box — đối chiếu với config đã train.

**Cổng 8000 bị chiếm.** `--port 8001`, rồi sửa ô Backend cho khớp.

---

## Lưu ý bảo mật

`serve.py` mở CORS cho mọi origin và không có xác thực. Nó được viết để chạy
trên `localhost` hoặc trong mạng LAN tin cậy. Đừng phơi nó ra Internet; ngrok
thì dùng xong tắt đi.
