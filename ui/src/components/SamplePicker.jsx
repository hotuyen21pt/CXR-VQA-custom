// Left column: pick a record from a built split, or drop in your own X-ray.
//
// A dataset record brings ground-truth boxes with it, so the answer can be
// scored on the spot. An uploaded image cannot be scored -- there is nothing
// to score it against -- which the panel says rather than quietly showing an
// empty IoU.

import { useRef } from 'react'

// What each split means for someone reading an answer. The difference that
// matters is whether the model was trained on it: a good answer on `train`
// proves nothing, because it memorised that image.
const SPLITS = [
  ['val', 'Model chưa từng thấy. Bảng 4-6 của paper tính trên tập này — thử ở đây.'],
  ['test', '300 ảnh held-out. Là TẬP CON của val, không phải split thứ ba.'],
  ['train', 'Model ĐÃ thấy lúc train. Trả lời đúng ở đây không chứng minh gì; '
    + 'trả lời sai thì chắc chắn có gì đó hỏng.'],
]

export default function SamplePicker({
  split, onSplit, counts,
  query, onQuery,
  type, onType, questionTypes,
  samples, total, loading,
  selectedId, onSelect,
  upload, onUpload, onClearUpload,
}) {
  const fileRef = useRef(null)

  function readFile(file) {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => onUpload({ name: file.name, dataUrl: reader.result })
    reader.readAsDataURL(file)
  }

  return (
    <aside className="picker">
      <div className="tabs">
        {SPLITS.map(([s, hint]) => (
          <button
            key={s}
            className={s === split ? 'tab on' : 'tab'}
            onClick={() => onSplit(s)}
            disabled={!counts?.[s]}
            title={hint}
          >
            {s}
            {counts?.[s] ? <em>{counts[s].toLocaleString()}</em> : null}
          </button>
        ))}
      </div>

      {split === 'train' && (
        <p className="notice">
          Model đã được huấn luyện trên split này — câu trả lời đúng ở đây
          không nói lên chất lượng. Dùng <strong>val</strong> để đánh giá.
        </p>
      )}

      <div className="filters">
        <input
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Tìm câu hỏi, image_id, bệnh lý…"
        />
        <select value={type} onChange={(e) => onType(e.target.value)}>
          <option value="">Mọi loại câu hỏi</option>
          {questionTypes.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <div
        className={`drop ${upload ? 'on' : ''}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          readFile(e.dataTransfer.files?.[0])
        }}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => readFile(e.target.files?.[0])}
        />
        {upload ? (
          <>
            <span className="mono ellip">{upload.name}</span>
            <button
              className="ghost tiny"
              onClick={(e) => { e.stopPropagation(); onClearUpload() }}
            >
              bỏ
            </button>
          </>
        ) : (
          <span className="muted">Hoặc kéo-thả / chọn ảnh X-quang của anh</span>
        )}
      </div>

      <div className="count muted">
        {loading ? 'đang tải…' : `${samples.length} / ${total.toLocaleString()}`}
      </div>

      <ul className="samples">
        {samples.map((s) => (
          <li
            key={s.sample_id}
            className={s.sample_id === selectedId && !upload ? 'on' : ''}
            onClick={() => onSelect(s)}
          >
            <div className="q">{s.question}</div>
            <div className="meta muted">
              <span className="chip">{s.type}</span>
              <span>{s.gt_finding}</span>
              {s.box_label_consistent === false && (
                <span
                  className="chip bad"
                  title="Box công bố thuộc bệnh lý khác; repo đã vá lại (docs/DATA_NOTES.md)"
                >
                  box vá
                </span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </aside>
  )
}
