// Left column: pick a record from a built split, or drop in your own X-ray.
//
// A dataset record brings ground-truth boxes with it, so the answer can be
// scored on the spot. An uploaded image cannot be scored -- there is nothing
// to score it against -- which the panel says rather than quietly showing an
// empty IoU.

import { useRef } from 'react'

const SPLITS = ['val', 'test', 'train']

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
        {SPLITS.map((s) => (
          <button
            key={s}
            className={s === split ? 'tab on' : 'tab'}
            onClick={() => onSplit(s)}
            disabled={!counts?.[s]}
          >
            {s}
            {counts?.[s] ? <em>{counts[s].toLocaleString()}</em> : null}
          </button>
        ))}
      </div>

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
