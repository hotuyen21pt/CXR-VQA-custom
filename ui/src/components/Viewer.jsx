// The X-ray with the boxes drawn on it.
//
// Boxes arrive as `rel`, 0-1 of the image's own extent, so they are laid out
// as percentages of the picture element. That keeps the overlay correct at any
// rendered size -- the stored PNGs are 224x224 but the browser shows them much
// larger, and the model's coordinates are in a third space again.
//
// Colours follow visualize.py / Figure 2: green is ground truth, red is the
// model.

const GT = 'gt'
const PRED = 'pred'

function Box({ rel, kind, label }) {
  const [x1, y1, x2, y2] = rel
  const style = {
    left: `${x1 * 100}%`,
    top: `${y1 * 100}%`,
    width: `${(x2 - x1) * 100}%`,
    height: `${(y2 - y1) * 100}%`,
  }
  return (
    <div className={`box ${kind}`} style={style}>
      {label && <span className="tag">{label}</span>}
    </div>
  )
}

export default function Viewer({
  src, result, showGt, showPred, onToggleGt, onTogglePred, busy, empty,
}) {
  const gtBoxes = result?.gt_boxes || []
  const predBoxes = result?.pred_boxes || []

  return (
    <div className="viewer">
      <div className="legend">
        <label className={showGt ? '' : 'off'}>
          <input type="checkbox" checked={showGt} onChange={onToggleGt} />
          <span className="swatch gt" /> Ground truth ({gtBoxes.length})
        </label>
        <label className={showPred ? '' : 'off'}>
          <input type="checkbox" checked={showPred} onChange={onTogglePred} />
          <span className="swatch pred" /> Dự đoán ({predBoxes.length})
        </label>
        {result?.mock && <span className="chip warn">MOCK</span>}
        {typeof result?.latency_s === 'number' && (
          <span className="muted">{result.latency_s}s</span>
        )}
      </div>

      <div className="stage">
        {empty && <div className="placeholder muted">Chọn một mẫu bên trái, hoặc tải ảnh lên</div>}
        {src && (
          <div className="frame">
            <img src={src} alt="chest X-ray" />
            {showGt && gtBoxes.map((b, i) => (
              <Box key={`g${i}`} rel={b.rel} kind={GT} label={b.class_name} />
            ))}
            {showPred && predBoxes.map((b, i) => (
              <Box
                key={`p${i}`}
                rel={b.rel}
                kind={PRED}
                label={b.best_iou == null ? 'pred' : `IoU ${b.best_iou.toFixed(2)}`}
              />
            ))}
            {busy && <div className="scrim">đang suy luận…</div>}
          </div>
        )}
      </div>
    </div>
  )
}
