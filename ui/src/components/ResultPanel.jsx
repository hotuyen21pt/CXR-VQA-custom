// Right column: what the model said, and how close it landed.
//
// The raw generated string is kept visible at the bottom. Everything above it
// -- the split answer, the findings, the per-box IoU -- is derived from that
// string by the server, so when a number looks wrong the text that produced it
// is right there.

function Iou({ value, threshold, good }) {
  if (value == null) return <span className="muted">—</span>
  const tone = value >= good ? 'good' : value >= threshold ? 'warn' : 'bad'
  return <span className={`iou ${tone}`}>{value.toFixed(3)}</span>
}

export default function ResultPanel({ result, meta, error }) {
  if (error) {
    return (
      <section className="result">
        <div className="panel bad-panel">
          <h3>Lỗi</h3>
          <pre className="raw">{error}</pre>
        </div>
      </section>
    )
  }
  if (!result) {
    return (
      <section className="result">
        <div className="panel muted empty-panel">
          Chưa có kết quả. Bấm <strong>Chạy</strong> để model trả lời.
        </div>
      </section>
    )
  }

  const thr = meta?.iou_threshold ?? 0.3
  const good = meta?.good_iou ?? 0.5
  const scored = result.gt_boxes.length > 0

  return (
    <section className="result">
      <div className="panel">
        <h3>Câu trả lời</h3>
        <p className="answer">{result.answer || <em className="muted">(trống)</em>}</p>
        {result.reason && (
          <>
            <h4>Lý giải</h4>
            <p className="reason">{result.reason}</p>
          </>
        )}
      </div>

      <div className="panel">
        <h3>Bệnh lý được nhắc tới</h3>
        {result.findings.length ? (
          <div className="chips">
            {result.findings.map((f) => (
              <span
                key={f}
                className={`chip ${f === result.record?.gt_finding ? 'good' : ''}`}
              >
                {f}
              </span>
            ))}
          </div>
        ) : (
          <p className="muted">Không nhận ra bệnh lý nào trong 14 lớp.</p>
        )}
        {result.record?.gt_finding && (
          <p className="muted small">
            Nhãn đúng: <strong>{result.record.gt_finding}</strong>
          </p>
        )}
      </div>

      <div className="panel">
        <h3>
          Box dự đoán
          <span className="muted small"> · {result.pred_space}</span>
        </h3>
        {result.pred_boxes.length === 0 ? (
          <p className="muted">
            Model không sinh ra <code>&lt;loc_…&gt;</code> nào — với baseline
            chưa fine-tune thì đây là chuyện thường.
          </p>
        ) : (
          <table className="boxes">
            <thead>
              <tr>
                <th>Model viết ra</th>
                <th>Pixel gốc</th>
                <th>IoU</th>
              </tr>
            </thead>
            <tbody>
              {result.pred_boxes.map((b, i) => (
                <tr key={i}>
                  <td className="mono small">
                    &lt;loc_{b.raw.join('_')}&gt;
                  </td>
                  <td className="mono small">
                    {b.pixel.map((v) => Math.round(v)).join(', ')}
                  </td>
                  <td>
                    {scored
                      ? <Iou value={b.best_iou} threshold={thr} good={good} />
                      : <span className="muted">n/a</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!scored && (
          <p className="muted small">
            Ảnh tải lên không có ground truth nên không chấm IoU được.
          </p>
        )}
        {scored && (
          <p className="muted small">
            Ngưỡng của paper: TP khi IoU ≥ {thr}, "tốt" khi IoU ≥ {good}.
          </p>
        )}
      </div>

      {result.record?.gt_answer && (
        <details className="panel">
          <summary>Đáp án gốc trong dataset</summary>
          <pre className="raw">{result.record.gt_answer}</pre>
          {result.record.gt_reason && (
            <pre className="raw">{result.record.gt_reason}</pre>
          )}
        </details>
      )}

      <details className="panel">
        <summary>Chuỗi sinh thô</summary>
        <pre className="raw">{result.prediction}</pre>
      </details>
    </section>
  )
}
