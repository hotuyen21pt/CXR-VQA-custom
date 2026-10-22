// The backend URL bar, and what the server says about itself.
//
// The health payload is the one place that answers "which model am I actually
// talking to" -- fine-tuned or zero-shot, real or mock, dataset found or not --
// so it is on screen at all times rather than behind a click.

export default function Header({ backend, onBackend, health, error, onRetry }) {
  const state = health?.state
  const tone =
    error ? 'bad' : state === 'ready' || state === 'cold' ? 'good'
      : state === 'mock' ? 'warn' : state === 'error' ? 'bad' : 'warn'

  const label =
    error ? 'không kết nối được'
      : state === 'mock' ? 'MOCK — không có model'
        : state === 'ready' ? 'model đã nạp'
          : state === 'loading' ? 'đang nạp model…'
            : state === 'error' ? 'model lỗi'
              : state === 'cold' ? 'sẵn sàng (model nạp ở câu hỏi đầu tiên)'
                : '…'

  return (
    <header className="header">
      <div className="brand">
        <strong>VinDr-CXR-VQA</strong>
        <span className="muted">inference</span>
      </div>

      <label className="backend">
        <span className="muted">Backend</span>
        <input
          value={backend}
          onChange={(e) => onBackend(e.target.value)}
          spellCheck={false}
          placeholder="http://127.0.0.1:8000"
        />
      </label>

      <button className="ghost" onClick={onRetry}>Kiểm tra lại</button>

      <div className={`badge ${tone}`} title={error || health?.error || ''}>
        <span className="dot" />
        {label}
      </div>

      {health && !error && (
        <div className="health muted">
          <span>{health.mode}</span>
          <span>·</span>
          <span title={health.model}>{health.model.split('/').pop()}</span>
          {health.adapter && (
            <>
              <span>·</span>
              <span className="mono" title={health.adapter}>
                {health.adapter.split(/[\\/]/).pop()}
              </span>
            </>
          )}
          <span>·</span>
          <span>{health.images_indexed.toLocaleString()} ảnh</span>
          {!health.dataset_dir_exists && (
            <span className="warn-text" title={health.dataset_dir}>
              · thiếu thư mục ảnh
            </span>
          )}
        </div>
      )}
    </header>
  )
}
