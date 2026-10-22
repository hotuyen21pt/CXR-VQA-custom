import { useCallback, useEffect, useMemo, useState } from 'react'
import Header from './components/Header.jsx'
import SamplePicker from './components/SamplePicker.jsx'
import Viewer from './components/Viewer.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import {
  defaultBackend, rememberBackend, getHealth, getMeta, getSamples,
  imageUrl, predict,
} from './api.js'

export default function App() {
  const [backend, setBackend] = useState(defaultBackend)
  const [health, setHealth] = useState(null)
  const [meta, setMeta] = useState(null)
  const [connError, setConnError] = useState(null)
  const [probe, setProbe] = useState(0)

  const [split, setSplit] = useState('val')
  const [query, setQuery] = useState('')
  const [type, setType] = useState('')
  const [samples, setSamples] = useState([])
  const [total, setTotal] = useState(0)
  const [listing, setListing] = useState(false)

  const [selected, setSelected] = useState(null)
  const [upload, setUpload] = useState(null)
  const [question, setQuestion] = useState('')

  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [runError, setRunError] = useState(null)

  const [showGt, setShowGt] = useState(true)
  const [showPred, setShowPred] = useState(true)

  // -- connection ---------------------------------------------------------
  useEffect(() => {
    let alive = true
    setConnError(null)
    Promise.all([getHealth(backend), getMeta(backend)])
      .then(([h, m]) => {
        if (!alive) return
        setHealth(h)
        setMeta(m)
        rememberBackend(backend)
      })
      .catch((e) => alive && (setHealth(null), setConnError(e.message)))
    return () => { alive = false }
  }, [backend, probe])

  // Keep polling only while the server is in the middle of loading weights.
  useEffect(() => {
    if (health?.state !== 'loading') return
    const id = setInterval(() => setProbe((n) => n + 1), 3000)
    return () => clearInterval(id)
  }, [health?.state])

  // -- sample list --------------------------------------------------------
  useEffect(() => {
    if (!health) return
    let alive = true
    setListing(true)
    const timer = setTimeout(() => {
      getSamples(backend, { split, q: query, type })
        .then((d) => {
          if (!alive) return
          setSamples(d.items)
          setTotal(d.total)
        })
        .catch(() => alive && (setSamples([]), setTotal(0)))
        .finally(() => alive && setListing(false))
    }, query ? 250 : 0)                     // debounce only the typing
    return () => { alive = false; clearTimeout(timer) }
  }, [backend, health, split, query, type])

  // -- what is on screen --------------------------------------------------
  const src = upload ? upload.dataUrl
    : selected ? imageUrl(backend, selected.image_id)
      : null

  const canRun = Boolean(upload ? question.trim() : selected)

  function pickSample(s) {
    setUpload(null)
    setSelected(s)
    setQuestion(s.question)
    setResult(null)
    setRunError(null)
  }

  function takeUpload(file) {
    setUpload(file)
    setSelected(null)
    setResult(null)
    setRunError(null)
    if (!question.trim()) setQuestion('Where is the Cardiomegaly?')
  }

  const run = useCallback(async () => {
    setBusy(true)
    setRunError(null)
    try {
      const body = upload
        ? { image_b64: upload.dataUrl, question: question.trim() }
        : { sample_id: selected.sample_id, question: question.trim() || undefined }
      setResult(await predict(backend, body))
    } catch (e) {
      setResult(null)
      setRunError(e.message)
    } finally {
      setBusy(false)
      setProbe((n) => n + 1)                // health may have flipped to ready
    }
  }, [backend, upload, selected, question])

  const presets = useMemo(() => {
    const name = selected?.gt_finding || 'Cardiomegaly'
    return [
      `Where is the ${name}?`,
      `Is there ${name}?`,
      'What abnormality is present in this chest X-ray?',
      'How many lesions are visible?',
      'Which lung field is affected?',
    ]
  }, [selected?.gt_finding])

  return (
    <div className="app">
      <Header
        backend={backend}
        onBackend={setBackend}
        health={health}
        error={connError}
        onRetry={() => setProbe((n) => n + 1)}
      />

      {connError ? (
        <div className="fatal">
          <h2>Không gọi được backend</h2>
          <p className="mono">{connError}</p>
          <p>Trên máy có GPU và adapter, chạy:</p>
          <pre className="raw">
python -m vindr_cxr_vqa.serve --adapter outputs/medgemma-lora --host 0.0.0.0
          </pre>
          <p>Chỉ muốn thử giao diện, không cần GPU:</p>
          <pre className="raw">python -m vindr_cxr_vqa.serve --mock</pre>
          <p>Rồi dán địa chỉ nó in ra vào ô <strong>Backend</strong> phía trên.</p>
        </div>
      ) : (
        <main className="layout">
          <SamplePicker
            split={split}
            onSplit={setSplit}
            counts={health?.splits}
            query={query}
            onQuery={setQuery}
            type={type}
            onType={setType}
            questionTypes={meta?.question_types || []}
            samples={samples}
            total={total}
            loading={listing}
            selectedId={selected?.sample_id}
            onSelect={pickSample}
            upload={upload}
            onUpload={takeUpload}
            onClearUpload={() => setUpload(null)}
          />

          <section className="middle">
            <Viewer
              src={src}
              result={result}
              showGt={showGt}
              showPred={showPred}
              onToggleGt={() => setShowGt((v) => !v)}
              onTogglePred={() => setShowPred((v) => !v)}
              busy={busy}
              empty={!src}
            />

            <div className="ask">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Câu hỏi bằng tiếng Anh, ví dụ: Where is the Cardiomegaly?"
                rows={2}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canRun && !busy) run()
                }}
              />
              <button className="primary" onClick={run} disabled={!canRun || busy}>
                {busy ? 'Đang chạy…' : 'Chạy'}
              </button>
            </div>

            <div className="presets">
              {presets.map((p) => (
                <button key={p} className="ghost tiny" onClick={() => setQuestion(p)}>
                  {p}
                </button>
              ))}
            </div>
          </section>

          <ResultPanel result={result} meta={meta} error={runError} />
        </main>
      )}
    </div>
  )
}
