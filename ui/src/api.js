// Every call to vindr_cxr_vqa.serve goes through here.
//
// The backend URL is not baked in: the GPU is usually somewhere else (a Kaggle
// session behind a tunnel, a remote box) while this page runs on a laptop, so
// it is typed into the header and kept in localStorage.

const KEY = 'vindr.backend'

// Same origin by default, which is what happens when serve.py is serving the
// built `dist/` itself. In `npm run dev` the page is on :5173 and the backend
// on :8000, so that is the fallback guess.
export function defaultBackend() {
  const saved = localStorage.getItem(KEY)
  if (saved) return saved
  const { protocol, hostname, port } = window.location
  if (port === '5173') return `${protocol}//${hostname}:8000`
  return window.location.origin
}

export function rememberBackend(url) {
  localStorage.setItem(KEY, url.replace(/\/+$/, ''))
}

async function call(backend, path, options) {
  const res = await fetch(`${backend.replace(/\/+$/, '')}${path}`, options)
  const text = await res.text()
  let data
  try {
    data = JSON.parse(text)
  } catch {
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`)
  }
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`)
  return data
}

export const getHealth = (backend) => call(backend, '/api/health')
export const getMeta = (backend) => call(backend, '/api/meta')

export function getSamples(backend, { split, q, type, finding, limit = 60 }) {
  const params = new URLSearchParams({ split, limit: String(limit) })
  if (q) params.set('q', q)
  if (type) params.set('type', type)
  if (finding) params.set('finding', finding)
  return call(backend, `/api/samples?${params}`)
}

export const imageUrl = (backend, imageId) =>
  `${backend.replace(/\/+$/, '')}/api/image?image_id=${encodeURIComponent(imageId)}`

// `body` is either { sample_id, question? } for a dataset record, or
// { image_b64, question } for an X-ray the user dropped in.
export const predict = (backend, body) =>
  call(backend, '/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
