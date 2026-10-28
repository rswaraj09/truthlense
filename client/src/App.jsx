import { useCallback, useEffect, useRef, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'
const MAX_BYTES = 25 * 1024 * 1024
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

const VERDICTS = {
  authentic: { title: 'Likely a real photo', tone: 'real' },
  ai_generated: { title: 'Likely AI-generated', tone: 'ai' },
  inconclusive: { title: "Can't tell", tone: 'mixed' },
}

// How sure the model is about its verdict (not the raw AI probability).
function confidenceText(prediction, score) {
  if (prediction === 'inconclusive') return 'Low confidence'
  return `${Math.round(Math.max(score, 1 - score) * 100)}% confident`
}

function useServiceStatus() {
  const [status, setStatus] = useState({ state: 'checking' })
  useEffect(() => {
    let alive = true
    axios
      .get(`${API}/api/health`, { timeout: 6000 })
      .then(({ data }) => {
        if (!alive) return
        const ai = data.aiService || {}
        if (ai.status === 'ok') {
          setStatus({
            state: ai.model_loaded ? 'ready' : 'untrained',
            accuracy: ai.model_metadata?.val_acc ?? null,
          })
        } else {
          setStatus({ state: 'ai-down' })
        }
      })
      .catch(() => alive && setStatus({ state: 'server-down' }))
    return () => {
      alive = false
    }
  }, [])
  return status
}

const STATUS_TEXT = {
  checking: 'Checking services',
  ready: 'Model ready',
  untrained: 'No trained weights loaded',
  'ai-down': 'AI service is offline',
  'server-down': 'Server is offline',
}

function ScoreMeter({ score, tone }) {
  const pct = Math.round(score * 100)
  return (
    <div className="meter" role="img" aria-label={`AI probability ${pct} percent`}>
      <div className="meter-track">
        <span className="meter-zone zone-real" />
        <span className="meter-zone zone-mixed" />
        <span className="meter-zone zone-ai" />
        <span className={`meter-pin pin-${tone}`} style={{ left: `${pct}%` }}>
          <b>{pct}%</b>
        </span>
      </div>
      <div className="meter-scale">
        <span style={{ left: '0%' }}>Real</span>
        <span style={{ left: '35%' }}>35</span>
        <span style={{ left: '65%' }}>65</span>
        <span style={{ left: '100%' }}>AI</span>
      </div>
    </div>
  )
}

function Readout({ result, previewUrl, accuracy }) {
  const verdict = VERDICTS[result.prediction] || VERDICTS.inconclusive
  const m = result.forensicMetrics || {}
  const untrained = m.modelWeightsActive === false
  return (
    <section className={`readout tone-${verdict.tone}`} aria-live="polite">
      {untrained && (
        <p className="notice">
          No trained weights are loaded, so this score comes from a placeholder heuristic. Train the
          model and restart the AI service for real results.
        </p>
      )}
      <h2>{verdict.title}</h2>
      <p className="verdict-sub">{confidenceText(result.prediction, result.score)} in this result.</p>
      <ScoreMeter score={result.score} tone={verdict.tone} />
      <dl className="facts">
        <div>
          <dt>Confidence</dt>
          <dd>{confidenceText(result.prediction, result.score).replace(' confident', '')}</dd>
        </div>
        <div title="Accuracy on held-out test images during training">
          <dt>Model accuracy</dt>
          <dd>{accuracy != null ? `${Math.round(accuracy * 100)}%` : 'Not available'}</dd>
        </div>
        <div>
          <dt>Analysis time</dt>
          <dd>{result.processingTimeMs} ms</dd>
        </div>
      </dl>
      <p className="disclaimer">{result.limitationDisclaimer}</p>
      <img className="sr-only" src={previewUrl} alt="" />
    </section>
  )
}

export default function App() {
  const status = useServiceStatus()
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [history, setHistory] = useState([])
  const inputRef = useRef(null)

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl])

  const choose = useCallback((f) => {
    if (!f) return
    if (!ACCEPTED.includes(f.type)) {
      setError('Use a JPG, PNG or WEBP image.')
      return
    }
    if (f.size > MAX_BYTES) {
      setError('That image is over 25 MB. Choose a smaller one.')
      return
    }
    setError('')
    setResult(null)
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
  }, [])

  const analyze = async () => {
    if (!file || loading) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const { data } = await axios.post(`${API}/api/detect/image`, body, { timeout: 90000 })
      setResult(data)
      setHistory((h) =>
        [{ id: Date.now(), name: file.name, url: previewUrl, prediction: data.prediction, score: data.score }, ...h].slice(0, 5),
      )
    } catch (err) {
      const apiMsg = err.response?.data?.message
      setError(
        apiMsg ||
          (err.code === 'ERR_NETWORK'
            ? `Can't reach the server at ${API}. Start it with "npm run dev" in the server folder.`
            : 'Analysis failed. Try again.'),
      )
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreviewUrl('')
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const tone = result ? (VERDICTS[result.prediction] || VERDICTS.inconclusive).tone : 'idle'

  return (
    <div className="app">
      <header className="bar">
        <h1 className="wordmark">TruthLens</h1>
        <p className={`status status-${status.state}`}>
          <span className="dot" />
          {STATUS_TEXT[status.state]}
          {status.state === 'ready' && status.accuracy != null && ` · ${Math.round(status.accuracy * 100)}% accuracy`}
        </p>
      </header>

      <main className="workspace">
        <div className="stage-col">
          <div
            className={`stage stage-${tone} ${dragging ? 'is-dragging' : ''} ${loading ? 'is-scanning' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              choose(e.dataTransfer.files?.[0])
            }}
          >
            <i className="crop tl" />
            <i className="crop tr" />
            <i className="crop bl" />
            <i className="crop br" />
            {previewUrl ? (
              <img className="photo" src={previewUrl} alt="Image selected for analysis" />
            ) : (
              <button type="button" className="drop" onClick={() => inputRef.current?.click()}>
                <span className="drop-title">Drop a photo here</span>
                <span className="drop-sub">or click to browse. JPG, PNG or WEBP, up to 25 MB.</span>
              </button>
            )}
            {loading && <span className="sweep" />}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(',')}
            hidden
            onChange={(e) => choose(e.target.files?.[0])}
          />

          <div className="actions">
            <button type="button" className="btn primary" onClick={analyze} disabled={!file || loading}>
              {loading ? 'Analyzing…' : 'Analyze image'}
            </button>
            <button type="button" className="btn" onClick={() => (file ? reset() : inputRef.current?.click())}>
              {file ? 'Clear image' : 'Choose image'}
            </button>
          </div>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </div>

        <aside className="side">
          {result ? (
            <Readout result={result} previewUrl={previewUrl} accuracy={status.accuracy} />
          ) : (
            <section className="readout empty">
              <h2>Is this photo real?</h2>
              <p>
                Add an image and TruthLens estimates the chance it was generated by an AI model, using a
                neural network trained on real and AI-made pictures.
              </p>
            </section>
          )}

          {history.length > 0 && (
            <section className="history">
              <h3>This session</h3>
              <ul>
                {history.map((h) => {
                  const v = VERDICTS[h.prediction] || VERDICTS.inconclusive
                  return (
                    <li key={h.id}>
                      <img src={h.url} alt="" />
                      <span className="h-name">{h.name}</span>
                      <span className={`h-score tone-${v.tone}`}>
                        {v.title} · {confidenceText(h.prediction, h.score)}
                      </span>
                    </li>
                  )
                })}
              </ul>
            </section>
          )}
        </aside>
      </main>
    </div>
  )
}