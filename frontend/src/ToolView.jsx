/* The Phase 7 registration tool: pick any two files, run, inspect, download.
 *
 * Everything shown here is read back out of the job's own artifact set. The
 * metrics table renders metrics.json, the match lines come from the tie points
 * the run actually kept, and the before/after wipe compares the reference
 * against the registered raster the run wrote. Nothing on this screen is
 * precomputed or illustrative, and any input that is not an archive product is
 * labelled a fixture wherever it appears.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import { Panel, num, pct } from './components'

const MODELS = ['auto', 'similarity', 'affine', 'homography']
const SIDES = [768, 1024, 2048, 4096]

const fmtBytes = (b) =>
  b == null ? '--'
    : b > 1e9 ? (b / 1e9).toFixed(1) + ' GB'
      : b > 1e6 ? (b / 1e6).toFixed(1) + ' MB'
        : b > 1e3 ? (b / 1e3).toFixed(0) + ' kB' : b + ' B'

function FilePicker({ label, files, value, onChange, filter, onFilter }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input className="tool-search" placeholder="filter by name or instrument"
             value={filter} onChange={(e) => onFilter(e.target.value)} />
      <div className="tool-files">
        {files.length === 0 && <div className="tool-empty">no matching files</div>}
        {files.map((f) => (
          <button key={f.path} className="tool-file" aria-selected={value === f.path}
                  onClick={() => onChange(f.path)} title={f.path}>
            <span className={'chip ' + (f.kind === 'product' ? 'real' : 'synth')}>
              {f.kind === 'product' ? 'PRODUCT' : 'FIXTURE'}
            </span>
            <span className="nm">{f.name}</span>
            <span className="tg">{f.instrument} · {fmtBytes(f.bytes)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/* Source and reference side by side with the kept tie points drawn between
 * them. Inliers in green, rejected candidates in faint red - showing only the
 * survivors would hide how much the verification stage threw away. */
function MatchLines({ jobId, preview, showOutliers }) {
  const ref = useRef(null)
  const [imgs, setImgs] = useState(null)

  useEffect(() => {
    let dead = false
    const load = (n) => new Promise((res, rej) => {
      const im = new Image()
      im.onload = () => res(im)
      im.onerror = rej
      im.src = api.toolFileUrl(jobId, n)
    })
    Promise.all([load('source.png'), load('reference.png')])
      .then(([a, b]) => { if (!dead) setImgs([a, b]) })
      .catch(() => { if (!dead) setImgs(null) })
    return () => { dead = true }
  }, [jobId])

  useEffect(() => {
    if (!imgs || !preview || !ref.current) return
    const [sa, sb] = imgs
    const cv = ref.current
    const GAP = 14
    // Draw at a fixed working width so line coordinates stay exact under CSS scaling.
    const H = Math.max(sa.height, sb.height)
    cv.width = sa.width + GAP + sb.width
    cv.height = H
    const g = cv.getContext('2d')
    g.fillStyle = '#0c0d10'
    g.fillRect(0, 0, cv.width, cv.height)
    g.drawImage(sa, 0, 0)
    g.drawImage(sb, sa.width + GAP, 0)

    const sx = sa.width / preview.source.width
    const sy = sa.height / preview.source.height
    const rx = sb.width / preview.reference.width
    const ry = sb.height / preview.reference.height
    const off = sa.width + GAP

    for (const m of preview.matches) {
      if (!m.inlier && !showOutliers) continue
      g.strokeStyle = m.inlier ? 'rgba(90,220,140,0.55)' : 'rgba(230,103,103,0.28)'
      g.lineWidth = m.inlier ? 1.1 : 0.8
      g.beginPath()
      g.moveTo(m.sx * sx, m.sy * sy)
      g.lineTo(off + m.rx * rx, m.ry * ry)
      g.stroke()
    }
    for (const m of preview.matches) {
      if (!m.inlier && !showOutliers) continue
      g.fillStyle = m.inlier ? '#5adc8c' : 'rgba(230,103,103,0.5)'
      g.beginPath(); g.arc(m.sx * sx, m.sy * sy, 1.6, 0, 7); g.fill()
      g.beginPath(); g.arc(off + m.rx * rx, m.ry * ry, 1.6, 0, 7); g.fill()
    }
  }, [imgs, preview, showOutliers])

  if (!preview) return <div className="ph">no tie points to draw</div>
  return <canvas ref={ref} className="tool-canvas" />
}

/* Reference vs registered under a wipe. Same frame, same pixel grid, so the
 * seam is the alignment - which is the only honest way to show it. */
function BeforeAfter({ jobId }) {
  const [x, setX] = useState(50)
  return (
    <div>
      <div className="tool-wipe">
        <img src={api.toolFileUrl(jobId, 'reference.png')} alt="reference" />
        <img src={api.toolFileUrl(jobId, 'registered.png')} alt="registered"
             className="over" style={{ clipPath: `inset(0 0 0 ${x}%)` }} />
        <div className="seam" style={{ left: x + '%' }} />
        <span className="tool-tag left">REFERENCE</span>
        <span className="tool-tag right">REGISTERED SOURCE</span>
      </div>
      <input className="tool-range" type="range" min="0" max="100" value={x}
             onChange={(e) => setX(+e.target.value)} />
    </div>
  )
}

function Row({ k, v, mono = true, tone }) {
  return (
    <tr>
      <th>{k}</th>
      <td className={(mono ? 'mono ' : '') + (tone || '')}>{v}</td>
    </tr>
  )
}

function MetricsTable({ m }) {
  const a = m.accuracy || {}
  const d = m.distribution || {}
  const p = m.pair || {}
  const il = m.illumination || {}
  const tone = m.status === 'pass' ? 'ok' : m.status === 'warning' ? 'warn' : 'bad'
  return (
    <table className="tool-table">
      <tbody>
        <Row k="status" v={String(m.status).toUpperCase()} tone={tone} />
        {m.reason && <Row k="reason" v={m.reason} mono={false} />}
        <Row k="method used" v={m.method_used} />
        <Row k="model" v={m.model} />
        <Row k="candidates" v={m.matches?.candidates} />
        <Row k="inliers" v={m.matches?.inliers} />
        <Row k="inlier ratio" v={pct(m.matches?.inlier_ratio)} />
        <Row k="held-out points" v={a.held_out_n} />
        {/* The working grid is internal to the run, so a residual quoted only in
            its pixels says nothing about either input. All four, always. */}
        <Row k="RMSE (source px)" v={num(a.rmse_source_px, 3)}
             tone={a.subpixel ? 'ok' : ''} />
        <Row k="RMSE (reference px)" v={num(a.rmse_reference_px, 3)} />
        <Row k="RMSE (m)" v={a.rmse_m == null
          ? <span title="no scale on the reference; metres would be invented">null</span>
          : num(a.rmse_m, 3)} />
        <Row k="RMSE (working-grid px)" v={num(a.rmse_working_px, 4)} />
        <Row k="median / p90 (working px)"
             v={`${num(a.held_out_median_px, 3)} / ${num(a.held_out_p90_px, 3)}`} />
        <Row k="sub-pixel (source)" v={a.subpixel == null ? 'unknown — no scale'
          : String(a.subpixel)} tone={a.subpixel ? 'ok' : ''} />
        {a.subpixel_attainable === false && (
          <Row k="sub-pixel floor" mono={false}
               v={`1 reference px = ${num(a.subpixel_floor_source_px, 2)} source px, so sub-source-pixel is not reachable against this reference`} />
        )}
        <Row k="refinement" v={a.subpixel_method} />
        {a.ecc?.attempted && (
          <Row k="ECC polish" v={a.ecc.adopted
            ? `adopted, ${num(a.ecc.rmse_px_before, 4)} -> ${num(a.rmse_px, 4)} px`
            : (a.ecc.note || 'not adopted')} mono={false} />
        )}
        {m.fine_stage?.adopted && (
          <Row k="fine stage" mono={false}
               v={`${m.fine_stage.method} at native resolution, ${m.fine_stage.points} points from ${m.fine_stage.windows_used} windows`} />
        )}
        {m.fine_stage?.attempted && !m.fine_stage?.adopted && (
          <Row k="fine stage" mono={false} v={m.fine_stage.note || 'not adopted'} />
        )}
        <Row k="coverage" v={`${pct(d.coverage_fraction)} of ${d.eligible_cells} eligible cells`} />
        <Row k="dispersion" v={num(d.dispersion, 3)} />
        <Row k="extrapolated area" v={pct(d.extrapolation_fraction)} />
        <Row k="Δ Sun azimuth" v={il.delta_sun_azimuth_deg == null ? 'unknown'
          : num(il.delta_sun_azimuth_deg, 2) + '°'} />
        <Row k="cross-correlation" v={num(p.cross_correlation, 4)} />
        <Row k="scale ratio" v={num(p.scale_ratio, 4)} />
        <Row k="overlap" v={pct(p.overlap_fraction)} />
        <Row k="runtime" v={num(m.runtime_s, 2) + ' s'} />
      </tbody>
    </table>
  )
}

export default function ToolView({ view, setView }) {
  const [files, setFiles] = useState([])
  const [err, setErr] = useState(null)
  const [src, setSrc] = useState(null)
  const [ref_, setRef] = useState(null)
  const [fa, setFa] = useState('')
  const [fb, setFb] = useState('')

  const [model, setModel] = useState('auto')
  const [maxSide, setMaxSide] = useState(1024)
  const [grid, setGrid] = useState(8)
  const [segments, setSegments] = useState(0)
  const [subpixel, setSubpixel] = useState(true)

  const [job, setJob] = useState(null)
  const [preview, setPreview] = useState(null)
  const [showOut, setShowOut] = useState(false)
  const [tab, setTab] = useState('matches')

  useEffect(() => {
    api.toolFiles().then((d) => setFiles(d.files)).catch((e) => setErr(String(e)))
  }, [])

  const filtered = (all, q) => {
    const s = q.trim().toLowerCase()
    if (!s) return all
    return all.filter((f) =>
      f.name.toLowerCase().includes(s) || (f.instrument || '').toLowerCase().includes(s))
  }
  const listA = useMemo(() => filtered(files, fa), [files, fa])
  const listB = useMemo(() => filtered(files, fb), [files, fb])

  const byPath = useMemo(() => Object.fromEntries(files.map((f) => [f.path, f])), [files])
  const usesFixture = [src, ref_].some((p) => p && byPath[p]?.kind === 'fixture')

  // Poll while a run is in flight. The log the server returns is the pipeline's
  // own, so the stages that appear are stages that actually happened.
  useEffect(() => {
    if (!job || job.state !== 'running') return
    const t = setInterval(() => {
      api.toolJob(job.id).then((d) => {
        setJob(d)
        if (d.state === 'done' && d.job_dir_id) {
          api.toolFile(d.id, 'preview.json').then(setPreview).catch(() => setPreview(null))
        }
      }).catch((e) => setErr(String(e)))
    }, 700)
    return () => clearInterval(t)
    // keyed on id and state only: re-running on every log append would restart
    // the interval on each tick and the poll would never settle
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, job?.state])

  const run = useCallback(() => {
    setErr(null); setPreview(null); setTab('matches')
    api.toolRegister({
      source: src, reference: ref_, model, max_side: maxSide,
      grid, segments, subpixel,
    })
      .then((d) => setJob({ id: d.job, state: 'running', log: [], stage: 'starting' }))
      .catch((e) => setErr(String(e)))
  }, [src, ref_, model, maxSide, grid, segments, subpixel])

  const m = job?.metrics
  const running = job?.state === 'running'
  const failed = job?.state === 'done' && job.status === 'failed'

  return (
    <div className="app">
      <aside className="rail">
        <div className="masthead">
          <h1 className="wordmark">SELENO<span>.</span></h1>
          <p className="tagline">
            Register any two rasters. No ISRO geometry assumed; whatever is
            missing is reported rather than guessed.
          </p>
        </div>

        <div className="rail-section">
          <div className="seg">
            <button aria-selected={view === 'tool'} onClick={() => setView('tool')}>
              Registration tool
            </button>
            <button aria-selected={view === 'study'} onClick={() => setView('study')}>
              Phase 2 study
            </button>
          </div>
        </div>

        <div className="rail-section">
          <FilePicker label="Source" files={listA} value={src} onChange={setSrc}
                      filter={fa} onFilter={setFa} />
          <FilePicker label="Reference" files={listB} value={ref_} onChange={setRef}
                      filter={fb} onFilter={setFb} />
        </div>

        <div className="rail-section">
          <div className="field">
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Working grid (max side)</label>
            <select value={maxSide} onChange={(e) => setMaxSide(+e.target.value)}>
              {SIDES.map((x) => <option key={x} value={x}>{x} px</option>)}
            </select>
          </div>
          <div className="field">
            <label>Coverage grid</label>
            <select value={grid} onChange={(e) => setGrid(+e.target.value)}>
              {[6, 8, 12, 16].map((x) => <option key={x} value={x}>{x} × {x}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Segments (long strips)</label>
            <select value={segments} onChange={(e) => setSegments(+e.target.value)}>
              {[0, 3, 6, 10].map((x) => (
                <option key={x} value={x}>{x === 0 ? 'single transform' : x + ' bands'}</option>
              ))}
            </select>
          </div>
          <label className="check">
            <input type="checkbox" checked={subpixel}
                   onChange={(e) => setSubpixel(e.target.checked)} />
            <span>ECC sub-pixel polish
              <span className="hint">adopted only if it beats the held-out RMSE</span>
            </span>
          </label>
          <button className="btn" style={{ width: '100%', marginTop: 10 }}
                  disabled={!src || !ref_ || running} onClick={run}>
            {running ? 'Running…' : 'Register'}
          </button>
          {src && ref_ && src === ref_ && (
            <div className="note">same file both sides — expect a near-identity result</div>
          )}
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <div>
            <h2>Registration tool</h2>
            <div className="sub">
              {src && ref_
                ? <><code>{byPath[src]?.name}</code> onto <code>{byPath[ref_]?.name}</code></>
                : 'pick a source and a reference'}
            </div>
          </div>
          {job?.state === 'done' && job.job_dir_id && (
            <a className="btn ghost" href={api.toolDownloadUrl(job.id)}>Download outputs</a>
          )}
        </div>

        {usesFixture && (
          <div className="note warnbox">
            <strong>Fixture data.</strong> At least one input is a generated
            fixture, not a Chandrayaan-2 or reference-mission product. Numbers
            below describe this synthetic pair only.
          </div>
        )}
        {err && <div className="note bad">{err}</div>}

        {job && (
          <div className="section">
            <h3>Run</h3>
            <div className="tool-log">
              {(job.log || []).map((l, i) => (
                <div key={i} className="tool-logline">
                  <span className="t">{l.t.toFixed(2)}s</span>
                  <span className="l">{l.line}</span>
                </div>
              ))}
              {running && <div className="tool-logline"><span className="t" />
                <span className="l pulse">… {job.stage}</span></div>}
              {job.state === 'crashed' && (
                <div className="tool-logline"><span className="t" />
                  <span className="l bad">tool error: {job.error}</span></div>
              )}
            </div>
          </div>
        )}

        {failed && m && (
          <div className="section">
            <div className="note bad">
              <strong>Registration failed — {m.reason}</strong>
              <div style={{ marginTop: 4 }}>{m.message}</div>
              <div className="hint" style={{ marginTop: 6 }}>
                Declared failure codes: {(job.failure_codes || []).join(', ')}
              </div>
            </div>
          </div>
        )}

        {m && !failed && (
          <>
            <div className="section">
              <h3>Result</h3>
              <div className="grid2">
                <Panel title="Metrics" meta="metrics.json">
                  <MetricsTable m={m} />
                  {(m.notes || []).map((n, i) => (
                    <div key={i} className="caption">{n}</div>
                  ))}
                </Panel>
                <Panel title="Degraded capability"
                       meta={(m.degraded || []).length + ' notes'}>
                  {(m.degraded || []).length === 0
                    ? <div className="caption">Nothing degraded: both inputs carried
                        everything the tool wanted.</div>
                    : <ul className="tool-degraded">
                        {m.degraded.map((x, i) => <li key={i}>{x}</li>)}
                      </ul>}
                  <div className="caption">
                    Every assumption the tool could not verify is listed here
                    rather than silently applied.
                  </div>
                </Panel>
              </div>
            </div>

            <div className="section">
              <div className="seg" style={{ marginBottom: 12 }}>
                <button aria-selected={tab === 'matches'} onClick={() => setTab('matches')}>
                  Match lines
                </button>
                <button aria-selected={tab === 'wipe'} onClick={() => setTab('wipe')}>
                  Before / after
                </button>
                <button aria-selected={tab === 'overlay'} onClick={() => setTab('overlay')}>
                  Composite
                </button>
              </div>

              {tab === 'matches' && (
                <Panel title="Tie points"
                       meta={preview ? `${preview.n_shown} of ${preview.n_total} drawn` : ''}
                       caption="Green: inliers kept by geometric verification. Red: candidates it rejected.">
                  <label className="check">
                    <input type="checkbox" checked={showOut}
                           onChange={(e) => setShowOut(e.target.checked)} />
                    <span>show rejected candidates</span>
                  </label>
                  <MatchLines jobId={job.id} preview={preview} showOutliers={showOut} />
                </Panel>
              )}

              {tab === 'wipe' && (
                <Panel title="Reference vs registered source" meta="registered.png"
                       caption="Drag the handle. Both panels are the same pixel grid, so features crossing the seam without stepping are aligned.">
                  <BeforeAfter jobId={job.id} />
                </Panel>
              )}

              {tab === 'overlay' && (
                <Panel title="Composite" meta="overlay.png"
                       caption="Top: source and reference with tie lines. Bottom: checkerboard of reference and registered source.">
                  <img src={api.toolFileUrl(job.id, 'overlay.png')} alt="overlay" />
                </Panel>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  )
}
