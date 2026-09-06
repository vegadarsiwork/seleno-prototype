import { useState } from 'react'
import { imageUrl } from './api'

export const pct = (v) => (v == null ? '--' : (100 * v).toFixed(1) + ' %')
export const num = (v, d = 3) => (v == null || Number.isNaN(v) ? '--' : Number(v).toFixed(d))

export function Panel({ title, meta, children, caption }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        {meta && <span className="meta">{meta}</span>}
      </div>
      <div className="panel-body">
        {children}
        {caption && <div className="caption">{caption}</div>}
      </div>
    </div>
  )
}

export function Metric({ label, value, unit, sub, tone }) {
  return (
    <div className={'metric' + (tone ? ' ' + tone : '')}>
      <div className="k">{label}</div>
      <div className="v">
        {value}
        {unit && <small>{unit}</small>}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

export function Stages({ stages, running }) {
  const PLANNED = [
    'Preprocessing',
    'Feature extraction and candidate matching',
    'Geometric verification',
    'Spatial match selection',
    'Registration',
    'Metrics',
  ]
  const done = new Map((stages || []).map((s) => [s.name, s]))
  return (
    <div className="stages">
      {PLANNED.map((name) => {
        const s = done.get(name)
        const status = s ? s.status : running ? 'pending' : 'pending'
        const mark = { ok: '✓', failed: '✗', skipped: '–', pending: '·' }[status]
        return (
          <div className={'stage ' + status} key={name}>
            <span className="mark">{mark}</span>
            <div>
              <div className="stage-name">{name}</div>
              {s?.note && <div className="stage-note">{s.note}</div>}
            </div>
            <span className="stage-time">{s ? s.seconds.toFixed(3) + ' s' : ''}</span>
          </div>
        )
      })}
    </div>
  )
}

export function Verdict({ verdict, warnings }) {
  if (!verdict) return null
  const pass = verdict.registered
  return (
    <div className={'verdict ' + (pass ? 'pass' : 'fail')}>
      <h4>{pass ? 'Registration accepted' : 'Registration rejected'}</h4>
      <ul>
        {verdict.reasons.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
        {(warnings || []).map((w, i) => (
          <li key={'w' + i} style={{ color: 'var(--accent)' }}>
            plausibility check: {w}
          </li>
        ))}
      </ul>
      {verdict.expected_outcome && (
        <div className="expect">Expected for this pair: {verdict.expected_outcome}</div>
      )}
    </div>
  )
}

export function ImageTabs({ runId, tabs }) {
  const [active, setActive] = useState(tabs[0].key)
  const cur = tabs.find((t) => t.key === active) || tabs[0]
  return (
    <>
      <div className="tabs">
        {tabs.map((t) => (
          <button
            key={t.key}
            className="tab"
            aria-selected={t.key === active}
            onClick={() => setActive(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="panel">
        <div className="panel-body">
          <img src={imageUrl(runId, cur.key)} alt={cur.label} />
          <div className="caption">{cur.caption}</div>
          {cur.legend && <div className="legend">{cur.legend}</div>}
        </div>
      </div>
    </>
  )
}

export function Flow({ result }) {
  const opts = result?.options
  const steps = [
    {
      name: 'INPUT',
      desc: 'Two Chandrayaan-2 OHRC products as delivered by ISRO',
      state: 'ok',
    },
    {
      name: 'PREPROCESSING',
      desc: opts
        ? opts.preprocess_enabled
          ? 'Illumination flattening, CLAHE, GSD harmonisation'
          : 'disabled for this run'
        : 'Illumination flattening, CLAHE, GSD harmonisation',
      state: opts && !opts.preprocess_enabled ? 'off' : 'ok',
    },
    {
      name: 'MULTI-SCALE CORRESPONDENCE',
      desc: opts
        ? `${opts.matcher} · ratio ${opts.ratio}${opts.mutual_check ? ' · mutual check' : ''}`
        : 'detector, descriptor, ratio test',
      state: 'ok',
    },
    {
      name: 'GEOMETRIC VERIFICATION',
      desc: opts
        ? opts.verify_enabled
          ? `MAGSAC++/RANSAC · ${opts.model_type} · ${opts.ransac_threshold} px`
          : 'disabled for this run'
        : 'robust estimator',
      state: opts && !opts.verify_enabled ? 'off' : 'ok',
    },
    {
      name: 'SPATIAL MATCH SELECTION',
      desc: opts
        ? opts.spatial_enabled
          ? `${opts.grid}×${opts.grid} grid · max ${opts.per_cell}/cell · min separation`
          : 'disabled — top-K by confidence instead'
        : 'grid quota and minimum separation',
      state: opts && !opts.spatial_enabled ? 'off' : 'ok',
    },
    { name: 'REGISTRATION', desc: 'Least-squares refit, warp, overlay, difference', state: 'ok' },
    { name: 'METRICS', desc: 'Counts, RMSE, coverage, runtime, ground truth where known', state: 'ok' },
    {
      name: 'PHOTOMETRIC / CROSS-MODAL MATCHING (IIRS)',
      desc: 'Not implemented. Requires IIRS products and a sensor model.',
      state: 'planned',
    },
    {
      name: 'LEARNED MATCHER (LoFTR / SuperGlue)',
      desc: 'Interface implemented; weights not installed in this environment.',
      state: 'experimental',
    },
  ]
  return (
    <div className="flow">
      {steps.map((s, i) => (
        <div key={s.name}>
          <div
            className={'flow-step' + (s.state === 'planned' || s.state === 'experimental' ? ' experimental' : '')}
          >
            <span style={{ color: s.state === 'off' ? 'var(--text-faint)' : undefined }}>
              {s.name}
            </span>
            <span className="row" style={{ justifyContent: 'flex-end' }}>
              <span className="desc">{s.desc}</span>
              {s.state === 'planned' && <span className="badge exp">PLANNED</span>}
              {s.state === 'experimental' && <span className="badge exp">EXPERIMENTAL</span>}
              {s.state === 'off' && <span className="badge">OFF</span>}
            </span>
          </div>
          {i < steps.length - 1 && <div className="flow-arrow">&#9660;</div>}
        </div>
      ))}
    </div>
  )
}
