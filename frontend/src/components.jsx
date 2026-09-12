import { useState } from 'react'
import { imageUrl } from './api'

export const pct = (v, d = 1) => (v == null ? '--' : (100 * v).toFixed(d) + ' %')
export const num = (v, d = 3) =>
  v == null || Number.isNaN(Number(v)) ? '--' : Number(v).toFixed(d)
export const shortTs = (ts) => (ts ? `${ts.slice(4, 8)}.${ts.slice(9, 13)}` : '--')
export const niceTs = (ts) =>
  ts ? `${ts.slice(0, 4)}-${ts.slice(4, 6)}-${ts.slice(6, 8)} ${ts.slice(9, 11)}:${ts.slice(11, 13)}` : '--'

export function Panel({ title, meta, children, caption, legend }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        {meta && <span className="meta">{meta}</span>}
      </div>
      <div className="panel-body">
        {children}
        {caption && <div className="caption">{caption}</div>}
        {legend && <div className="legend">{legend}</div>}
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

const STAGE_ORDER = [
  'Metadata and geometry',
  'Reference and overlap estimation',
  'Illumination-aware preprocessing',
  'Correspondence',
  'Robust geometric verification',
  'Correspondence selection',
  'Sub-pixel refinement',
  'Metrics',
  'Trust and refusal decision',
]

export function Stages({ stages, running }) {
  const done = new Map((stages || []).map((s) => [s.name, s]))
  const mark = { ok: '✓', warning: '!', failed: '✗', skipped: '–', pending: '·' }
  return (
    <div className="stages">
      {STAGE_ORDER.map((name) => {
        const s = done.get(name)
        const status = s ? s.status : 'pending'
        return (
          <div className={'stage ' + status} key={name}>
            <span className="mark">{mark[status] || '·'}</span>
            <div>
              <div className="nm">{name}</div>
              {s?.note && <div className="nt">{s.note}</div>}
              {!s && running && <div className="nt">waiting…</div>}
            </div>
            <span className="tm">{s ? s.seconds.toFixed(3) + ' s' : ''}</span>
          </div>
        )
      })}
    </div>
  )
}

export function Verdict({ status, reasons, confidence, warnings, calibrated }) {
  if (!status) return null
  const title = {
    accepted: 'Registration accepted',
    warning: 'Registration accepted with warnings',
    refused: 'Registration refused',
    error: 'Pipeline error',
  }[status] || status
  return (
    <div className={'verdict ' + status}>
      <h4>{title}</h4>
      <ul>
        {(reasons || []).map((r, i) => (
          <li key={i}>{r}</li>
        ))}
        {(warnings || []).map((w, i) => (
          <li key={'w' + i} style={{ color: 'var(--warning)' }}>
            {w}
          </li>
        ))}
      </ul>
      <div className="conf">
        confidence {num(confidence, 3)} — a summary of the numbers above, not a calibrated
        probability{calibrated === false ? '; decision thresholds are hand-picked, not yet calibrated' : ''}
      </div>
    </div>
  )
}

export function ImageTabs({ runId, tabs }) {
  const avail = tabs.filter((t) => t.available !== false)
  const [active, setActive] = useState(avail[0]?.key)
  const cur = avail.find((t) => t.key === active) || avail[0]
  if (!cur) return null
  return (
    <>
      <div className="tabs">
        {avail.map((t) => (
          <button key={t.key} className="tab" aria-selected={t.key === cur.key}
                  onClick={() => setActive(t.key)}>
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

export function Flow({ options, maskSummary }) {
  const o = options
  const steps = [
    { name: 'INPUT', desc: 'Two Chandrayaan-2 OHRC Level-2 products, memory-mapped' },
    {
      name: 'METADATA / GEOMETRY',
      desc: 'PDS4 label, geometry lattice, per-line Sun series from .spm',
    },
    {
      name: 'REFERENCE / OVERLAP',
      desc: 'Footprints in polar stereographic metres; coarse offset between products measured',
    },
    {
      name: 'ILLUMINATION-AWARE PREPROCESSING',
      desc: o
        ? `${o.mask_mode === 'none' ? 'no mask' : o.mask_mode + ' mask'}${
            o.terrain_model && o.terrain_model !== 'none' ? ' · terrain ' + o.terrain_model : ''
          }${o.preprocess_enabled ? ' · flatten + CLAHE' : ' · no enhancement'}`
        : 'usability mask, flattening, CLAHE, GSD harmonisation',
      state: o && o.mask_mode === 'none' ? 'off' : 'ok',
    },
    {
      name: 'CORRESPONDENCE',
      desc: o ? `${o.matcher} · ratio ${o.ratio}${o.mutual_check ? ' · mutual check' : ''}` : 'detector, descriptor, ratio test',
    },
    {
      name: 'ROBUST VERIFICATION',
      desc: o
        ? o.verify_enabled
          ? `MAGSAC++/RANSAC · ${o.model_type} · ${o.ransac_threshold} px`
          : 'disabled for this run'
        : 'robust estimator',
      state: o && !o.verify_enabled ? 'off' : 'ok',
    },
    {
      name: 'CORRESPONDENCE SELECTION',
      desc: o
        ? o.selection_mode === 'grid'
          ? `${o.grid}×${o.grid} grid quota, max ${o.per_cell}/cell`
          : o.selection_mode === 'topk'
            ? 'top-K by confidence (same budget)'
            : 'all verified inliers'
        : 'grid quota and minimum separation',
      state: o && o.selection_mode !== 'grid' ? 'off' : 'ok',
    },
    {
      name: 'SUB-PIXEL REFINEMENT',
      desc: o ? (o.subpixel_enabled ? 'local NCC + paraboloid peak fit' : 'disabled for this run') : 'local NCC',
      state: o && !o.subpixel_enabled ? 'off' : 'ok',
    },
    { name: 'TRUST / REFUSAL DECISION', desc: 'Transparent rule-based accept / warn / refuse' },
    { name: 'RESULT', desc: 'Transform, diagnostics and the reasons behind the decision' },
    {
      name: 'PREDICTED SHADOW FROM A REAL DEM',
      desc: 'Interface and mask plumbing implemented; no DEM is present in this repository',
      state: 'plan',
    },
    {
      name: 'PHOTOMETRIC MODEL / IIRS CROSS-MODAL',
      desc: 'Not implemented',
      state: 'plan',
    },
  ]
  return (
    <div className="flow">
      {steps.map((s, i) => (
        <div key={s.name}>
          <div className={'flow-step' + (s.state === 'plan' ? ' exp' : '')}>
            <span style={{ color: s.state === 'off' ? 'var(--text-faint)' : undefined }}>{s.name}</span>
            <span className="row" style={{ justifyContent: 'flex-end' }}>
              <span className="desc">{s.desc}</span>
              {s.state === 'plan' && <span className="badge plan">PLANNED</span>}
              {s.state === 'off' && <span className="badge off">OFF</span>}
            </span>
          </div>
          {i < steps.length - 1 && <div className="flow-arrow">&#9660;</div>}
        </div>
      ))}
      {maskSummary?.is_synthetic_terrain && (
        <div className="note synth" style={{ marginTop: 12 }}>
          <strong>This run used a synthetic terrain model.</strong> The predicted-shadow mask
          exercises the pipeline end to end; it says nothing about the real surface. Real
          ray-casting is pending an external DEM.
        </div>
      )}
    </div>
  )
}

export function KV({ items }) {
  return (
    <dl className="kv">
      {items.filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => (
        <>
          <dt key={k + 'k'}>{k}</dt>
          <dd key={k + 'v'}>{v}</dd>
        </>
      ))}
    </dl>
  )
}
