import { useEffect, useState } from 'react'
import { toolQuality } from './api'
import { Panel, num, pct } from './components'

const pair = (v, d = 3) => v ? v.map((x) => num(x, d)).join(' / ') : '--'
const span = (v, d = 2) => v ? `${num(v[0], d)}–${num(v[1], d)}` : null
const pctSpan = (v) => v ? `${pct(v[0], 0)}–${pct(v[1], 0)}` : null
const pm = (v, d = 1) => v && v[0] != null ? `${num(v[0], d)} ± ${v[1] == null ? '--' : num(v[1], d)}` : '--'

function ErrorTable({ title, groups, unit = 'px', axes = 'xy', axisLabel = 'x / y', digits = 3, nominalMetres }) {
  const raw = groups?.all
  const screened = groups?.screened
  if (!raw) return <p className="caption">{title}: no saved directional measurements.</p>
  const n = (v) => num(v, digits)
  const p = (v) => pair(v, digits)
  const rows = [
    ['Observations', 'n'], ['Invalid predictions', 'invalid_n'],
    [`RMSE (${unit})`, `rmse_${unit}`, n], [`Median (${unit})`, `median_${unit}`, n],
    [`p90 (${unit})`, `p90_${unit}`, n], [`p95 (${unit})`, `p95_${unit}`, n],
    [`Maximum (${unit})`, `max_${unit}`, n],
    [`Mean ${axisLabel} (${unit})`, `bias_${axes}_${unit}`, p],
    [`1σ ${axisLabel} (${unit})`, `std_${axes}_${unit}`, p],
    [`RMS ${axisLabel} (${unit})`, `rmse_${axes}_${unit}`, p],
    [`Robust scatter ${axisLabel} (NMAD, ${unit})`, `nmad_${axes}_${unit}`, p],
  ]
  return <div className="quality-table-wrap">
    <table className="tool-table quality-table">
      <caption>{title}</caption>
      <thead><tr><th scope="col">Metric</th><th scope="col">All held-out</th><th scope="col">Screened</th></tr></thead>
      <tbody>
        {rows.map(([label, key, format = (v) => v ?? '--']) => <tr key={key}>
          <th scope="row">{label}</th><td>{format(raw[key])}</td><td>{format(screened?.[key])}</td>
        </tr>)}
        {raw.thresholds?.map((t, i) => <tr key={t.below_px}>
          <th scope="row">Below {t.below_px} px</th><td>{pct(t.fraction_all)}</td>
          <td>{pct(screened?.thresholds?.[i]?.fraction_all)}</td>
        </tr>)}
        {nominalMetres > 0 && <tr>
          <th scope="row">RMSE × nominal pixel size (projected m, not surface)</th>
          <td>{num(raw.rmse_px == null ? null : raw.rmse_px * nominalMetres, 1)}</td>
          <td>{num(screened?.rmse_px == null ? null : screened.rmse_px * nominalMetres, 1)}</td>
        </tr>}
      </tbody>
    </table>
  </div>
}

function Card({ label, value, unit, interval }) {
  return <div>
    <span>{label}</span>
    <strong>{value} {unit && <small>{unit}</small>}</strong>
    {interval && <em title="95% spatial-block bootstrap interval">95% {interval}</em>}
  </div>
}

function ResidualPlot({ quality }) {
  const [mode, setMode] = useState('magnitude')
  const [showExcluded, setShowExcluded] = useState(true)
  const shape = quality.source_shape
  if (!shape || !quality.residuals?.length) return null
  const hasGround = quality.residuals.some((p) => p.de != null)
  const alongRows = shape[0] >= shape[1]
  const length = Math.max(1, (alongRows ? shape[0] : shape[1]) - 1)
  const rows = quality.residuals.filter((p) => showExcluded || p.screened !== false)
  const ground = mode === 'ground'
  const comps = (p) => ground ? [p.de, p.dn] : [p.dx, p.dy]
  const finite = (p) => comps(p).every(Number.isFinite)
  const valid = rows.filter(finite)
  const bins = mode === 'magnitude' ? (quality.profiles?.along || []) : []
  const floor = ground ? 10 : 1.25
  const limit = Math.max(floor, ...valid.map((p) => mode === 'magnitude'
    ? Math.hypot(p.dx, p.dy) : Math.max(...comps(p).map(Math.abs)))) * 1.08
  const W = 820, H = 290, left = 56, right = 800, top = 20, bottom = 240
  const at = (v) => left + Math.min(Math.max(v, 0), length) / length * (right - left)
  const x = (p) => at(alongRows ? p.y : p.x)
  const y = (v) => mode === 'magnitude' ? bottom - v / limit * (bottom - top)
    : (top + bottom) / 2 - v / limit * (bottom - top) / 2
  const levels = mode === 'magnitude' ? [0, limit / 2, limit] : [-limit, 0, limit]
  const unit = ground ? 'm' : 'source px'
  return <div className="quality-plot">
    <div className="quality-controls">
      <label>Residual display <select value={mode} onChange={(e) => setMode(e.target.value)}>
        <option value="magnitude">Error magnitude with along-track profile</option>
        <option value="components">Signed sample / line errors</option>
        {hasGround && <option value="ground">Signed east / north surface errors</option>}
      </select></label>
      <label className="check"><input type="checkbox" checked={showExcluded}
        onChange={(e) => setShowExcluded(e.target.checked)} />Include points set aside by screening</label>
    </div>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Held-out errors by position along the source image">
      <title>Held-out residuals. Hover over a point for its coordinates and error.</title>
      {levels.map((v) => <g key={v}>
        <line x1={left} x2={right} y1={y(v)} y2={y(v)} className="quality-grid" />
        <text x={left - 8} y={y(v) + 4} textAnchor="end">{num(v, ground ? 0 : 1)}</text>
      </g>)}
      {!ground && <>
        <line x1={left} x2={right} y1={y(1)} y2={y(1)} className="quality-threshold" />
        <text x={right} y={y(1) - 5} textAnchor="end">1 source px</text>
      </>}
      {[0, 25, 50, 75, 100].map((v) => <text key={v}
        x={left + v / 100 * (right - left)} y={bottom + 20} textAnchor="middle">{v}%</text>)}
      <text x={(left + right) / 2} y={H - 5} textAnchor="middle">
        Position along native source {alongRows ? 'lines' : 'samples'}
      </text>
      <text transform="translate(14 140) rotate(-90)" textAnchor="middle">Error ({unit})</text>
      {bins.map((b) => <g key={b.from}>
        {b.n === 0 && <rect x={at(b.from)} width={at(b.to) - at(b.from)} y={top} height={bottom - top}
          className="quality-gap"><title>{`Lines ${num(b.from, 0)}–${num(b.to, 0)}: no held-out observation`}</title></rect>}
        {b.median_px != null && <line x1={at(b.from)} x2={at(b.to)} y1={y(b.median_px)} y2={y(b.median_px)}
          className="quality-bin-median"><title>{`Median ${num(b.median_px)} px over ${b.n} points`}</title></line>}
        {b.p95_px != null && <line x1={at(b.from)} x2={at(b.to)} y1={y(b.p95_px)} y2={y(b.p95_px)}
          className="quality-bin-p95"><title>{`p95 ${num(b.p95_px)} px over ${b.n} points`}</title></line>}
      </g>)}
      {rows.map((p, i) => {
        if (!Number.isFinite(alongRows ? p.y : p.x)) return null
        const label = `Source (${num(p.x, 1)}, ${num(p.y, 1)}); ${Number.isFinite(p.dx)
          ? `dx ${num(p.dx)}, dy ${num(p.dy)} px; magnitude ${num(Math.hypot(p.dx, p.dy))} px`
          : 'invalid prediction'}${p.de != null ? `; east ${num(p.de, 1)}, north ${num(p.dn, 1)} m` : ''}${p.screened === false ? '; set aside by screen' : ''}`
        if (!finite(p)) return <path key={i} d={`M${x(p)-4},${top}l8,8m-8,0l8,-8`} stroke="var(--refused)"><title>{label}</title></path>
        const values = mode === 'magnitude' ? [Math.hypot(p.dx, p.dy)] : comps(p)
        return values.map((v, j) => <circle key={`${i}-${j}`} cx={x(p)} cy={y(v)} r={p.screened === false ? 3.5 : 3}
          fill={p.screened === false ? 'none' : mode !== 'magnitude' && j === 0 ? 'var(--s1)' : 'var(--s4)'}
          stroke={p.screened === false ? 'var(--text-dim)' : 'none'} opacity="0.8"><title>{label}</title></circle>)
      })}
    </svg>
    <p className="caption">{mode === 'components' ? 'Blue: sample error. Amber: line error.'
      : ground ? 'Blue: east. Amber: north. Metres on the reference datum.'
      : 'Amber: error magnitude. Green: median per along-track bin. Red dashes: p95 per bin (8+ points). Hatched: no held-out observation.'} Hollow circles: screened out. Red crosses: invalid predictions. Gaps have no measured accuracy; points are matcher observations.</p>
  </div>
}

function AcrossTrack({ profiles }) {
  if (!profiles?.across?.length) return null
  const axis = profiles.across_axis
  return <div className="quality-table-wrap">
    <table className="tool-table quality-table">
      <caption>Across-track bins (native source {axis}s)</caption>
      <thead><tr><th scope="col">{axis === 'sample' ? 'Detector samples' : 'Lines'}</th><th scope="col">n</th>
        <th scope="col">Median (px)</th><th scope="col">Mean sample / line (px)</th></tr></thead>
      <tbody>{profiles.across.map((b) => <tr key={b.from}>
        <th scope="row">{num(b.from + .5, 0)}–{num(b.to - .5, 0)}</th><td>{b.n}</td>
        <td>{num(b.median_px)}</td><td>{pair(b.bias_xy_px, 2)}</td>
      </tr>)}</tbody>
    </table>
    <p className="caption">A mean that changes sign across the detector is a systematic, position-dependent error a global model cannot remove. Along-track drift: {pair(profiles.drift_xy_px_per_1000, 3)} sample / line px per 1000 {profiles.along_axis}s.</p>
  </div>
}

function Conventions({ rows }) {
  if (!rows?.length) return null
  const guidance = (c) => {
    if (c.id === 'asp') {
      const tone = c.guidance_met === 'under 0.5 px' ? 'ok' : c.guidance_met === 'under 1 px' ? 'warn' : 'bad'
      return <><span className={`quality-flag ${tone}`}>{c.guidance_met}</span>{' '}
        <span className={`quality-flag ${c.count_met ? 'ok' : 'bad'}`}>{c.count_met ? '≥ 12 points' : 'fewer than 12 points'}</span>
        <div className="caption">{c.guidance}</div></>
    }
    if (c.id === 'kaguya') {
      const p = c.published
      return <>Longitude {pm(p.longitude_m)} m, latitude {pm(p.latitude_m)} m over {p.locations} sites at {p.gsd_m} m GSD
        <div className="caption">= {pm(c.published_per_gsd.longitude, 2)} / {pm(c.published_per_gsd.latitude, 2)} GSD</div></>
    }
    return <span className="caption">No published threshold</span>
  }
  const ours = (c) => {
    const o = c.ours
    switch (c.id) {
      case 'asp': return <>Mean {num(o.mean_px, 2)} / median {num(o.median_px, 2)} source px, n {o.count}
        <div className="caption">{o.basis}; all held-out: {num(o.all_held_out_mean_px, 2)} / {num(o.all_held_out_median_px, 2)}</div></>
      case 'isis': return <>Sample {num(o.rmse_sample_px, 2)} / line {num(o.rmse_line_px, 2)} / overall {num(o.rmse_px, 2)} px RMS
        <div className="caption">{o.basis}, n {o.count}</div></>
      case 'kaguya': return <>East {pm(o.east_m)} m, north {pm(o.north_m)} m
        {o.per_source_gsd && <div className="caption">= {pm(o.per_source_gsd.east, 2)} / {pm(o.per_source_gsd.north, 2)} source GSD ({num(o.source_gsd_m, 2)} m); {o.basis}, n {o.count}</div>}</>
      case 'circular_error': return <>CE90 {num(o.ce90_m, 1)} m, CE95 {num(o.ce95_m, 1)} m
        <div className="caption">{o.basis}, n {o.count}</div></>
      case 'sldem': return <>Worst along-track bin median {num(o.worst_median_px, 2)} px ({o.along_axis || 'line'}s {num(o.worst_from + .5, 0)}–{num(o.worst_to - .5, 0)}); best {num(o.best_median_px, 2)} px
        <div className="caption">{o.bins_with_data} of {o.bins} bins measured</div></>
      default: return null
    }
  }
  return <div className="quality-section">
    <h4>Compared with NASA, USGS and JAXA reporting</h4>
    <p className="caption">Each row restates this run's errors in the form an agency publishes. The form matches; the measurement does not, so no row is a pass mark and the published figures are context, not targets.</p>
    <div className="quality-table-wrap">
      <table className="tool-table quality-table quality-conventions">
        <thead><tr><th scope="col">Source</th><th scope="col">They report</th><th scope="col">This run</th>
          <th scope="col">Guidance / published</th><th scope="col">Why it differs</th></tr></thead>
        <tbody>{rows.map((c) => <tr key={c.id}>
          <th scope="row">{c.url ? <a href={c.url} target="_blank" rel="noreferrer">{c.agency}</a> : c.agency}
            <div className="caption">{c.tool}</div></th>
          <td>{c.reports}</td><td>{ours(c)}</td><td>{guidance(c)}</td>
          <td><span className="caption">{c.different}</span></td>
        </tr>)}</tbody>
      </table>
    </div>
  </div>
}

function WarpCheck({ warp }) {
  if (!warp) return null
  if (!warp.available) return <p className="caption">Final-warp check unavailable: {warp.reason}</p>
  const c = warp.local_correction_native_px
  return <div className="quality-support">
    <p>Final warp: folds on <strong>{pct(warp.folded_fraction, 2)}</strong> of the footprint; local area
      {' '}<strong>{num(warp.relative_area.min, 2)}–{num(warp.relative_area.max, 2)}×</strong> and shear up to
      {' '}<strong>{num(warp.relative_shear.max, 2)}:1</strong> relative to the base model
      {warp.nonlinear_terms?.length ? ` (${warp.nonlinear_terms.join(', ')})` : ''}.
      {c && <> Non-global correction median {num(c.median, 1)}, p95 {num(c.p95, 1)}, max {num(c.max, 1)} native source px (approx.).</>}</p>
    {warp.flags?.length ? <ul className="quality-reasons">{warp.flags.map((f) => <li key={f}>{f}</li>)}</ul>
      : <p className="caption">No fold, area or shear flags ({warp.flag_basis}).</p>}
  </div>
}

export default function QualityPanel({ metrics, jobId }) {
  const [loaded, setLoaded] = useState(null)
  useEffect(() => {
    if (metrics.quality) return
    let live = true
    toolQuality(jobId).then((data) => { if (live) setLoaded({ id: jobId, data }) })
      .catch((error) => { if (live) setLoaded({ id: jobId, error: error.message }) })
    return () => { live = false }
  }, [jobId, metrics.quality])
  const quality = metrics.quality || (loaded?.id === jobId ? loaded.data : null)
  const assessment = quality?.acceptance || metrics.acceptance
  const failed = assessment?.passed === false
  const label = failed ? 'Quality failed' : assessment?.independently_verified
    ? 'Independent checks passed' : assessment?.passed ? 'Internal checks passed · independently unverified' : 'Quality unknown'
  const raw = quality?.source?.all
  const screened = quality?.source?.screened
  const groundAll = quality?.ground?.all
  const support = quality?.support
  const ci = quality?.intervals?.source?.all
  const gci = quality?.intervals?.ground?.all
  const distance = support?.distance_to_fit_point
  return <Panel title="Registration quality" meta="held-out evaluation">
    <div className={`quality-status ${failed ? 'bad' : assessment?.independently_verified ? 'ok' : 'warn'}`}>
      <strong>{label}</strong><span>Export completed</span>
    </div>
    {assessment?.reasons?.length > 0 && <ul className="quality-reasons">{assessment.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
    <p className="caption">Errors below describe matcher consistency. Independent controls are required to verify accuracy against a reference.</p>
    {!quality && <p className="caption">{loaded?.id === jobId && loaded.error ? loaded.error : 'Loading saved evaluation…'}</p>}
    {quality && <>
      <div className="quality-cards">
        <Card label="All held-out RMSE" value={num(raw?.rmse_px)} unit="source px" interval={span(ci?.rmse_px)} />
        <Card label="All held-out p95" value={num(raw?.p95_px)} unit="source px" interval={span(ci?.p95_px)} />
        <Card label="Below 1 source pixel" value={pct(raw?.thresholds?.find((t) => t.below_px === 1)?.fraction_all)}
          interval={pctSpan(ci?.below_1_px_fraction)} />
        {groundAll && <Card label="All held-out surface RMSE" value={num(groundAll.rmse_m, 1)} unit="m"
          interval={span(gci?.rmse_m, 1)} />}
        <Card label="Invalid predictions" value={raw?.invalid_n ?? '--'} unit={`of ${raw?.n ?? quality.n}`} />
      </div>
      <p className="caption">{quality.n} held-out observations{screened ? `; ${screened.n} retained and ${quality.excluded_n} set aside by neighbour screening` : ''}. Distance statistics use finite predictions. Threshold percentages count invalid predictions as failures and use strict “below”, not “at or below”.
        {ci?.blocks && ` Intervals: 95% spatial-block bootstrap over ${ci.blocks} held-out cells (${quality.intervals.replicates} replicates); they cover sampling of these observations only, not bias shared with the reference or matcher.`}</p>
      <div className="grid2">
        <ErrorTable title="Native source · backward transfer" groups={quality.source} axisLabel="sample / line" />
        {quality.ground
          ? <ErrorTable title={`Surface · east / north on ${quality.ground_frame?.datum || 'the reference datum'}`}
              groups={quality.ground} unit="m" axes="en" axisLabel="east / north" digits={1} />
          : <ErrorTable title="Native reference · forward transfer" groups={quality.reference}
              nominalMetres={quality.reference_metres_per_pixel} />}
      </div>
      <p className="caption">{quality.ground
        ? `Surface metres map each forward error through the reference's own projection onto its datum (radius ${num(quality.ground_frame?.semi_major_m / 1000, 1)} km), so east–west distances shrink with latitude on a cylindrical map instead of using the nominal pixel size. They are relative to the reference, not absolute lunar positions.`
        : 'Surface metres were not recorded for this run; a new registration against a georeferenced reference measures them. Nominal metres ignore map distortion.'} NMAD is robust scatter around the median; the mean remains separate.</p>
      {quality.ground && <details className="quality-independent"><summary>Native reference · forward transfer (pixels)</summary>
        <ErrorTable title="Native reference · forward transfer" groups={quality.reference} />
      </details>}
      <Conventions rows={quality.conventions} />
      <div className="quality-section">
        <h4>Where the error is</h4>
        <ResidualPlot quality={quality} />
        <AcrossTrack profiles={quality.profiles} />
      </div>
      <div className="quality-support">
        {support ? <p>Inside the fit-point hull: <strong>{pct(support.supported_fraction)}</strong> of valid overlap; <strong>{pct(support.extrapolation_fraction)}</strong> outside. {support.sampled_pixels.toLocaleString()} / {support.valid_pixels.toLocaleString()} valid pixel centres evaluated ({support.sampling}).
          {distance && <> Distance to the nearest fit point: median {num(distance.median, 1)}, p95 {num(distance.p95, 1)}, largest gap radius {num(distance.max, 1)} {distance.unit}.</>}</p>
          : <p>Valid-pixel support was not recorded in this run. A new registration is needed to measure it.</p>}
        <p className="caption">Being inside the hull does not guarantee accuracy between measurements. Legacy acceptance uses the grid-cell estimate: {pct(metrics.distribution?.extrapolation_fraction)} outside support. Quality gates retain their existing thresholds.</p>
      </div>
      <WarpCheck warp={quality.warp} />
      <details className="quality-independent"><summary>Independent reference assessment</summary>
        {quality.independent ? <>
          <p>{quality.independent.kind || 'Reference'} · {quality.independent.passed ? 'Passed' : 'Not passed'}</p>
          <p>RMSE {num(quality.independent.statistics?.rmse_px)} source px; p95 {num(quality.independent.statistics?.p95_px)} source px.</p>
          <p>Reference: {quality.independent.provenance?.reference || 'Unavailable'}</p>
          <p>Declared reference uncertainty: {num(quality.independent.provenance?.uncertainty_source_px)} source px.</p>
          {(quality.independent.reasons || []).map((r) => <p key={r}>{r}</p>)}
        </> : <p>No independent controls supplied. Absolute geolocation accuracy is unknown.</p>}
      </details>
    </>}
  </Panel>
}
