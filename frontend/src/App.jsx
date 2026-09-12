import { useEffect, useMemo, useState } from 'react'
import * as api from './api'
import {
  Flow, ImageTabs, KV, Metric, Panel, Stages, Verdict,
  niceTs, num, pct, shortTs,
} from './components'

const SIZES = [512, 1024, 2048]

export default function App() {
  const [cfg, setCfg] = useState(null)
  const [fatal, setFatal] = useState(null)
  const [mode, setMode] = useState('ohrc')          // ohrc | legacy

  const [srcTs, setSrcTs] = useState(null)
  const [refTs, setRefTs] = useState(null)
  const [size, setSize] = useState(1024)
  const [useOffset, setUseOffset] = useState(true)
  const [align, setAlign] = useState(null)
  const [windows, setWindows] = useState([])
  const [win, setWin] = useState(null)
  const [loadingWins, setLoadingWins] = useState(false)

  const [legacyId, setLegacyId] = useState(null)
  const [preset, setPreset] = useState('seleno')
  const [opts, setOpts] = useState({})

  const [res, setRes] = useState(null)
  const [running, setRunning] = useState(false)
  const [cmp, setCmp] = useState(null)
  const [cmpRunning, setCmpRunning] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getConfig()
      .then((c) => {
        setCfg(c)
        const ps = c.ohrc?.products || []
        if (ps.length >= 2) {
          const a = ps.find((p) => p.timestamp.startsWith('20241115T1326')) || ps[0]
          const b = ps.find((p) => p.timestamp.startsWith('20241115T1525')) || ps[1]
          setSrcTs(a.timestamp); setRefTs(b.timestamp)
        } else if (ps.length === 1) {
          setSrcTs(ps[0].timestamp)
        }
        if (!ps.length) setMode('legacy')
        if (c.legacy_pairs?.length) setLegacyId(c.legacy_pairs[0].id)
      })
      .catch((e) => setFatal(String(e)))
  }, [])

  // load candidate windows whenever the pair or size changes
  useEffect(() => {
    if (mode !== 'ohrc' || !srcTs || !refTs || srcTs === refTs) return
    let cancelled = false
    setLoadingWins(true); setWindows([]); setWin(null); setRes(null); setCmp(null)
    api.getWindows(srcTs, refTs, size, useOffset)
      .then((d) => {
        if (cancelled) return
        setAlign(d.coarse_alignment)
        setWindows(d.windows || [])
        setWin((d.windows || [])[0] || null)
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoadingWins(false))
    return () => { cancelled = true }
  }, [mode, srcTs, refTs, size, useOffset])

  const products = cfg?.ohrc?.products || []
  const srcP = products.find((p) => p.timestamp === srcTs)
  const refP = products.find((p) => p.timestamp === refTs)
  const legacy = (cfg?.legacy_pairs || []).find((p) => p.id === legacyId)

  const set = (k, v) => setOpts((o) => ({ ...o, [k]: v }))
  const effective = useMemo(() => ({ ...(res?.options || cfg?.defaults || {}), ...opts }), [res, cfg, opts])

  function payload(extra = {}) {
    if (mode === 'legacy') return { legacy: legacyId, preset, options: opts, ...extra }
    return {
      source: srcTs, reference: refTs,
      sample0: win?.sample0, line0: win?.line0, size,
      apply_coarse_offset: useOffset, preset, options: opts, ...extra,
    }
  }

  async function doRun() {
    setRunning(true); setError(null); setRes(null)
    try { setRes(await api.register(payload())) }
    catch (e) { setError(String(e)) }
    finally { setRunning(false) }
  }

  async function doCompare() {
    setCmpRunning(true); setError(null)
    try { setCmp(await api.compare(payload())) }
    catch (e) { setError(String(e)) }
    finally { setCmpRunning(false) }
  }

  if (fatal)
    return (
      <div style={{ padding: 40 }}>
        <h1 className="wordmark">SELE<span>NO</span></h1>
        <p className="err">Backend unreachable: {fatal}</p>
        <p className="muted">Start it with <code>python run.py</code>.</p>
      </div>
    )
  if (!cfg) return <div style={{ padding: 40 }} className="spinner">loading…</div>

  const m = res?.metrics
  const pairMeta = res?.pair
  const hasGt = pairMeta?.ground_truth_available
  const gt = m?.ground_truth
  const dis = m?.geometry_prior_disagreement
  const canRun = mode === 'legacy' ? !!legacyId : !!(srcTs && refTs && win)

  return (
    <div className="app">
      {/* ------------------------------------------------------------- rail */}
      <aside className="rail">
        <div className="masthead">
          <h1 className="wordmark">SELE<span>NO</span></h1>
          <p className="tagline">
            Geometry-first lunar image registration with predicted unmatchable regions
            and calibrated refusal
          </p>
        </div>

        <div className="rail-section">
          <div className="seg">
            <button aria-selected={mode === 'ohrc'} onClick={() => setMode('ohrc')}
                    disabled={!products.length}>
              OHRC pair
            </button>
            <button aria-selected={mode === 'legacy'} onClick={() => setMode('legacy')}
                    disabled={!(cfg.legacy_pairs || []).length}>
              Legacy pairs
            </button>
          </div>

          {mode === 'ohrc' ? (
            <>
              {!cfg.ohrc.available && <div className="note bad">{cfg.ohrc.error}</div>}
              <div className="field">
                <label>Source product</label>
                <select value={srcTs || ''} onChange={(e) => setSrcTs(e.target.value)}>
                  {products.map((p) => (
                    <option key={p.timestamp} value={p.timestamp}>
                      {niceTs(p.timestamp)} · el {num(p.sun_elevation_deg, 2)}°
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Reference product</label>
                <select value={refTs || ''} onChange={(e) => setRefTs(e.target.value)}>
                  {products.map((p) => (
                    <option key={p.timestamp} value={p.timestamp}
                            disabled={p.timestamp === srcTs}>
                      {niceTs(p.timestamp)} · el {num(p.sun_elevation_deg, 2)}°
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Window size</label>
                <select value={size} onChange={(e) => setSize(parseInt(e.target.value))}>
                  {SIZES.map((s) => (
                    <option key={s} value={s}>{s} px · {num(s * (srcP?.gsd_m || 0.24), 0)} m</option>
                  ))}
                </select>
              </div>
              <label className="check">
                <input type="checkbox" checked={useOffset}
                       onChange={(e) => setUseOffset(e.target.checked)} />
                <span>
                  Apply measured coarse offset
                  <span className="hint"> — corrects the disagreement between the two
                  products' delivered geolocation. Turn off to see what happens without it.</span>
                </span>
              </label>

              <p className="rail-title" style={{ marginTop: 12 }}>
                Candidate windows {loadingWins ? '(loading…)' : `(${windows.length})`}
              </p>
              <div className="wlist">
                {windows.map((w) => (
                  <button key={`${w.sample0}-${w.line0}`} className="witem"
                          aria-selected={win && w.sample0 === win.sample0 && w.line0 === win.line0}
                          onClick={() => { setWin(w); setRes(null); setCmp(null) }}>
                    <span>s{w.sample0} l{w.line0}</span>
                    <span>sd {num(w.source_std, 0)}/{num(w.reference_std, 0)}</span>
                  </button>
                ))}
                {!windows.length && !loadingWins && (
                  <span className="muted" style={{ fontSize: 11 }}>no usable window found</span>
                )}
              </div>
            </>
          ) : (
            <div className="plist">
              {(cfg.legacy_pairs || []).map((p) => (
                <button key={p.id} className="pitem" aria-selected={p.id === legacyId}
                        onClick={() => { setLegacyId(p.id); setRes(null); setCmp(null) }}>
                  <img src={api.legacyPairImageUrl(p.id, 'source', 96)} alt="" />
                  <span>
                    <span className="nm">{p.name}</span>
                    <span className="tg">{p.difficulty}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="rail-section">
          <p className="rail-title">Configuration</p>
          <div className="field">
            <label>Preset</label>
            <select value={preset} onChange={(e) => { setPreset(e.target.value); setOpts({}) }}>
              {cfg.presets.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Matcher</label>
            <select value={opts.matcher ?? effective.matcher ?? 'sift'}
                    onChange={(e) => set('matcher', e.target.value)}>
              {cfg.matchers.map((x) => (
                <option key={x.id} value={x.id} disabled={!x.available}>
                  {x.label}{x.available ? '' : '  (unavailable)'}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Usability mask</label>
            <select value={opts.mask_mode ?? effective.mask_mode ?? 'none'}
                    onChange={(e) => set('mask_mode', e.target.value)}>
              {cfg.mask_modes.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Terrain model</label>
            <select value={opts.terrain_model ?? effective.terrain_model ?? 'none'}
                    onChange={(e) => set('terrain_model', e.target.value)}>
              {cfg.terrain_models.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.label}{x.status === 'synthetic' ? '' : ''}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Transform model</label>
            <select value={opts.model_type ?? effective.model_type ?? 'similarity'}
                    onChange={(e) => set('model_type', e.target.value)}>
              {cfg.models.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Correspondence selection</label>
            <select value={opts.selection_mode ?? effective.selection_mode ?? 'grid'}
                    onChange={(e) => set('selection_mode', e.target.value)}>
              {cfg.selection_modes.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label>
              RANSAC threshold — {num(opts.ransac_threshold ?? effective.ransac_threshold ?? 3, 1)} px
            </label>
            <input type="range" min="0.5" max="12" step="0.5"
                   value={opts.ransac_threshold ?? effective.ransac_threshold ?? 3}
                   onChange={(e) => set('ransac_threshold', parseFloat(e.target.value))} />
          </div>
          <label className="check">
            <input type="checkbox"
                   checked={opts.subpixel_enabled ?? effective.subpixel_enabled ?? true}
                   onChange={(e) => set('subpixel_enabled', e.target.checked)} />
            <span>Sub-pixel refinement<span className="hint"> — local NCC peak fit</span></span>
          </label>
          <label className="check">
            <input type="checkbox"
                   checked={opts.remove_shading ?? effective.remove_shading ?? false}
                   onChange={(e) => set('remove_shading', e.target.checked)} />
            <span>Remove broad terrain shading<span className="hint"> — needs a terrain model</span></span>
          </label>

          <div className="stack" style={{ marginTop: 13 }}>
            <button className="btn" onClick={doRun} disabled={running || !canRun}>
              {running ? 'RUNNING…' : 'RUN REGISTRATION'}
            </button>
            <button className="btn ghost" onClick={doCompare} disabled={cmpRunning || !canRun}>
              {cmpRunning ? 'COMPARING…' : 'RUN ABLATION'}
            </button>
          </div>
        </div>
      </aside>

      {/* ------------------------------------------------------------- main */}
      <main className="main">
        <div className="topbar">
          <div>
            <h2>
              {mode === 'ohrc'
                ? `OHRC ${shortTs(srcTs)} → ${shortTs(refTs)}`
                : legacy?.name || '—'}
            </h2>
            <div className="sub">
              {mode === 'ohrc'
                ? win
                  ? `window sample ${win.sample0}, line ${win.line0} · ${size} px · ${num(size * (srcP?.gsd_m || 0.24), 0)} m across`
                  : 'select a candidate window'
                : legacy?.short}
            </div>
          </div>
          <div className="chips">
            {mode === 'ohrc' && <span className="chip real">REAL REPEAT-PASS OHRC</span>}
            <span className={'chip ' + (hasGt ? 'real' : 'nogt')}>
              {hasGt ? 'GROUND TRUTH KNOWN' : 'NO GROUND TRUTH EXISTS'}
            </span>
            {pairMeta?.d_azimuth_deg != null && (
              <span className="chip sun">
                Δaz {num(pairMeta.d_azimuth_deg, 1)}° · Δel {num(pairMeta.d_elevation_deg, 2)}°
              </span>
            )}
            {m?.mask_is_synthetic_terrain && <span className="chip synth">SYNTHETIC TERRAIN</span>}
          </div>
        </div>

        {error && <div className="note bad" style={{ marginBottom: 14 }}>{error}</div>}

        {/* ------------------------------------------- three-panel main view */}
        <div className="section">
          <h3>Source · Reference · Registration</h3>
          <div className="grid3">
            <Panel
              title="Source image"
              meta={mode === 'ohrc'
                ? `${size}×${size} px · ${num(srcP?.gsd_m, 3)} m/px`
                : `${pairMeta?.source_shape?.[1] || ''}×${pairMeta?.source_shape?.[0] || ''} px`}
              caption={mode === 'ohrc'
                ? `${niceTs(srcTs)} · orbit ${srcP?.imaging_orbit} · sun el ${num(pairMeta?.sun_source?.elevation_deg ?? srcP?.sun_elevation_deg, 3)}° az ${num(pairMeta?.sun_source?.azimuth_deg ?? srcP?.sun_azimuth_deg, 1)}°`
                : legacy?.sensor_source}
            >
              {mode === 'ohrc'
                ? win
                  ? <img src={api.pairPreviewUrl(srcTs, refTs, win.sample0, win.line0, size, 'source', useOffset)} alt="source" />
                  : <div className="ph">no window selected</div>
                : <img src={api.legacyPairImageUrl(legacyId, 'source')} alt="source" />}
            </Panel>

            <Panel
              title="Reference image"
              meta={mode === 'ohrc'
                ? `${size}×${size} px · ${num(refP?.gsd_m, 3)} m/px`
                : `${pairMeta?.reference_shape?.[1] || ''}×${pairMeta?.reference_shape?.[0] || ''} px`}
              caption={mode === 'ohrc'
                ? `${niceTs(refTs)} · orbit ${refP?.imaging_orbit} · sun el ${num(pairMeta?.sun_reference?.elevation_deg ?? refP?.sun_elevation_deg, 3)}° az ${num(pairMeta?.sun_reference?.azimuth_deg ?? refP?.sun_azimuth_deg, 1)}°`
                : legacy?.sensor_reference}
            >
              {mode === 'ohrc'
                ? win
                  ? <img src={api.pairPreviewUrl(srcTs, refTs, win.sample0, win.line0, size, 'reference', useOffset)} alt="reference" />
                  : <div className="ph">no window selected</div>
                : <img src={api.legacyPairImageUrl(legacyId, 'reference')} alt="reference" />}
            </Panel>

            <Panel
              title="Registration result"
              meta={res ? res.status.toUpperCase() : '—'}
              caption={res
                ? 'Reference in green, registered source in magenta. Grey means agreement; coloured fringes are residual misalignment. A display-only gain/bias match is applied so radiometry does not mask geometry.'
                : 'run the pipeline to produce this'}
            >
              {res?.run_id && (res.available_images || []).includes('overlay')
                ? <img src={api.imageUrl(res.run_id, 'overlay')} alt="overlay" />
                : <div className="ph">{running ? 'running…' : 'no result yet'}</div>}
            </Panel>
          </div>
        </div>

        {/* ------------------------------------------------ geometry / sun */}
        {mode === 'ohrc' && (
          <div className="section">
            <h3>Geometry, illumination and the coarse offset</h3>
            <div className="grid3">
              <Panel title="Delivered geolocation">
                <KV items={[
                  ['source corners refined', String(srcP?.corners_refined)],
                  ['reference corners refined', String(refP?.corners_refined)],
                  ['reference data used', srcP?.reference_data_used],
                  ['prior origin', pairMeta?.geometry_prior_origin ? 'measured' : '—'],
                ]} />
                <div className="caption">
                  Both products report <code>reference_data_used = System</code> and their
                  "refined" corners are byte-identical to the system-predicted ones — no
                  photogrammetric refinement has been applied, so this geolocation is a
                  prior, never truth.
                </div>
              </Panel>

              <Panel title="Coarse offset between products"
                     meta={align?.confident ? 'CONFIDENT' : align?.ok ? 'NOT CONFIDENT' : 'FAILED'}>
                {align?.ok ? (
                  <>
                    <KV items={[
                      ['offset', `${num(align.offset_m, 0)} m`],
                      ['in source pixels', num(align.offset_px_source_gsd, 0)],
                      ['components (x, y)', `${align.offset_stereo_m?.map((v) => num(v, 0)).join(', ')} m`],
                      ['peak NCC', num(align.peak_ncc, 3)],
                      ['runner-up', num(align.second_peak_ncc, 3)],
                      ['margin', num(align.peak_margin, 3)],
                      ['on', align.representation],
                    ]} />
                    <div className="caption">
                      The two products' delivered geolocation disagrees by{' '}
                      <strong>{num(align.offset_m, 0)} m</strong> — about{' '}
                      {num(align.offset_px_source_gsd, 0)} pixels. Without correcting it a
                      window and its geometry-predicted counterpart share almost no ground,
                      and any matcher looks broken for the wrong reason.
                    </div>
                  </>
                ) : (
                  <div className="caption">{align?.reason || 'not measured'}</div>
                )}
              </Panel>

              <Panel title="Illumination change">
                <KV items={[
                  ['source elevation', `${num(pairMeta?.sun_source?.elevation_deg ?? srcP?.sun_elevation_deg, 3)}°`],
                  ['reference elevation', `${num(pairMeta?.sun_reference?.elevation_deg ?? refP?.sun_elevation_deg, 3)}°`],
                  ['source azimuth', `${num(pairMeta?.sun_source?.azimuth_deg ?? srcP?.sun_azimuth_deg, 1)}°`],
                  ['reference azimuth', `${num(pairMeta?.sun_reference?.azimuth_deg ?? refP?.sun_azimuth_deg, 1)}°`],
                  ['Δ azimuth', pairMeta?.d_azimuth_deg != null ? `${num(pairMeta.d_azimuth_deg, 1)}°` : undefined],
                  ['Δ elevation', pairMeta?.d_elevation_deg != null ? `${num(pairMeta.d_elevation_deg, 3)}°` : undefined],
                ]} />
                <div className="caption">
                  Solar incidence is 89–91° throughout this archive. At these grazing
                  elevations one metre of relief casts tens to hundreds of metres of shadow,
                  so an azimuth change of this size moves cast shadows bodily rather than
                  just rescaling brightness.
                </div>
              </Panel>
            </div>
          </div>
        )}

        {/* --------------------------------------------------------- stages */}
        <div className="section">
          <h3>Pipeline execution</h3>
          <Stages stages={res?.stages} running={running} />
        </div>

        {/* -------------------------------------------------------- results */}
        {res && m && (
          <>
            <div className="section">
              <h3>Diagnostics</h3>
              <ImageTabs
                runId={res.run_id}
                tabs={[
                  {
                    key: 'mask', label: '1 · Usability mask',
                    available: (res.available_images || []).includes('mask'),
                    caption: `Regions judged unmatchable are tinted. Source: ${m.mask_source}. ` +
                      `${pct(m.usable_fraction)} of the window was kept.` +
                      (m.mask_is_prediction
                        ? ' This mask is a PREDICTION from a terrain model.'
                        : ' This mask is MEASURED from the image itself — no terrain model involved.'),
                  },
                  {
                    key: 'candidates', label: `2 · Candidates (${m.candidate_matches})`,
                    available: (res.available_images || []).includes('candidates'),
                    caption: 'Correspondences surviving descriptor matching, before any geometric reasoning.' +
                      (m.dropped_by_mask ? ` ${m.dropped_by_mask} were dropped as unmatchable.` : ''),
                  },
                  {
                    key: 'verified', label: `3 · Verified (${m.inliers} in / ${m.outliers_rejected} out)`,
                    available: (res.available_images || []).includes('verified'),
                    caption: 'The robust estimator splits candidates into a geometrically self-consistent set and the rest.',
                    legend: (
                      <>
                        <span><i style={{ background: 'var(--accepted)' }} />inlier</span>
                        <span><i style={{ background: 'var(--refused)' }} />rejected outlier</span>
                      </>
                    ),
                  },
                  {
                    key: 'select_before', label: '4a · Ranked by confidence',
                    available: (res.available_images || []).includes('select_before'),
                    caption: `The same budget of matches chosen by confidence alone, reaching ` +
                      `${pct(m.spatial_coverage_before)} of the ${m.matchable_cells} matchable cells.`,
                  },
                  {
                    key: 'select_after', label: '4b · Selected',
                    available: (res.available_images || []).includes('select_after'),
                    caption: `The set the pipeline used, reaching ${pct(m.spatial_coverage)} of the ` +
                      `${m.matchable_cells} matchable cells (${pct(m.spatial_coverage_whole_window)} of the whole window). ` +
                      'Cells with nothing matchable in them are excluded from the denominator, otherwise a good ' +
                      'registration over shadowed terrain would be penalised for the wrong reason.',
                  },
                  {
                    key: 'checker', label: '5 · Checkerboard',
                    available: (res.available_images || []).includes('checker'),
                    caption: 'Alternating tiles from each image. Crater rims should run straight across every seam.',
                  },
                  {
                    key: 'difference', label: '6 · Difference',
                    available: (res.available_images || []).includes('difference'),
                    caption: 'Absolute difference inside the overlap after a display-only gain/bias match. ' +
                      'Residual structure is geometric error plus any radiometric difference a global gain cannot absorb.',
                  },
                  {
                    key: 'footprint', label: '7 · Footprint',
                    available: (res.available_images || []).includes('footprint'),
                    caption: 'Where the source lands in the reference frame under the estimated transform.',
                  },
                ]}
              />
            </div>

            <div className="section">
              <h3>Metrics — computed by this run</h3>
              <div className="metrics">
                <Metric label="Candidates" value={m.candidate_matches}
                        sub={m.candidates_before_mask != null
                          ? `${m.candidates_before_mask} before masking` : null} />
                <Metric label="Inliers" value={m.inliers} tone="good" />
                <Metric label="Outliers rejected" value={m.outliers_rejected} tone="bad" />
                <Metric label="Inlier ratio" value={pct(m.inlier_ratio)}
                        tone={m.inlier_ratio > 0.5 ? 'good' : m.inlier_ratio > 0.15 ? 'warn' : 'bad'} />
                <Metric label="Retained" value={m.selected_matches} sub="after selection" />
                <Metric label="Reprojection RMSE" value={num(m.reprojection?.rmse_px)} unit="px"
                        sub="self-consistency, not accuracy" />
                <Metric label="Held-out RMSE" value={num(m.heldout?.rmse_px)} unit="px"
                        sub={`${m.heldout_count || 0} points excluded from the fit`}
                        tone={m.heldout?.rmse_px == null ? undefined
                          : m.heldout.rmse_px < 3 ? 'good' : m.heldout.rmse_px < 6 ? 'warn' : 'bad'} />
                {hasGt && gt?.corner_error_px != null ? (
                  <Metric label="Ground-truth error" value={num(gt.corner_error_px)} unit="px"
                          sub={gt.corner_error_m != null ? `${num(gt.corner_error_m)} m` : null}
                          tone="good" />
                ) : (
                  <Metric label="Ground-truth error" value="none exists" tone="nogt"
                          sub="both windows are real acquisitions" />
                )}
                <Metric label="vs delivered geometry"
                        value={num(dis?.corner_disagreement_m, 0)} unit="m"
                        sub="independent check, not truth" />
                <Metric label="Matchable area" value={pct(m.usable_fraction)}
                        sub={m.mask_source} />
                <Metric label="Coverage" value={pct(m.spatial_coverage)}
                        sub={`of ${m.matchable_cells} matchable cells · by confidence ${pct(m.spatial_coverage_before)}`}
                        tone={m.spatial_coverage > 0.6 ? 'good' : 'warn'} />
                <Metric label="Frame overlap" value={pct(m.overlap_fraction, 0)}
                        sub="warped source over the reference frame"
                        tone={m.overlap_fraction == null ? undefined
                          : m.overlap_fraction >= 0.5 ? 'good' : 'warn'} />
                <Metric label="Overlap NCC" value={num(m.ncc_after, 3)}
                        sub={`before: ${num(m.ncc_before, 3)}`}
                        tone={m.ncc_after > 0.7 ? 'good' : 'warn'} />
                <Metric label="Runtime" value={num(m.runtime_total_s, 2)} unit="s" />
              </div>
              <div className="note" style={{ marginTop: 11 }}>
                <strong>Four different error concepts, never merged.</strong>{' '}
                <em>Reprojection RMSE</em> says the retained correspondences agree with the
                fitted model — a wrong model can score well. <em>Held-out RMSE</em> evaluates
                the fit on points it never saw, so it tests generalisation.{' '}
                <em>Ground-truth error</em> exists only where a correct transform is known by
                construction{hasGt ? '' : ', which is not the case for this pair'}.{' '}
                <em>Disagreement with the delivered geometry</em> compares against ISRO's
                system-level geolocation, itself only accurate to metres-to-decametres.
              </div>
            </div>

            <div className="section">
              <h3>Decision</h3>
              <Verdict status={res.status} reasons={res.reasons} confidence={res.confidence}
                       warnings={res.warnings} calibrated={cfg.thresholds_calibrated} />
            </div>
          </>
        )}

        {/* ------------------------------------------------------ ablation */}
        {cmp && (
          <div className="section">
            <h3>Ablation — every row is a real run on this pair</h3>
            <div className="panel">
              <div className="panel-body" style={{ overflowX: 'auto' }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>Arm</th><th>Configuration</th><th>Cand.</th><th>Inliers</th>
                      <th>Ratio</th><th>Retained</th><th>Reproj RMSE</th><th>Held-out</th>
                      <th>GT error</th><th>Coverage</th><th>Matchable</th><th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cmp.rows.map((r, i) => (
                      <tr key={i}>
                        <td>{r.code}</td>
                        <td>{r.label}</td>
                        {r.metrics ? (
                          <>
                            <td>{r.metrics.candidate_matches}</td>
                            <td>{r.metrics.inliers}</td>
                            <td>{pct(r.metrics.inlier_ratio, 0)}</td>
                            <td>{r.metrics.selected_matches}</td>
                            <td>{num(r.metrics.reprojection?.rmse_px)}</td>
                            <td>{num(r.metrics.heldout?.rmse_px)}</td>
                            <td>{r.metrics.ground_truth?.corner_error_px != null
                              ? num(r.metrics.ground_truth.corner_error_px) : 'none'}</td>
                            <td>{pct(r.metrics.spatial_coverage, 0)}</td>
                            <td>{pct(r.metrics.usable_fraction, 0)}</td>
                            <td className={'st-' + r.status}>{r.status}</td>
                          </>
                        ) : (
                          <td colSpan={10} className="err">{r.error}</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="caption">
                  Arm E (observed usability mask, no DEM) is the <strong>control</strong> for
                  arms F and G. Excluding dark pixels helps a matcher whether or not a shadow
                  <em> prediction</em> was any good, so a predicted mask that does not beat the
                  observed mask has demonstrated nothing. Where a synthetic terrain model was
                  used, the row is not evidence about the real surface.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* -------------------------------------------------------- pipeline */}
        <div className="section">
          <h3>Architecture</h3>
          <Flow options={res?.options || effective}
                maskSummary={{ is_synthetic_terrain: m?.mask_is_synthetic_terrain }} />
        </div>

        <div className="section">
          <h3>Scope</h3>
          <div className="note">
            Seleno is a proof of concept. The proposed contribution — using delivered Sun
            geometry and a coarse terrain model to predict which regions cannot be matched,
            then refusing registration on calibrated evidence — is <strong>implemented but
            not yet validated</strong>. No digital elevation model ships with this repository,
            so predicted-shadow results come from a synthetic terrain model and are labelled
            as such. The decision thresholds are hand-picked, not calibrated, and no
            precision/recall for the refusal decision has been measured. Matcher choice is not
            claimed as a contribution: SIFT/AKAZE/ORB comparisons on Chandrayaan-2 data are
            established work (see RESEARCH.md). IIRS cross-modal matching is not implemented.
          </div>
        </div>
      </main>
    </div>
  )
}
