import { useEffect, useMemo, useState } from 'react'
import { getConfig, runPipeline, compare, pairImageUrl, imageUrl } from './api'
import { Panel, Metric, Stages, Verdict, ImageTabs, Flow, pct, num } from './components'

export default function App() {
  const [cfg, setCfg] = useState(null)
  const [pairId, setPairId] = useState(null)
  const [opts, setOpts] = useState(null)
  const [result, setResult] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [cmp, setCmp] = useState(null)
  const [cmpRunning, setCmpRunning] = useState(false)

  useEffect(() => {
    getConfig()
      .then((c) => {
        setCfg(c)
        const first = c.pairs[0]
        setPairId(first.id)
        setOpts({ ...c.defaults, model_type: first.recommended_model || c.defaults.model_type })
      })
      .catch((e) => setError(String(e)))
  }, [])

  const pair = useMemo(() => cfg?.pairs.find((p) => p.id === pairId) || null, [cfg, pairId])

  function choosePair(p) {
    setPairId(p.id)
    setResult(null)
    setCmp(null)
    setOpts((o) => ({ ...o, model_type: p.recommended_model || o.model_type }))
  }

  const set = (k, v) => setOpts((o) => ({ ...o, [k]: v }))

  async function doRun() {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      setResult(await runPipeline(pairId, opts))
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  async function doCompare() {
    setCmpRunning(true)
    setError(null)
    try {
      const configs = [
        {
          label: 'Naive: no preprocessing, ratio 0.95, no mutual check, no spatial selection',
          ...opts,
          preprocess_enabled: false,
          ratio: 0.95,
          mutual_check: false,
          spatial_enabled: false,
        },
        { label: 'SIFT baseline (Seleno pipeline)', ...opts, matcher: 'sift' },
        { label: 'SIFT multi-scale pyramid', ...opts, matcher: 'sift_pyramid' },
        { label: 'AKAZE', ...opts, matcher: 'akaze' },
        { label: 'ORB', ...opts, matcher: 'orb' },
        { label: 'LoFTR (learned)', ...opts, matcher: 'loftr' },
      ]
      setCmp(await compare(pairId, configs))
    } catch (e) {
      setError(String(e))
    } finally {
      setCmpRunning(false)
    }
  }

  if (error && !cfg)
    return (
      <div style={{ padding: 40 }}>
        <h1 className="wordmark">SELE<span>NO</span></h1>
        <p className="err">Backend unreachable: {error}</p>
        <p className="muted">Start it with <code>python backend/app.py</code>.</p>
      </div>
    )
  if (!cfg || !opts) return <div style={{ padding: 40 }} className="spinner">loading…</div>

  const m = result?.metrics
  const gt = m?.ground_truth
  const hasGt = pair?.ground_truth_available

  return (
    <div className="app">
      {/* ------------------------------------------------------------ rail */}
      <aside className="rail">
        <div className="masthead">
          <h1 className="wordmark">SELE<span>NO</span></h1>
          <p className="tagline">Robust Lunar Image Correspondence &amp; Registration</p>
        </div>

        <div className="rail-section">
          <p className="rail-title">Image pair</p>
          <div className="pair-list">
            {cfg.pairs.map((p) => (
              <button
                key={p.id}
                className="pair"
                aria-selected={p.id === pairId}
                onClick={() => choosePair(p)}
              >
                <img src={pairImageUrl(p.id, 'source', 96)} alt="" />
                <span>
                  <span className="pair-name">{p.name}</span>
                  <span className="pair-tag">{p.difficulty}</span>
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="rail-section">
          <p className="rail-title">Experiment</p>

          <div className="field">
            <label>Matcher</label>
            <select value={opts.matcher} onChange={(e) => set('matcher', e.target.value)}>
              {cfg.matchers.map((mm) => (
                <option key={mm.id} value={mm.id} disabled={!mm.available}>
                  {mm.label}
                  {mm.available ? '' : '  (unavailable)'}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Transformation model</label>
            <select value={opts.model_type} onChange={(e) => set('model_type', e.target.value)}>
              {cfg.models.map((mm) => (
                <option key={mm.id} value={mm.id}>
                  {mm.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>RANSAC threshold — {opts.ransac_threshold.toFixed(1)} px</label>
            <input
              type="range"
              min="0.5"
              max="12"
              step="0.5"
              value={opts.ransac_threshold}
              onChange={(e) => set('ransac_threshold', parseFloat(e.target.value))}
            />
          </div>

          <div className="field">
            <label>Lowe ratio — {opts.ratio.toFixed(2)}</label>
            <input
              type="range"
              min="0.6"
              max="0.98"
              step="0.01"
              value={opts.ratio}
              onChange={(e) => set('ratio', parseFloat(e.target.value))}
            />
          </div>

          <div className="field">
            <label>Grid — {opts.grid}×{opts.grid}, max {opts.per_cell} per cell</label>
            <div className="row">
              <input
                type="range"
                min="2"
                max="12"
                step="1"
                value={opts.grid}
                onChange={(e) => set('grid', parseInt(e.target.value))}
                style={{ flex: 1 }}
              />
              <input
                type="number"
                min="1"
                max="20"
                value={opts.per_cell}
                onChange={(e) => set('per_cell', parseInt(e.target.value) || 1)}
                style={{ width: 62 }}
              />
            </div>
          </div>

          {[
            ['preprocess_enabled', 'Preprocessing', 'illumination flattening + CLAHE'],
            ['harmonise_gsd', 'GSD harmonisation', 'resample the finer image to the coarser scale'],
            ['mutual_check', 'Mutual-consistency check', 'keep only mutually-best matches'],
            ['verify_enabled', 'Geometric verification', 'robust estimator; off = accept everything'],
            ['spatial_enabled', 'Spatial distribution', 'off = top-K by confidence'],
          ].map(([k, label, hint]) => (
            <label className="check" key={k}>
              <input type="checkbox" checked={!!opts[k]} onChange={(e) => set(k, e.target.checked)} />
              <span>
                {label}
                <span className="hint"> — {hint}</span>
              </span>
            </label>
          ))}

          <div className="stack" style={{ marginTop: 14 }}>
            <button className="btn" onClick={doRun} disabled={running}>
              {running ? 'RUNNING…' : 'RUN CORRESPONDENCE'}
            </button>
            <button className="btn ghost" onClick={doCompare} disabled={cmpRunning}>
              {cmpRunning ? 'COMPARING…' : 'RUN MATCHER COMPARISON'}
            </button>
          </div>
        </div>

        {!cfg.learned_matcher_status.available && (
          <div className="rail-section">
            <div className="note warn">
              <strong>Learned matcher not installed.</strong> The LoFTR interface is implemented and
              selectable, but this environment has no weights: {cfg.learned_matcher_status.reason}.
              Selecting it returns an error rather than silently falling back to SIFT.
            </div>
          </div>
        )}
      </aside>

      {/* ------------------------------------------------------------ main */}
      <main className="main">
        <div className="topbar">
          <div>
            <h2>{pair.name}</h2>
            <div className="sub">{pair.short}</div>
          </div>
          <div className="chips">
            <span className="chip real">REAL: {pair.real_component?.slice(0, 60)}</span>
            {pair.synthetic_component && pair.synthetic_component !== 'none' && (
              <span className="chip synth">SYNTHETIC: {pair.synthetic_component.slice(0, 70)}</span>
            )}
            <span className="chip">{hasGt ? 'GROUND TRUTH KNOWN' : 'NO GROUND TRUTH'}</span>
          </div>
        </div>

        {/* inputs */}
        <div className="section">
          <h3>Input</h3>
          <div className="grid2">
            <Panel
              title="Source image"
              meta={`${pair.source_shape[1]}×${pair.source_shape[0]} px · ${num(pair.gsd_source_m, 3)} m/px`}
              caption={pair.sensor_source}
            >
              <img src={pairImageUrl(pair.id, 'source', 700)} alt="source" />
            </Panel>
            <Panel
              title="Reference image"
              meta={`${pair.reference_shape[1]}×${pair.reference_shape[0]} px · ${num(pair.gsd_reference_m, 3)} m/px`}
              caption={pair.sensor_reference}
            >
              <img src={pairImageUrl(pair.id, 'reference', 700)} alt="reference" />
            </Panel>
          </div>
          <div className="note" style={{ marginTop: 12 }}>
            <strong>Footprint</strong> {num(pair.footprint_m, 0)} m across ·{' '}
            {pair.geo?.center_lon_lat
              ? `centre ${pair.geo.center_lon_lat[1].toFixed(4)}°, ${pair.geo.center_lon_lat[0].toFixed(4)}° (lat, lon)`
              : pair.geo?.source
                ? `source centre ${pair.geo.source.center_lon_lat[1].toFixed(4)}°, reference centre ${pair.geo.reference.center_lon_lat[1].toFixed(4)}° lat`
                : ''}
            {pair.separation_km != null && ` · ground separation ${pair.separation_km} km`}
            <br />
            Selenographic coordinates are interpolated from the geometry grid ISRO ships with the
            product, not from a projection we invented.
          </div>
        </div>

        {error && <div className="err" style={{ marginBottom: 16 }}>{error}</div>}

        {/* stages */}
        <div className="section">
          <h3>Pipeline execution</h3>
          <Stages stages={result?.stages} running={running} />
        </div>

        {result && m && (
          <>
            <div className="section">
              <h3>Results</h3>
              <ImageTabs
                runId={result.run_id}
                tabs={[
                  {
                    key: 'candidates',
                    label: `1 · Candidate matches (${m.candidate_matches})`,
                    caption:
                      'Every correspondence surviving descriptor matching, before any geometric reasoning. ' +
                      'Shown on the preprocessed images the matcher actually saw.',
                  },
                  {
                    key: 'verified',
                    label: `2 · Verified (${m.inliers} in / ${m.outliers_rejected} out)`,
                    caption:
                      'The robust estimator splits the candidates into a geometrically self-consistent set and the rest.',
                    legend: (
                      <>
                        <span><i style={{ background: 'var(--good)' }} />inlier</span>
                        <span><i style={{ background: 'var(--bad)' }} />rejected outlier</span>
                      </>
                    ),
                  },
                  {
                    key: 'spatial_before',
                    label: '3a · Ranked by confidence',
                    caption:
                      `The ${m.selected_matches} highest-confidence inliers, with no spatial constraint. ` +
                      `They occupy ${pct(m.spatial_coverage_before)} of the grid cells — the clustering this stage exists to fix.`,
                  },
                  {
                    key: 'spatial_after',
                    label: '3b · Spatially selected',
                    caption:
                      `The same budget of ${m.selected_matches} matches chosen under a grid quota and a minimum separation, ` +
                      `reaching ${pct(m.spatial_coverage)} cell coverage.`,
                  },
                  {
                    key: 'overlay',
                    label: '4 · Registered overlay',
                    caption:
                      'Reference in green, registered source in magenta. Grey means the two agree; coloured fringes are residual misalignment. ' +
                      'A single global gain and bias is fitted to the warped image for display only, so the radiometric difference between the two products does not mask the geometry — no metric uses it.',
                  },
                  {
                    key: 'checker',
                    label: '4b · Checkerboard',
                    caption:
                      'Alternating tiles from each image. Crater rims and ridges should run straight across every seam.',
                  },
                  {
                    key: 'difference',
                    label: '4c · Difference',
                    caption:
                      'Absolute difference inside the overlap, contrast-stretched, after the same display-only gain/bias match. ' +
                      'Remaining structure is geometric residual plus any radiometric difference a global gain and bias cannot absorb.',
                  },
                  {
                    key: 'footprint',
                    label: '4d · Footprint',
                    caption: 'Where the source lands in the reference frame under the estimated transform.',
                  },
                ]}
              />
            </div>

            <div className="section">
              <h3>Metrics — computed from this run</h3>
              <div className="metrics">
                <Metric label="Candidate matches" value={m.candidate_matches} />
                <Metric label="Inliers" value={m.inliers} tone="good" />
                <Metric label="Outliers rejected" value={m.outliers_rejected} tone="bad" />
                <Metric
                  label="Inlier ratio"
                  value={pct(m.inlier_ratio)}
                  tone={m.inlier_ratio > 0.5 ? 'good' : m.inlier_ratio > 0.15 ? 'warn' : 'bad'}
                />
                <Metric label="Retained matches" value={m.selected_matches} sub="after spatial selection" />
                <Metric
                  label="Reprojection RMSE"
                  value={num(m.reprojection.rmse_px)}
                  unit="px"
                  sub={
                    m.reprojection.rmse_m != null
                      ? `${num(m.reprojection.rmse_m)} m · self-consistency only`
                      : 'self-consistency only'
                  }
                />
                <Metric
                  label="Spatial coverage"
                  value={pct(m.spatial_coverage)}
                  sub={`vs ${pct(m.spatial_coverage_before)} by confidence alone`}
                  tone={m.spatial_coverage > 0.6 ? 'good' : 'warn'}
                />
                <Metric
                  label="Overlap NCC"
                  value={num(m.ncc_after, 3)}
                  sub={`before registration: ${num(m.ncc_before, 3)}`}
                  tone={m.ncc_after > 0.7 ? 'good' : 'warn'}
                />
                <Metric label="Runtime" value={num(m.runtime_total_s, 2)} unit="s" />
                {hasGt && gt?.corner_error_px != null ? (
                  <>
                    <Metric
                      label="GT corner error"
                      value={num(gt.corner_error_px)}
                      unit="px"
                      sub={gt.corner_error_m != null ? `${num(gt.corner_error_m)} m on the ground` : null}
                      tone="good"
                    />
                    <Metric label="GT RMSE" value={num(gt.gt_rmse_px)} unit="px" tone="good" />
                  </>
                ) : (
                  <Metric label="Ground truth" value="none" sub="no absolute accuracy figure for this pair" />
                )}
              </div>
              <div className="note" style={{ marginTop: 12 }}>
                <strong>Reading these numbers.</strong> Reprojection RMSE measures whether the retained
                correspondences agree with the fitted model. It can be small for a model that is
                completely wrong, so it is not a registration accuracy figure.
                {hasGt
                  ? ' The GT rows compare the estimated transform against the transform used to build this pair, and are the accuracy figure.'
                  : ' This pair has no known transform, so no absolute accuracy can be reported at all.'}
              </div>
            </div>

            <div className="section">
              <h3>Verdict</h3>
              <Verdict verdict={result.verdict} warnings={result.warnings} />
            </div>
          </>
        )}

        {cmp && (
          <div className="section">
            <h3>Matcher comparison — every row is a real run on this pair</h3>
            <div className="panel">
              <div className="panel-body" style={{ overflowX: 'auto' }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>Configuration</th>
                      <th>Candidates</th>
                      <th>Inliers</th>
                      <th>Inlier ratio</th>
                      <th>Retained</th>
                      <th>Reproj RMSE</th>
                      <th>GT error</th>
                      <th>Coverage</th>
                      <th>Runtime</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cmp.rows.map((r, i) => (
                      <tr key={i}>
                        <td>{r.label}</td>
                        {r.metrics ? (
                          <>
                            <td>{r.metrics.candidate_matches}</td>
                            <td>{r.metrics.inliers}</td>
                            <td>{pct(r.metrics.inlier_ratio)}</td>
                            <td>{r.metrics.selected_matches}</td>
                            <td>{num(r.metrics.reprojection.rmse_px)}</td>
                            <td>{num(r.metrics.ground_truth?.corner_error_px)}</td>
                            <td>{pct(r.metrics.spatial_coverage)}</td>
                            <td>{num(r.metrics.runtime_total_s, 2)} s</td>
                            <td style={{ color: r.verdict?.registered ? 'var(--good)' : 'var(--bad)' }}>
                              {r.verdict?.registered ? 'accepted' : 'rejected'}
                            </td>
                          </>
                        ) : (
                          <td colSpan={9} className="err">{r.error}</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="caption">
                  Configurations that fail are shown failing. A matcher that is not installed reports
                  the import error rather than being quietly replaced.
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="section">
          <h3>Pipeline</h3>
          <Flow result={result} />
        </div>

        <div className="section">
          <h3>Scope</h3>
          <div className="note">
            Seleno is a proof of concept. It does not claim sub-pixel geodetic accuracy,
            state-of-the-art performance, a novel architecture, or any validation by ISRO.
            Feature matching on lunar data is established work — see RESEARCH.md, in particular
            Makharia et al. (2025), which benchmarks SIFT, ASIFT, AKAZE, RIFT2 and SuperGlue on
            Chandrayaan-2 data. What this prototype demonstrates is the engineering path: measured
            preprocessing, geometric verification, spatially distributed selection, and metrics that
            distinguish self-consistency from accuracy. IIRS cross-modal matching is not implemented.
          </div>
        </div>
      </main>
    </div>
  )
}
