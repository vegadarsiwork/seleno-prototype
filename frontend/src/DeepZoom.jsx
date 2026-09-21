/* Any input at any zoom, from the whole mosaic down to single native pixels.
 *
 * Tiles are cut on demand by /api/tool/view/tile from the raster the tool
 * registers, so what is on screen is the file itself, not a preview of it.
 * Past 1:1 the pixels are drawn as blocks, never smoothed: a smoothed pixel
 * suggests detail the sensor did not record.
 *
 * `Explore` puts source and reference side by side with the run's tie points
 * on both, and can lock the two views together through those tie points.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import * as api from './api'

const DEG = 180 / Math.PI

/* lat/lon for the two cylindrical projections the server describes. Both are
 * linear in pixel coordinates, so no projection library is needed. */
function pxToLonLat(g, x, y) {
  if (!g || !g.kind) return null
  const [a, b, c, d, e, f] = g.affine
  const X = c + a * x + b * y
  const Y = f + d * x + e * y
  if (g.kind === 'longlat') return [X, Y]
  return [g.lon_0 + DEG * (X - g.x_0) / (g.R * Math.cos(g.lat_ts / DEG)),
          DEG * (Y - g.y_0) / g.R]
}

function lonLatToPx(g, lon, lat) {
  let X = lon
  let Y = lat
  if (g.kind === 'eqc') {
    X = g.x_0 + g.R * Math.cos(g.lat_ts / DEG) * (lon - g.lon_0) / DEG
    Y = g.y_0 + g.R * lat / DEG
  }
  const [a, b, c, d, e, f] = g.affine
  const det = a * e - b * d
  return [(e * (X - c) - b * (Y - f)) / det, (-d * (X - c) + a * (Y - f)) / det]
}

const fmtM = (m) => m >= 1000 ? (m / 1000).toFixed(m >= 1e4 ? 0 : 1) + ' km'
  : m >= 1 ? m.toFixed(m >= 10 ? 0 : 1) + ' m' : (m * 100).toFixed(0) + ' cm'

function Hud({ info, hover, scale }) {
  const ll = hover && pxToLonLat(info.georef, hover.x, hover.y)
  return (
    <div className="dz-hud">
      {hover
        ? <span>x {hover.x.toFixed(1)} &nbsp;y {hover.y.toFixed(1)}</span>
        : <span className="dim">pointer outside image</span>}
      {ll && <span>lat {ll[1].toFixed(4)}° &nbsp;lon {ll[0].toFixed(4)}°</span>}
      {scale != null && (
        <span className="dim">
          {scale >= 1
            ? `1 screen px = ${scale >= 10 ? scale.toFixed(0) : scale.toFixed(1)} px`
            : `1 px = ${(1 / scale).toFixed(1)} screen px`}
          {info.gsd_m ? ` (${fmtM(info.gsd_m * Math.max(scale, 1))})` : ''}
        </span>
      )}
    </div>
  )
}

/* One viewer. `points` = { x, y, inl, n } in this image's pixel coordinates
 * (OpenCV convention: integer = pixel centre), drawn on a canvas above it. */
export default function DeepZoom({ path, label, points, showOutliers, onViewer,
                                   onInfo, height = '70vh' }) {
  const host = useRef(null)
  const canvas = useRef(null)
  const viewer = useRef(null)
  const drawRef = useRef(() => {})
  const [info, setInfo] = useState(null)
  const [err, setErr] = useState(null)
  const [wait, setWait] = useState(0)
  const [hover, setHover] = useState(null)
  const [scale, setScale] = useState(null)
  const [mark, setMark] = useState(null)
  const [goMode, setGoMode] = useState('px')
  const [goText, setGoText] = useState('')
  const [goErr, setGoErr] = useState(null)
  const [saved, setSaved] = useState(null)

  useEffect(() => {
    let dead = false
    setInfo(null); setErr(null); setWait(0); setMark(null); setSaved(null)
    const t0 = Date.now()
    const tick = setInterval(() => setWait(Math.round((Date.now() - t0) / 1000)), 1000)
    api.viewInfo(path)
      .then((d) => {
        if (dead) return
        setInfo(d)
        if (onInfo) onInfo(d)
        setGoMode(d.georef?.kind ? 'll' : 'px')
      })
      .catch((e) => { if (!dead) setErr(e.message || String(e)) })
      .finally(() => clearInterval(tick))
    return () => { dead = true; clearInterval(tick) }
    // onInfo is a notification, not an input
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path])

  useEffect(() => {
    if (!info || !host.current) return
    const maxLevel = Math.round(Math.log2(info.max_scale))
    const v = OpenSeadragon({
      element: host.current,
      tileSources: {
        width: info.width, height: info.height,
        tileSize: info.tile_size, tileOverlap: 0, minLevel: 0, maxLevel,
        getTileUrl: (level, x, y) =>
          api.viewTileUrl(path, 2 ** (maxLevel - level), x, y, info.version),
      },
      showNavigationControl: false,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      navigatorSizeRatio: 0.16,
      navigatorAutoFade: false,
      maxZoomPixelRatio: 24,
      imageSmoothingEnabled: false,
      visibilityRatio: 0.2,
      animationTime: 0.5,
      zoomPerScroll: 1.4,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
      gestureSettingsTouch: { pinchRotate: false },
    })
    viewer.current = v
    const updateScale = () => {
      if (!v.world.getItemCount()) return
      setScale(1 / v.viewport.viewportToImageZoom(v.viewport.getZoom(true)))
    }
    const redraw = () => drawRef.current()
    v.addHandler('open', updateScale)
    v.addHandler('animation-finish', updateScale)
    v.addHandler('update-viewport', redraw)
    const el = host.current
    let raf = 0
    const move = (e) => {
      if (raf || !v.world.getItemCount()) return
      const { clientX, clientY } = e
      raf = requestAnimationFrame(() => {
        raf = 0
        const r = el.getBoundingClientRect()
        const p = v.viewport.viewerElementToImageCoordinates(
          new OpenSeadragon.Point(clientX - r.left, clientY - r.top))
        const inside = p.x >= 0 && p.y >= 0 && p.x <= info.width && p.y <= info.height
        setHover(inside ? { x: p.x, y: p.y } : null)
      })
    }
    const leave = () => setHover(null)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerleave', leave)
    if (onViewer) onViewer(v)
    return () => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerleave', leave)
      if (raf) cancelAnimationFrame(raf)
      if (onViewer) onViewer(null)
      v.destroy()
      viewer.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info, path])

  // The overlay is redrawn on every frame the viewer draws, so the points move
  // with the image during animation instead of snapping after it.
  drawRef.current = () => {
    const v = viewer.current
    const c = canvas.current
    const box = host.current
    if (!v || !c || !box || !v.world.getItemCount()) return
    const dpr = window.devicePixelRatio || 1
    const w = box.clientWidth
    const h = box.clientHeight
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr)
    }
    const g = c.getContext('2d')
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    const vp = v.viewport
    const p0 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0))
    const p1 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(1, 1))
    const kx = p1.x - p0.x
    const ky = p1.y - p0.y
    if (points && points.n) {
      const r = kx > 6 ? 3 : 1.6
      const passes = showOutliers ? [0, 1] : [1]
      for (const want of passes) {
        g.fillStyle = want ? 'rgba(90,220,140,0.9)' : 'rgba(230,103,103,0.75)'
        for (let i = 0; i < points.n; i++) {
          if (points.inl[i] !== want) continue
          const ex = p0.x + (points.x[i] + 0.5) * kx
          const ey = p0.y + (points.y[i] + 0.5) * ky
          if (ex < -4 || ey < -4 || ex > w + 4 || ey > h + 4) continue
          g.fillRect(ex - r, ey - r, 2 * r, 2 * r)
        }
      }
    }
    if (mark) {
      const ex = p0.x + mark.x * kx
      const ey = p0.y + mark.y * ky
      g.strokeStyle = '#c98500'
      g.lineWidth = 1.5
      g.beginPath()
      g.moveTo(ex - 14, ey); g.lineTo(ex - 4, ey); g.moveTo(ex + 4, ey); g.lineTo(ex + 14, ey)
      g.moveTo(ex, ey - 14); g.lineTo(ex, ey - 4); g.moveTo(ex, ey + 4); g.lineTo(ex, ey + 14)
      g.stroke()
    }
  }
  useEffect(() => { drawRef.current() }, [points, showOutliers, mark])

  const zoomBy = (f) => { const v = viewer.current; if (v) v.viewport.zoomBy(f).applyConstraints() }
  const home = () => viewer.current && viewer.current.viewport.goHome()
  const oneToOne = () => {
    const v = viewer.current
    if (v) v.viewport.zoomTo(v.viewport.imageToViewportZoom(1))
  }

  const go = (e) => {
    e.preventDefault()
    setGoErr(null)
    const v = viewer.current
    const nums = goText.split(/[\s,;]+/).filter(Boolean).map(Number)
    if (!v || nums.length !== 2 || nums.some((n) => !Number.isFinite(n))) {
      setGoErr(goMode === 'll' ? 'enter "lat, lon"' : 'enter "x, y"')
      return
    }
    let x, y
    if (goMode === 'll') {
      const [lat, lon] = nums
      if (Math.abs(lat) > 90) { setGoErr('latitude must be within ±90'); return }
      // Try the longitude as given and wrapped, keep whichever lands on the image.
      const cands = [lon, lon - 360, lon + 360].map((l) => lonLatToPx(info.georef, l, lat))
      ;[x, y] = cands.find(([cx]) => cx >= 0 && cx <= info.width) || cands[0]
    } else {
      [x, y] = nums
    }
    if (x < 0 || y < 0 || x > info.width || y > info.height) {
      setGoErr('that point is outside this image')
      return
    }
    setMark({ x, y })
    const vp = v.viewport
    vp.panTo(vp.imageToViewportCoordinates(new OpenSeadragon.Point(x, y)))
    // Zoom in to 1:1 unless already closer.
    if (vp.viewportToImageZoom(vp.getZoom()) < 1) vp.zoomTo(vp.imageToViewportZoom(1))
  }

  const save = () => {
    const v = viewer.current
    if (!v) return
    const r = v.viewport.viewportToImageRectangle(v.viewport.getBounds(true))
    const x0 = Math.max(0, r.x)
    const y0 = Math.max(0, r.y)
    const x1 = Math.min(info.width, r.x + r.width)
    const y1 = Math.min(info.height, r.y + r.height)
    if (x1 <= x0 || y1 <= y0) return
    let s = 1
    while (Math.max(x1 - x0, y1 - y0) / s > 8192) s *= 2
    setSaved(`${Math.round(x1 - x0)} × ${Math.round(y1 - y0)} px` +
             (s > 1 ? `, saved at 1:${s} or coarser` : ', saved at full resolution'))
    const a = document.createElement('a')
    a.href = api.viewRegionUrl(path, x0, y0, x1, y1)
    a.download = ''
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  return (
    <div className="dz">
      <div className="dz-stage" style={{ height }}>
        <div ref={host} className="dz-host" />
        <canvas ref={canvas} className="dz-overlay" />
        {label && <span className="tool-tag left">{label}</span>}
        {info && (
          <div className="dz-tools">
            <button onClick={() => zoomBy(1.6)} title="zoom in">+</button>
            <button onClick={() => zoomBy(1 / 1.6)} title="zoom out">−</button>
            <button onClick={oneToOne} title="one image pixel per screen pixel">1:1</button>
            <button onClick={home} title="whole image">Fit</button>
            <button onClick={save} title="download what is on screen as a PNG, at native resolution where it fits in 8192 px">
              Save view
            </button>
          </div>
        )}
        {!info && !err && (
          <div className="dz-wait">
            opening {path.split('/').pop()}…{wait >= 2 && (
              <div className="dim">
                first open of a large file builds its overview, one pass over the
                whole file ({wait} s). It is cached after that.
              </div>
            )}
          </div>
        )}
        {err && <div className="dz-wait bad">{err}</div>}
        {info && <Hud info={info} hover={hover} scale={scale} />}
      </div>
      {info && (
        <form className="dz-go" onSubmit={go}>
          {info.georef?.kind && (
            <select value={goMode} onChange={(e) => setGoMode(e.target.value)}>
              <option value="ll">lat, lon</option>
              <option value="px">x, y (px)</option>
            </select>
          )}
          <input value={goText} onChange={(e) => setGoText(e.target.value)}
                 placeholder={goMode === 'll' ? 'go to lat, lon  e.g. -89.9, 0' : 'go to x, y  e.g. 5000, 1200'} />
          <button className="btn ghost" type="submit">Go</button>
          {goErr && <span className="bad">{goErr}</span>}
          {saved && !goErr && <span className="dim">{saved}</span>}
        </form>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------------ */
/* Source and reference side by side, locked together through the tie points */

function parseMatches(text) {
  // matches.csv is written by Python's csv module: CRLF line endings
  const lines = text.trim().split(/\r?\n/)
  const head = lines[0].split(',').map((h) => h.trim())
  const col = (k) => head.indexOf(k)
  const [cx, cy, cu, cv, ci] = ['src_x', 'src_y', 'ref_x', 'ref_y', 'inlier'].map(col)
  const n = lines.length - 1
  const sx = new Float64Array(n); const sy = new Float64Array(n)
  const rx = new Float64Array(n); const ry = new Float64Array(n)
  const inl = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const p = lines[i + 1].split(',')
    sx[i] = +p[cx]; sy[i] = +p[cy]; rx[i] = +p[cu]; ry[i] = +p[cv]
    inl[i] = +p[ci] ? 1 : 0
  }
  const idx = []
  for (let i = 0; i < n; i++) if (inl[i]) idx.push(i)
  // Reference pixels per source pixel over all inliers: the fallback scale
  // when the nearest tie points are too few or too collinear to fit locally.
  const spread = (a) => {
    let m = 0; for (const i of idx) m += a[i]; m /= idx.length || 1
    let s = 0; for (const i of idx) s += (a[i] - m) ** 2
    return Math.sqrt(s / (idx.length || 1))
  }
  const gs = idx.length > 2
    ? Math.sqrt((spread(rx) ** 2 + spread(ry) ** 2) / Math.max(1e-9, spread(sx) ** 2 + spread(sy) ** 2))
    : 1
  return { n, sx, sy, rx, ry, inl, idx: Int32Array.from(idx), gs }
}

/* Map (px, py) from one image to the other with an affine fitted to the k
 * inliers nearest to it. Local, because a long strip against a mosaic is not
 * one global affine; fitted, because a single nearest point would jitter. */
function localMap(X, Y, U, V, idx, px, py, fallbackScale, k = 24) {
  const n = idx.length
  if (!n) return null
  const kk = Math.min(k, n)
  const bd = new Float64Array(kk).fill(Infinity)
  const bi = new Int32Array(kk).fill(-1)
  for (let j = 0; j < n; j++) {
    const i = idx[j]
    const dx = X[i] - px
    const dy = Y[i] - py
    const d = dx * dx + dy * dy
    if (d >= bd[kk - 1]) continue
    let p = kk - 1
    while (p > 0 && bd[p - 1] > d) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p-- }
    bd[p] = d; bi[p] = i
  }
  let mx = 0, my = 0, mu = 0, mv = 0
  for (const i of bi) { mx += X[i]; my += Y[i]; mu += U[i]; mv += V[i] }
  mx /= kk; my /= kk; mu /= kk; mv /= kk
  let sxx = 0, sxy = 0, syy = 0, sxu = 0, syu = 0, sxv = 0, syv = 0
  for (const i of bi) {
    const x = X[i] - mx, y = Y[i] - my, u = U[i] - mu, v = V[i] - mv
    sxx += x * x; sxy += x * y; syy += y * y
    sxu += x * u; syu += y * u; sxv += x * v; syv += y * v
  }
  const det = sxx * syy - sxy * sxy
  if (kk >= 4 && det > 1e-3 * sxx * syy && sxx > 0 && syy > 0) {
    const a = (sxu * syy - syu * sxy) / det
    const b = (syu * sxx - sxu * sxy) / det
    const c = (sxv * syy - syv * sxy) / det
    const d = (syv * sxx - sxv * sxy) / det
    const s = Math.sqrt(Math.abs(a * d - b * c))
    if (Number.isFinite(s) && s > 0) {
      return { u: mu + a * (px - mx) + b * (py - my), v: mv + c * (px - mx) + d * (py - my), s }
    }
  }
  return { u: mu + (px - mx) * fallbackScale, v: mv + (py - my) * fallbackScale, s: fallbackScale }
}

export function Explore({ job, showOutliers }) {
  const [pts, setPts] = useState(null)
  const [ptsErr, setPtsErr] = useState(null)
  const [link, setLink] = useState(true)
  const [showPts, setShowPts] = useState(true)
  const [viewers, setViewers] = useState({ src: null, ref: null })
  const leader = useRef('src')

  // Keyed on the job by the parent, so a new run mounts a fresh Explore rather
  // than briefly linking the views through the previous run's tie points.
  useEffect(() => {
    let dead = false
    api.toolText(job.id, 'matches.csv')
      .then((t) => { if (!dead) setPts(parseMatches(t)) })
      .catch((e) => { if (!dead) setPtsErr(e.message || String(e)) })
    return () => { dead = true }
  }, [job.id])

  const srcPts = useMemo(() => pts && showPts && { x: pts.sx, y: pts.sy, inl: pts.inl, n: pts.n }, [pts, showPts])
  const refPts = useMemo(() => pts && showPts && { x: pts.rx, y: pts.ry, inl: pts.inl, n: pts.n }, [pts, showPts])

  // Whichever view the pointer is over leads; the other follows through the
  // local tie-point fit. Only the leader's changes propagate, so the follower
  // moving does not bounce back.
  useEffect(() => {
    const a = viewers.src
    const b = viewers.ref
    if (!a || !b || !pts || !pts.idx.length || !link) return
    const follow = (from) => () => {
      if (leader.current !== from) return
      const L = from === 'src' ? a : b
      const F = from === 'src' ? b : a
      if (!L.world.getItemCount() || !F.world.getItemCount()) return
      const lv = L.viewport
      const fv = F.viewport
      const c = lv.viewportToImageCoordinates(lv.getCenter(true))
      const z = lv.viewportToImageZoom(lv.getZoom(true))
      // tie points use pixel-centre = integer; the viewer uses pixel-edge = integer
      const m = from === 'src'
        ? localMap(pts.sx, pts.sy, pts.rx, pts.ry, pts.idx, c.x - 0.5, c.y - 0.5, pts.gs)
        : localMap(pts.rx, pts.ry, pts.sx, pts.sy, pts.idx, c.x - 0.5, c.y - 0.5, 1 / pts.gs)
      if (!m) return
      fv.panTo(fv.imageToViewportCoordinates(new OpenSeadragon.Point(m.u + 0.5, m.v + 0.5)), true)
      fv.zoomTo(fv.imageToViewportZoom(z / m.s), null, true)
    }
    const fa = follow('src')
    const fb = follow('ref')
    a.addHandler('viewport-change', fa)
    b.addHandler('viewport-change', fb)
    const sync = () => { leader.current = 'src'; fa() }
    if (a.world.getItemCount() && b.world.getItemCount()) sync()
    else { a.addOnceHandler('open', sync); b.addOnceHandler('open', sync) }
    return () => {
      a.removeHandler('viewport-change', fa)
      b.removeHandler('viewport-change', fb)
      a.removeHandler('open', sync)
      b.removeHandler('open', sync)
    }
  }, [viewers, pts, link])

  const setSrc = useMemo(() => (v) => setViewers((o) => ({ ...o, src: v })), [])
  const setRef = useMemo(() => (v) => setViewers((o) => ({ ...o, ref: v })), [])
  const nIn = pts ? pts.idx.length : 0

  return (
    <div>
      <div className="dz-bar">
        <label className="check">
          <input type="checkbox" checked={link} disabled={!nIn}
                 onChange={(e) => setLink(e.target.checked)} />
          <span>Link views
            <span className="hint"> — move one, the other follows through the nearest tie points</span>
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={showPts} onChange={(e) => setShowPts(e.target.checked)} />
          <span>Show tie points
            <span className="hint">{pts ? ` — ${nIn} inliers of ${pts.n}` : ptsErr ? ` — ${ptsErr}` : ' — loading…'}</span>
          </span>
        </label>
      </div>
      <div className="dz-pair">
        <div onPointerEnter={() => { leader.current = 'src' }}
             onPointerDown={() => { leader.current = 'src' }}>
          <DeepZoom path={job.source} label="SOURCE" points={srcPts}
                    showOutliers={showOutliers} onViewer={setSrc} height="68vh" />
        </div>
        <div onPointerEnter={() => { leader.current = 'ref' }}
             onPointerDown={() => { leader.current = 'ref' }}>
          <DeepZoom path={job.reference} label="REFERENCE" points={refPts}
                    showOutliers={showOutliers} onViewer={setRef} height="68vh" />
        </div>
      </div>
    </div>
  )
}
