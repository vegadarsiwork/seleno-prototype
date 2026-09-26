const BASE = ''

// FastAPI puts the reason in {"detail": ...}; show that, not the raw JSON.
async function failure(r) {
  const t = await r.text()
  try { return new Error(JSON.parse(t).detail || t || r.statusText) } catch { return new Error(t || r.statusText) }
}

async function jget(path) {
  const r = await fetch(BASE + path)
  if (!r.ok) throw await failure(r)
  return r.json()
}

async function jpost(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw await failure(r)
  return r.json()
}

export const getConfig = () => jget('/api/config')
export const getProduct = (ts) => jget(`/api/ohrc/product/${ts}`)
export const getOverlaps = () => jget('/api/ohrc/overlaps')
export const getAlignment = (a, b) => jget(`/api/ohrc/alignment/${a}/${b}`)
export const getWindows = (a, b, size, offset = true) =>
  jget(`/api/ohrc/windows/${a}/${b}?size=${size}&limit=8&apply_coarse_offset=${offset}`)

export const register = (payload) => jpost('/api/register', payload)
export const compare = (payload) => jpost('/api/compare', payload)
export const runLegacy = (pair, options) => jpost('/api/run', { pair, options })

export const imageUrl = (runId, name) => `${BASE}/api/image/${runId}/${name}`
export const stripThumbUrl = (ts, maxSide = 900) =>
  `${BASE}/api/ohrc/thumbnail/${ts}?max_side=${maxSide}`
export const pairPreviewUrl = (a, b, s0, l0, size, which, offset = true, maxSide = 640) =>
  `${BASE}/api/pair-preview?source=${a}&reference=${b}&sample0=${s0}&line0=${l0}` +
  `&size=${size}&which=${which}&apply_coarse_offset=${offset}&max_side=${maxSide}`
export const legacyPairImageUrl = (pairId, which, maxSide = 640) =>
  `${BASE}/api/pair-image/${pairId}/${which}?max_side=${maxSide}`

// --- Phase 7 registration tool ------------------------------------------- //
export const toolFiles = () => jget('/api/tool/files')
export const toolRegister = (payload) => jpost('/api/tool/register', payload)
export const toolProfiles = () => jget('/api/tool/profiles')
export const toolJob = (id) => jget(`/api/tool/jobs/${id}`)
export const toolQuality = (id) => jget(`/api/tool/jobs/${id}/quality`)
export const toolFile = (id, name) => jget(`/api/tool/jobs/${id}/file/${name}`)
export const toolFileUrl = (id, name) => `${BASE}/api/tool/jobs/${id}/file/${name}`
export const toolDownloadUrl = (id) => `${BASE}/api/tool/jobs/${id}/download`
export const toolText = async (id, name) => {
  const r = await fetch(toolFileUrl(id, name))
  if (!r.ok) throw await failure(r)
  return r.text()
}

// --- viewing any input at any zoom --------------------------------------- //
const q = encodeURIComponent
export const viewInfo = (path) => jget(`/api/tool/view/info?path=${q(path)}`)
export const viewTileUrl = (path, s, x, y, v) =>
  `${BASE}/api/tool/view/tile?path=${q(path)}&s=${s}&x=${x}&y=${y}&v=${v}`
export const viewRegionUrl = (path, x0, y0, x1, y1) =>
  `${BASE}/api/tool/view/region?path=${q(path)}` +
  `&x0=${Math.floor(x0)}&y0=${Math.floor(y0)}&x1=${Math.ceil(x1)}&y1=${Math.ceil(y1)}`

// --- uploads ------------------------------------------------------------- //
// XHR rather than fetch: fetch cannot report upload progress, and a 6 GB file
// with no progress bar looks exactly like a hung page.
export const uploadFile = (batch, name, file, onProgress) =>
  new Promise((resolve, reject) => {
    const x = new XMLHttpRequest()
    x.open('PUT', `${BASE}/api/tool/upload?batch=${q(batch)}&name=${q(name)}`)
    x.upload.onprogress = (e) => onProgress && onProgress(e.loaded)
    x.onload = () => {
      if (x.status >= 200 && x.status < 300) return resolve(JSON.parse(x.responseText))
      let msg = x.responseText || x.statusText
      try { msg = JSON.parse(x.responseText).detail || msg } catch { /* not JSON */ }
      reject(new Error(msg))
    }
    x.onerror = () => reject(new Error('network error during upload'))
    x.onabort = () => reject(new Error('upload aborted'))
    x.send(file)
  })
