const BASE = ''

async function jget(path) {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error((await r.text()) || r.statusText)
  return r.json()
}

async function jpost(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error((await r.text()) || r.statusText)
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
