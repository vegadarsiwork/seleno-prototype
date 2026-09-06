const BASE = ''

async function post(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error((await r.text()) || r.statusText)
  return r.json()
}

export async function getConfig() {
  const r = await fetch(BASE + '/api/config')
  if (!r.ok) throw new Error('config request failed')
  return r.json()
}

export const runPipeline = (pair, options) => post('/api/run', { pair, options })
export const compare = (pair, configs) => post('/api/compare', { pair, configs })

export const imageUrl = (runId, name) => `${BASE}/api/image/${runId}/${name}`
export const pairImageUrl = (pairId, which, maxSide = 420) =>
  `${BASE}/api/pair-image/${pairId}/${which}?max_side=${maxSide}`
