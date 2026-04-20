import type { SentimentRow } from './types'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

function joinUrl(base: string, path: string): string {
  const b = base.replace(/\/$/, '')
  const p = path.startsWith('/') ? path : `/${path}`
  return `${b}${p}`
}

export async function fetchSentimentCacheList(
  baseUrl: string,
  limit = 100,
): Promise<SentimentRow[]> {
  const url = new URL(joinUrl(baseUrl, '/sentiment/cache'))
  url.searchParams.set('limit', String(Math.min(Math.max(limit, 1), 500)))
  const res = await fetch(url.toString(), { method: 'GET' })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Cache list failed (${res.status}): ${text}`)
  }
  const data: unknown = await res.json()
  if (!Array.isArray(data)) {
    throw new Error('Cache list: expected JSON array')
  }
  return data as SentimentRow[]
}

export async function fetchSentimentCacheSymbol(
  baseUrl: string,
  symbol: string,
): Promise<SentimentRow | null> {
  const sym = encodeURIComponent(symbol.trim().toUpperCase())
  const res = await fetch(joinUrl(baseUrl, `/sentiment/cache/${sym}`), {
    method: 'GET',
  })
  if (res.status === 404) return null
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Cache read failed (${res.status}): ${text}`)
  }
  return (await res.json()) as SentimentRow
}

export async function postSentimentBySymbol(
  baseUrl: string,
  symbol: string,
): Promise<SentimentRow> {
  const res = await fetch(joinUrl(baseUrl, '/sentiment/by-symbol'), {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      symbol,
      options: { max_articles: 12, include_social: true },
    }),
  })
  const data = (await res.json()) as SentimentRow
  if (!res.ok) {
    const msg = data.message ?? data.error ?? res.statusText
    throw new Error(msg || `Request failed (${res.status})`)
  }
  return data
}
