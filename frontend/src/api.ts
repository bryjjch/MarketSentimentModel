import type { SentimentRow } from './types'

const JSON_HEADERS = { 'Content-Type': 'application/json' }
const DEFAULT_SUGGEST_LIMIT = 10

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
    await res.text()
    throw new Error(`Could not load overview (${res.status}).`)
  }
  const data: unknown = await res.json()
  if (!Array.isArray(data)) {
    throw new Error('Overview response was unexpected.')
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
    await res.text()
    throw new Error(`Could not load this symbol (${res.status}).`)
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
  if (!res.ok) {
    let msg = res.statusText
    try {
      const errData = (await res.json()) as SentimentRow
      msg = errData.message ?? errData.error ?? msg
    } catch {
      /* non-JSON error body — fall through to statusText */
    }
    throw new Error(msg || `Request failed (${res.status})`)
  }
  return (await res.json()) as SentimentRow
}

export async function fetchTickerSuggestions(
  baseUrl: string,
  query: string,
  limit = DEFAULT_SUGGEST_LIMIT,
): Promise<string[]> {
  const q = query.trim().toUpperCase()
  if (!q) return []

  const url = new URL(joinUrl(baseUrl, '/tickers/suggest'))
  url.searchParams.set('q', q)
  url.searchParams.set('limit', String(Math.min(Math.max(limit, 1), 25)))

  const res = await fetch(url.toString(), { method: 'GET' })
  if (!res.ok) {
    await res.text()
    throw new Error(`Could not load ticker suggestions (${res.status}).`)
  }

  const data: unknown = await res.json()
  if (!data || typeof data !== 'object' || !Array.isArray((data as { suggestions?: unknown }).suggestions)) {
    throw new Error('Suggestion response was unexpected.')
  }
  return (data as { suggestions: string[] }).suggestions
}
