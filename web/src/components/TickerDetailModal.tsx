import { useEffect, useState } from 'react'
import { fetchSentimentCacheSymbol, postSentimentBySymbol } from '../api'
import { formatScore } from '../formatScore'
import { scoreToCardStyle } from '../scoreColor'
import type { SentimentRow } from '../types'

type Props = {
  apiBase: string
  row: SentimentRow | null
  onClose: () => void
  onRowUpdate?: (row: SentimentRow) => void
}

export function TickerDetailModal({
  apiBase,
  row,
  onClose,
  onRowUpdate,
}: Props) {
  const [detail, setDetail] = useState<SentimentRow | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!row) {
      setDetail(null)
      setError(null)
      setLoading(false)
      return
    }

    setDetail(row)
    setError(null)

    const hasHeadlines =
      row.recent_headlines && row.recent_headlines.length > 0
    if (hasHeadlines || !apiBase) {
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    fetchSentimentCacheSymbol(apiBase, row.symbol)
      .then((r) => {
        if (cancelled) return
        if (r) setDetail(r)
      })
      .catch(() => {
        if (!cancelled) setError('Could not load story links for this symbol.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [row, apiBase])

  useEffect(() => {
    if (!row) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [row, onClose])

  if (!row) return null

  const display = detail ?? row

  async function runLatestAnalysis() {
    if (!apiBase) return
    setRefreshing(true)
    setError(null)
    try {
      const next = await postSentimentBySymbol(apiBase, display.symbol)
      setDetail(next)
      onRowUpdate?.(next)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRefreshing(false)
    }
  }

  const headlines = display.recent_headlines ?? []

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/80 p-4 backdrop-blur-sm sm:items-center"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="max-h-[min(90vh,720px)] w-full max-w-lg overflow-y-auto border border-white/[0.1] bg-[#080808] shadow-2xl shadow-black"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ticker-detail-title"
      >
        <div className="sticky top-0 flex items-center justify-between border-b border-white/[0.06] bg-[#080808]/95 px-6 py-5 backdrop-blur-md">
          <h2
            id="ticker-detail-title"
            className="font-mono text-lg font-semibold tracking-tight text-white"
          >
            {display.symbol}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="border border-white/10 px-3 py-1.5 font-mono text-xs uppercase tracking-wider text-zinc-400 transition hover:border-white/25 hover:text-white"
          >
            Close
          </button>
        </div>

        <div className="space-y-8 px-6 py-8">
          <div className="border p-6" style={scoreToCardStyle(display.score)}>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-xs font-semibold uppercase tracking-[0.2em] opacity-80">
                Sentiment
              </span>
              <span className="font-mono text-3xl font-semibold tabular-nums tracking-tight">
                {formatScore(display.score)}
              </span>
            </div>
            {display.label ? (
              <p className="mt-3 text-xs font-medium uppercase tracking-wider opacity-90">
                {display.label}
              </p>
            ) : null}
            {typeof display.article_count === 'number' ? (
              <p className="mt-2 font-mono text-xs opacity-80">
                Stories in this score: {display.article_count}
              </p>
            ) : null}
          </div>

          {loading ? (
            <div className="flex items-center gap-3 text-sm text-zinc-500">
              <div
                className="h-5 w-5 shrink-0 animate-spin rounded-full border-2 border-zinc-700 border-t-white"
                aria-hidden
              />
              <span>Loading related stories…</span>
            </div>
          ) : null}

          {error ? (
            <p
              className="border border-white/10 bg-black/40 px-4 py-3 font-mono text-xs text-zinc-300"
              role="alert"
            >
              {error}
            </p>
          ) : null}

          <div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-zinc-500">
                Related coverage
              </h3>
              {apiBase ? (
                <button
                  type="button"
                  disabled={refreshing}
                  onClick={runLatestAnalysis}
                  className="font-mono text-[10px] font-medium uppercase tracking-wider text-zinc-400 underline decoration-white/20 underline-offset-4 transition hover:text-white hover:decoration-white/50 disabled:opacity-40"
                >
                  {refreshing ? 'Updating…' : 'Refresh from latest news'}
                </button>
              ) : null}
            </div>
            {headlines.length > 0 ? (
              <ul className="mt-5 space-y-3 border-l border-white/10 pl-4">
                {headlines.map((h, i) => (
                  <li key={`${h.url}-${i}`}>
                    <a
                      href={h.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-zinc-300 underline decoration-white/15 underline-offset-[5px] transition hover:text-white hover:decoration-white/40"
                    >
                      {h.title || h.url}
                    </a>
                  </li>
                ))}
              </ul>
            ) : !loading ? (
              <p className="mt-5 font-mono text-xs text-zinc-600">
                No story links on file for this symbol yet. Try refreshing from
                latest news.
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}
