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
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-4 sm:items-center"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="max-h-[min(90vh,720px)] w-full max-w-lg overflow-y-auto rounded-2xl border border-slate-600 bg-slate-900 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ticker-detail-title"
      >
        <div className="sticky top-0 flex items-center justify-between border-b border-slate-700/80 bg-slate-900/95 px-5 py-4 backdrop-blur">
          <h2
            id="ticker-detail-title"
            className="font-mono text-xl font-semibold text-white"
          >
            {display.symbol}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-3 py-1.5 text-sm text-slate-300 transition hover:bg-slate-800 hover:text-white"
          >
            Close
          </button>
        </div>

        <div className="space-y-5 px-5 py-5">
          <div
            className="rounded-xl border border-white/10 p-5 shadow-inner"
            style={scoreToCardStyle(display.score)}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-sm font-medium opacity-90">Sentiment</span>
              <span className="font-mono text-3xl font-bold tabular-nums">
                {formatScore(display.score)}
              </span>
            </div>
            {display.label ? (
              <p className="mt-2 text-sm capitalize opacity-90">{display.label}</p>
            ) : null}
            {typeof display.article_count === 'number' ? (
              <p className="mt-1 text-sm opacity-80">
                Stories in this score: {display.article_count}
              </p>
            ) : null}
          </div>

          {loading ? (
            <div className="flex items-center gap-3 text-slate-300">
              <div
                className="h-7 w-7 shrink-0 animate-spin rounded-full border-2 border-violet-400 border-t-transparent"
                aria-hidden
              />
              <span>Loading related stories...</span>
            </div>
          ) : null}

          {error ? (
            <p className="text-sm text-red-300" role="alert">
              {error}
            </p>
          ) : null}

          <div>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-medium text-slate-300">
                Related coverage
              </h3>
              {apiBase ? (
                <button
                  type="button"
                  disabled={refreshing}
                  onClick={runLatestAnalysis}
                  className="text-xs font-medium text-violet-300 underline decoration-violet-500/40 underline-offset-2 hover:text-violet-200 disabled:opacity-50"
                >
                  {refreshing ? 'Updating...' : 'Refresh from latest news'}
                </button>
              ) : null}
            </div>
            {headlines.length > 0 ? (
              <ul className="mt-3 space-y-2.5">
                {headlines.map((h, i) => (
                  <li key={`${h.url}-${i}`}>
                    <a
                      href={h.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-violet-300 underline decoration-violet-500/50 underline-offset-2 transition hover:text-violet-200"
                    >
                      {h.title || h.url}
                    </a>
                  </li>
                ))}
              </ul>
            ) : !loading ? (
              <p className="mt-3 text-sm text-slate-500">
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
