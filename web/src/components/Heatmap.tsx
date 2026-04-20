import { useEffect, useState } from 'react'
import { fetchSentimentCacheList } from '../api'
import type { SentimentRow } from '../types'
import { scoreToCardStyle } from '../scoreColor'

type Props = {
  apiBase: string
  onRowsChange?: (rows: SentimentRow[]) => void
}

function formatScore(score: number): string {
  if (!Number.isFinite(score)) return '—'
  const s = score.toFixed(3)
  return score >= 0 ? `+${s}` : s
}

export function Heatmap({ apiBase, onRowsChange }: Props) {
  const [rows, setRows] = useState<SentimentRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!apiBase) {
      setLoading(false)
      setError(
        'Set VITE_API_BASE_URL in web/.env (HTTP API invoke URL from Terraform).',
      )
      setRows([])
      onRowsChange?.([])
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetchSentimentCacheList(apiBase, 500)
      .then((data) => {
        if (!cancelled) {
          setRows(data)
          onRowsChange?.(data)
        }
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [apiBase, onRowsChange])

  if (loading) {
    return (
      <div className="flex min-h-[200px] items-center justify-center rounded-xl border border-slate-700/50 bg-slate-900/40 px-4 py-16">
        <div
          className="h-10 w-10 animate-spin rounded-full border-2 border-violet-400 border-t-transparent"
          aria-hidden
        />
        <span className="sr-only">Loading heatmap</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-500/40 bg-red-950/30 px-4 py-6 text-left text-red-200">
        <p className="font-medium">Could not load cached tickers</p>
        <p className="mt-1 text-sm text-red-200/80">{error}</p>
      </div>
    )
  }

  if (rows.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700/50 bg-slate-900/30 px-4 py-12 text-slate-400">
        No cached symbols yet. Try searching for a ticker below, or wait for the
        scheduled cache refresh.
      </div>
    )
  }

  const sorted = [...rows].sort((a, b) => a.symbol.localeCompare(b.symbol))

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
      {sorted.map((row) => (
        <article
          key={row.symbol}
          className="flex flex-col rounded-xl border border-white/10 p-4 text-left shadow-lg transition hover:scale-[1.02] hover:shadow-xl"
          style={scoreToCardStyle(row.score)}
        >
          <span className="font-mono text-lg font-semibold tracking-tight">
            {row.symbol}
          </span>
          <span className="mt-1 font-mono text-2xl font-bold tabular-nums">
            {formatScore(row.score)}
          </span>
          {row.label ? (
            <span className="mt-2 inline-block w-fit rounded-md bg-black/20 px-2 py-0.5 text-xs capitalize">
              {row.label}
            </span>
          ) : null}
        </article>
      ))}
    </div>
  )
}
