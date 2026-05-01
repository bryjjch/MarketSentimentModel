import { useEffect, useMemo, useState } from 'react'
import {
  fetchSentimentCacheSymbol,
  postSentimentBySymbol,
} from '../api'
import { DEFAULT_HEATMAP_SYMBOLS } from '../defaultHeatmapSymbols'
import { formatScore } from '../formatScore'
import { mergeHeatmapRows } from '../mergeHeatmapRows'
import type { SentimentRow } from '../types'
import { scoreToCardStyle } from '../scoreColor'

async function loadDefaultHeatmapRows(
  baseUrl: string,
): Promise<{ rows: SentimentRow[]; loadError: string | null }> {
  const rows: SentimentRow[] = []
  const failures: string[] = []
  for (const sym of DEFAULT_HEATMAP_SYMBOLS) {
    try {
      const cached = await fetchSentimentCacheSymbol(baseUrl, sym)
      rows.push(cached ?? (await postSentimentBySymbol(baseUrl, sym)))
    } catch (e: unknown) {
      failures.push(`${sym}: ${e instanceof Error ? e.message : String(e)}`)
    }
  }
  if (failures.length === 0) return { rows, loadError: null }
  if (failures.length === DEFAULT_HEATMAP_SYMBOLS.length) {
    return {
      rows: [],
      loadError: `Could not load default symbols. ${failures[0]}`,
    }
  }
  return {
    rows,
    loadError: `Some default symbols failed to load. ${failures.join('; ')}`,
  }
}

type Props = {
  apiBase: string
  extraRows?: SentimentRow[]
  onRowsChange?: (rows: SentimentRow[]) => void
  onSelectSymbol?: (row: SentimentRow) => void
}

export function Heatmap({
  apiBase,
  extraRows = [],
  onRowsChange,
  onSelectSymbol,
}: Props) {
  const [serverRows, setServerRows] = useState<SentimentRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const mergedRows = useMemo(
    () => mergeHeatmapRows(serverRows, extraRows),
    [serverRows, extraRows],
  )

  useEffect(() => {
    onRowsChange?.(mergedRows)
  }, [mergedRows, onRowsChange])

  useEffect(() => {
    if (!apiBase) {
      setLoading(false)
      setError('FinSense is not connected. Try again later or contact support.')
      setServerRows([])
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    loadDefaultHeatmapRows(apiBase)
      .then(({ rows, loadError }) => {
        if (!cancelled) {
          setServerRows(rows)
          setError(loadError)
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
  }, [apiBase])

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

  if (error && mergedRows.length === 0) {
    return (
      <div className="rounded-xl border border-red-500/40 bg-red-950/30 px-4 py-6 text-left text-red-200">
        <p className="font-medium">Could not load the market overview</p>
        <p className="mt-1 text-sm text-red-200/80">{error}</p>
      </div>
    )
  }

  if (mergedRows.length === 0) {
    return (
      <div className="rounded-xl border border-slate-700/50 bg-slate-900/30 px-4 py-12 text-slate-400">
        No symbols yet. Use the search above to look up a ticker, then add it
        to your heatmap.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {error ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-950/25 px-4 py-3 text-sm text-amber-100">
          {mergedRows.length > 0
            ? error
            : 'Overview could not be refreshed; showing symbols you have added.'}
        </div>
      ) : null}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
      {mergedRows.map((row) => (
        <button
          key={row.symbol}
          type="button"
          onClick={() => onSelectSymbol?.(row)}
          className="flex flex-col rounded-xl border border-white/10 p-4 text-left shadow-lg transition hover:scale-[1.02] hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-violet-500/50"
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
        </button>
      ))}
      </div>
    </div>
  )
}
