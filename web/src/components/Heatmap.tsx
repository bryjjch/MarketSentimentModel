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
      <div className="flex min-h-[220px] items-center justify-center border border-white/[0.06] bg-white/[0.02] px-6 py-20">
        <div
          className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-700 border-t-white"
          aria-hidden
        />
        <span className="sr-only">Loading heatmap</span>
      </div>
    )
  }

  if (error && mergedRows.length === 0) {
    return (
      <div className="border border-white/15 bg-black/50 px-6 py-8 text-left">
        <p className="font-medium text-white">Could not load the overview</p>
        <p className="mt-2 font-mono text-xs leading-relaxed text-zinc-400">
          {error}
        </p>
      </div>
    )
  }

  if (mergedRows.length === 0) {
    return (
      <div className="border border-white/[0.06] bg-white/[0.02] px-6 py-16 text-center">
        <p className="text-sm text-zinc-500">
          No symbols yet. Search above, then add tickers to your heatmap.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {error ? (
        <div className="border border-white/10 bg-black/40 px-4 py-3 font-mono text-xs text-zinc-400">
          {mergedRows.length > 0
            ? error
            : 'Overview could not be refreshed; showing symbols you have added.'}
        </div>
      ) : null}
      <div className="grid grid-cols-2 gap-px bg-white/[0.06] sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {mergedRows.map((row) => (
          <button
            key={row.symbol}
            type="button"
            onClick={() => onSelectSymbol?.(row)}
            className="group relative flex flex-col border border-transparent bg-black/20 p-5 text-left transition focus:outline-none focus-visible:ring-1 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-black hover:bg-black/40 sm:p-6"
            style={scoreToCardStyle(row.score)}
          >
            <span className="font-mono text-sm font-medium tracking-tight opacity-80">
              {row.symbol}
            </span>
            <span className="mt-3 font-mono text-2xl font-semibold tabular-nums tracking-tight">
              {formatScore(row.score)}
            </span>
            {row.label ? (
              <span className="mt-4 inline-block w-fit border border-current/15 px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider opacity-90">
                {row.label}
              </span>
            ) : null}
            <span className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/0 transition group-hover:ring-white/10" />
          </button>
        ))}
      </div>
    </div>
  )
}
