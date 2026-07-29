import { useEffect, useMemo, useState } from 'react'
import { Card } from 'primereact/card'
import { Message } from 'primereact/message'
import { Skeleton } from 'primereact/skeleton'
import {
  fetchSentimentCacheSymbol,
  postSentimentBySymbol,
} from '../api'
import { DEFAULT_HEATMAP_SYMBOLS } from '../defaultHeatmapSymbols'
import { formatScore } from '../formatScore'
import { mergeHeatmapRows } from '../mergeHeatmapRows'
import type { SentimentRow } from '../types'
import { scoreToAccent, scoreToCardStyle } from '../scoreColor'

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

  return (
    <Card
      className="border border-[color:var(--color-fs-border)] bg-[color:var(--color-fs-surface)]/95 shadow-sm"
      pt={{
        body: { className: 'p-5 sm:p-6' },
        content: { className: 'p-0' },
      }}
    >
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Heatmap
          </h2>
          <p className="mt-1 text-lg font-semibold tracking-tight text-[color:var(--color-fs-text)]">
            Sentiment overview
          </p>
          <p className="mt-1 text-sm text-[color:var(--color-fs-text-subtle)]">
            Click a tile for headlines and a fresh refresh.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-[color:var(--color-fs-surface-muted)] px-3 py-1.5 text-[11px] font-medium text-[color:var(--color-fs-text-subtle)]">
          <i className="pi pi-info-circle text-[color:var(--color-fs-blue)]" />
          <span>Score range −1 … +1</span>
        </div>
      </div>

      <div className="mt-5">
        {loading ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} height="9rem" borderRadius="0.75rem" />
            ))}
          </div>
        ) : error && mergedRows.length === 0 ? (
          <Message
            severity="error"
            text={error}
            className="w-full"
          />
        ) : mergedRows.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-[color:var(--color-fs-border-strong)] bg-[color:var(--color-fs-surface-muted)] px-6 py-12 text-center">
            <i className="pi pi-inbox text-2xl text-[color:var(--color-fs-text-subtle)]" />
            <p className="text-sm text-[color:var(--color-fs-text-muted)]">
              No symbols yet. Search above to add tickers to your heatmap.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {error ? (
              <Message
                severity="warn"
                text={error}
                className="w-full"
              />
            ) : null}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {mergedRows.map((row, i) => {
                const accent = scoreToAccent(row.score)
                return (
                  <button
                    key={row.symbol}
                    type="button"
                    onClick={() => onSelectSymbol?.(row)}
                    className="group fs-tile-in relative flex flex-col rounded-xl border p-4 text-left shadow-sm transition-[transform,box-shadow] duration-200 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-fs-blue)] focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--color-fs-surface)] hover:-translate-y-0.5 hover:shadow-md sm:p-5"
                    style={{
                      ...scoreToCardStyle(row.score),
                      ['--fs-stagger' as string]: `${Math.min(i, 24) * 38}ms`,
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-sm font-semibold tracking-tight">
                        {row.symbol}
                      </span>
                      <span
                        className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
                        style={{
                          backgroundColor: 'rgb(6 11 20 / 55%)',
                          color: accent.fg,
                          border: `1px solid ${accent.border}`,
                        }}
                      >
                        {accent.label}
                      </span>
                    </div>
                    <span className="mt-3 font-mono text-2xl font-semibold tabular-nums tracking-tight">
                      {formatScore(row.score)}
                    </span>
                    {row.label ? (
                      <span className="mt-3 inline-block w-fit rounded-md bg-slate-950/45 px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider text-slate-100">
                        {row.label}
                      </span>
                    ) : null}
                    <span className="mt-auto pt-3 text-[10px] font-semibold text-slate-100 opacity-0 transition-opacity group-hover:opacity-100">
                      Open details →
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

