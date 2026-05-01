import {
  useCallback,
  useMemo,
  useState,
  type FormEvent,
} from 'react'
import {
  fetchSentimentCacheSymbol,
  postSentimentBySymbol,
} from './api'
import { Heatmap } from './components/Heatmap'
import { SearchPanel, type SearchResultContext } from './components/SearchPanel'
import { TickerDetailModal } from './components/TickerDetailModal'
import { upsertHeatmapExtra } from './mergeHeatmapRows'
import type { SentimentRow } from './types'

const SYMBOL_RE = /^[A-Z]{1,5}$/

function normalizeTicker(raw: string): string | null {
  const s = raw.trim().toUpperCase()
  if (!s) return null
  return SYMBOL_RE.test(s) ? s : null
}

function getApiBase(): string {
  const raw = import.meta.env.VITE_API_BASE_URL?.trim()
  return raw ? raw.replace(/\/$/, '') : ''
}

export default function App() {
  const apiBase = useMemo(() => getApiBase(), [])
  const [heatmapRows, setHeatmapRows] = useState<SentimentRow[]>([])
  const [heatmapExtras, setHeatmapExtras] = useState<SentimentRow[]>([])

  const [query, setQuery] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchResult, setSearchResult] = useState<SentimentRow | null>(null)
  const [searchSource, setSearchSource] = useState<SearchResultContext>(null)

  const [detailRow, setDetailRow] = useState<SentimentRow | null>(null)

  const onSearch = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()
      setSearchError(null)

      if (!apiBase) {
        setSearchError('FinSense is not connected. Try again later.')
        return
      }

      const sym = normalizeTicker(query)
      if (!sym) {
        setSearchError('Enter a valid ticker (1-5 letters, e.g. AAPL).')
        setSearchResult(null)
        setSearchSource(null)
        return
      }

      const fromGrid = heatmapRows.find(
        (r) => r.symbol.toUpperCase() === sym,
      )
      if (fromGrid) {
        setSearchResult(fromGrid)
        setSearchSource('heatmap')
        setSearchLoading(false)
        return
      }

      setSearchLoading(true)
      setSearchResult(null)
      setSearchSource(null)

      try {
        const saved = await fetchSentimentCacheSymbol(apiBase, sym)
        if (saved) {
          setSearchResult(saved)
          setSearchSource('saved')
          return
        }

        const fresh = await postSentimentBySymbol(apiBase, sym)
        setSearchResult(fresh)
        setSearchSource('fresh')
      } catch (err: unknown) {
        setSearchError(err instanceof Error ? err.message : String(err))
        setSearchResult(null)
        setSearchSource(null)
      } finally {
        setSearchLoading(false)
      }
    },
    [apiBase, heatmapRows, query],
  )

  const onAddToHeatmap = useCallback(() => {
    if (!searchResult) return
    setHeatmapExtras((prev) => upsertHeatmapExtra(prev, searchResult))
  }, [searchResult])

  const onDetailRowUpdate = useCallback((row: SentimentRow) => {
    setDetailRow(row)
    setHeatmapExtras((prev) => upsertHeatmapExtra(prev, row))
  }, [])

  return (
    <div className="min-h-svh text-zinc-100">
      <div className="mx-auto max-w-7xl px-5 py-14 sm:px-8 sm:py-20 lg:px-12">
        <header className="border-b border-white/[0.06] pb-12 text-left sm:pb-14">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between fs-rise">
            <div className="max-w-2xl space-y-4">
              <p className="font-mono text-[11px] font-medium uppercase tracking-[0.35em] text-zinc-500">
                FinSense
              </p>
              <h1 className="text-4xl font-semibold leading-[1.05] tracking-tight text-white sm:text-5xl">
                Market sentiment
              </h1>
              <p className="max-w-xl text-pretty text-base leading-relaxed text-zinc-500">
                News-driven scores per symbol, distilled into a monochrome
                overview. Search any ticker or open a tile for sources.
              </p>
            </div>
            <div className="hidden shrink-0 font-mono text-[10px] leading-relaxed text-zinc-600 sm:block sm:text-right">
              <div className="border border-white/[0.08] bg-white/[0.02] px-4 py-3">
                <div className="text-zinc-500">Signal</div>
                <div className="mt-1 text-zinc-400">Score −1 … +1</div>
                <div className="mt-2 text-zinc-600">Darker → bearish</div>
                <div className="text-zinc-600">Lighter → bullish</div>
              </div>
            </div>
          </div>
        </header>

        <div className="mt-14 space-y-16 sm:mt-16 sm:space-y-20">
          <SearchPanel
            className="fs-rise fs-rise-delay-2"
            query={query}
            onQueryChange={setQuery}
            onSubmit={onSearch}
            searchError={searchError}
            searchLoading={searchLoading}
            searchResult={searchResult}
            searchSource={searchSource}
            onAddToHeatmap={onAddToHeatmap}
          />

          <section className="fs-rise fs-rise-delay-3">
            <div className="mb-8 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-xs font-semibold uppercase tracking-[0.25em] text-zinc-500">
                  Overview
                </h2>
                <p className="mt-2 max-w-lg text-2xl font-semibold tracking-tight text-white">
                  Heatmap
                </p>
                <p className="mt-2 max-w-xl text-sm leading-relaxed text-zinc-500">
                  Tiles encode sentiment in grayscale. Open a symbol for
                  headlines and refresh. Add from search to pin extras on the
                  grid.
                </p>
              </div>
            </div>
            <Heatmap
              apiBase={apiBase}
              extraRows={heatmapExtras}
              onRowsChange={setHeatmapRows}
              onSelectSymbol={setDetailRow}
            />
          </section>
        </div>
      </div>

      <TickerDetailModal
        apiBase={apiBase}
        row={detailRow}
        onClose={() => setDetailRow(null)}
        onRowUpdate={onDetailRowUpdate}
      />
    </div>
  )
}
