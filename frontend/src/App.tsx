import {
  useCallback,
  useMemo,
  useState,
  type FormEvent,
} from 'react'
import {
  fetchTickerSuggestions,
  fetchSentimentCacheSymbol,
  postSentimentBySymbol,
} from './api'
import { DashboardLayout } from './components/DashboardLayout'
import { Heatmap } from './components/Heatmap'
import { KpiCards } from './components/KpiCards'
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
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [suggestionLoading, setSuggestionLoading] = useState(false)

  const [detailRow, setDetailRow] = useState<SentimentRow | null>(null)

  const runSearch = useCallback(
    async (rawQuery: string) => {
      setSearchError(null)

      if (!apiBase) {
        setSearchError('FinSense is not connected. Try again later.')
        return
      }

      const sym = normalizeTicker(rawQuery)
      if (!sym) {
        setSearchError('Enter a valid ticker (1-5 letters, e.g. AAPL).')
        setSearchResult(null)
        setSearchSource(null)
        return
      }

      const fromGrid = heatmapRows.find((r) => r.symbol.toUpperCase() === sym)
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
        const msg = err instanceof Error ? err.message : String(err)
        setSearchError(msg || 'Could not look up that ticker.')
        setSearchResult(null)
        setSearchSource(null)
      } finally {
        setSearchLoading(false)
      }
    },
    [apiBase, heatmapRows],
  )

  const onSearch = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()
      await runSearch(query)
    },
    [query, runSearch],
  )

  const onSuggestionRequest = useCallback(
    async (q: string) => {
      if (!apiBase || q.trim().length < 2) {
        setSuggestions([])
        setSuggestionLoading(false)
        return
      }
      setSuggestionLoading(true)
      try {
        const next = await fetchTickerSuggestions(apiBase, q, 10)
        setSuggestions(next)
      } catch {
        // Keep typeahead non-blocking; search submit still performs hard validation.
        setSuggestions([])
      } finally {
        setSuggestionLoading(false)
      }
    },
    [apiBase],
  )

  const onSuggestionSelect = useCallback(
    (symbol: string) => {
      setQuery(symbol)
      setSuggestions([])
      void runSearch(symbol)
    },
    [runSearch],
  )

  const onAddToHeatmap = useCallback(() => {
    if (!searchResult) return
    setHeatmapExtras((prev) => upsertHeatmapExtra(prev, searchResult))
  }, [searchResult])

  const onDetailRowUpdate = useCallback((row: SentimentRow) => {
    setDetailRow(row)
    setHeatmapExtras((prev) => upsertHeatmapExtra(prev, row))
  }, [])

  const onDetailModalClose = useCallback(() => {
    setDetailRow(null)
  }, [])

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <section className="fs-rise">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Overview
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[color:var(--color-fs-text)] sm:text-3xl">
            Market Sentiment Today
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-[color:var(--color-fs-text-subtle)]">
            News-driven sentiment scores per symbol. Search any ticker or open a
            tile for sources.
          </p>
        </section>

        <section className="grid grid-cols-1 items-start gap-6 xl:grid-cols-5">
          <div className="xl:col-span-3">
            <KpiCards rows={heatmapRows} />
          </div>
          <div className="xl:col-span-2">
            <SearchPanel
              query={query}
              onQueryChange={setQuery}
              onSubmit={onSearch}
              onSuggestionRequest={onSuggestionRequest}
              onSuggestionSelect={onSuggestionSelect}
              searchError={searchError}
              searchLoading={searchLoading}
              suggestionLoading={suggestionLoading}
              suggestions={suggestions}
              searchResult={searchResult}
              searchSource={searchSource}
              onAddToHeatmap={onAddToHeatmap}
            />
          </div>
        </section>

        <section>
          <Heatmap
            apiBase={apiBase}
            extraRows={heatmapExtras}
            onRowsChange={setHeatmapRows}
            onSelectSymbol={setDetailRow}
          />
        </section>
      </div>

      <TickerDetailModal
        apiBase={apiBase}
        row={detailRow}
        onClose={onDetailModalClose}
        onRowUpdate={onDetailRowUpdate}
      />
    </DashboardLayout>
  )
}
