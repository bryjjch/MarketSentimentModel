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
import { SearchPanel } from './components/SearchPanel'
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

  const [query, setQuery] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchResult, setSearchResult] = useState<SentimentRow | null>(null)
  const [searchSource, setSearchSource] = useState<
    'heatmap' | 'cached' | 'live' | null
  >(null)

  const onSearch = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()
      setSearchError(null)

      if (!apiBase) {
        setSearchError('Configure VITE_API_BASE_URL first.')
        return
      }

      const sym = normalizeTicker(query)
      if (!sym) {
        setSearchError('Enter a valid ticker (1–5 letters, e.g. AAPL).')
        setSearchResult(null)
        setSearchSource(null)
        return
      }

      const fromGrid = heatmapRows.find((r) => r.symbol === sym)
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
        const cached = await fetchSentimentCacheSymbol(apiBase, sym)
        if (cached) {
          setSearchResult(cached)
          setSearchSource('cached')
          return
        }

        const live = await postSentimentBySymbol(apiBase, sym)
        setSearchResult(live)
        setSearchSource('live')
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

  return (
    <div className="min-h-svh bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
        <header className="text-center">
          <p className="text-sm font-medium uppercase tracking-widest text-violet-400/90">
            FinSense
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Sentiment overview
          </h1>
          <p className="mx-auto mt-3 max-w-2xl text-pretty text-slate-400">
            Cached sentiment scores across your watchlist, plus on-demand lookups
            for any symbol.
          </p>
        </header>

        <div className="mt-10 space-y-12">
          <SearchPanel
            query={query}
            onQueryChange={setQuery}
            onSubmit={onSearch}
            searchError={searchError}
            searchLoading={searchLoading}
            searchResult={searchResult}
            searchSource={searchSource}
          />

          <section>
            <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">Global heatmap</h2>
                <p className="text-sm text-slate-400">
                  Colors map score (bearish → bullish). Data from DynamoDB cache.
                </p>
              </div>
            </div>
            <Heatmap apiBase={apiBase} onRowsChange={setHeatmapRows} />
          </section>
        </div>
      </div>
    </div>
  )
}
