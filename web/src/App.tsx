import {
  useCallback,
  useMemo,
  useState,
  type FormEvent,
} from 'react'
import { Button } from 'primereact/button'
import {
  fetchTickerSuggestions,
  fetchSentimentCacheSymbol,
  postSentimentBySymbol,
} from './api'
import { DashboardLayout } from './components/DashboardLayout'
import { Heatmap } from './components/Heatmap'
import { KpiCards } from './components/KpiCards'
import { PlaceholderView } from './components/PlaceholderView'
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

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  overview: { subtitle: 'Dashboard', title: 'Market Sentiment' },
  heatmap: { subtitle: 'Dashboard', title: 'Heatmap' },
  search: { subtitle: 'Dashboard', title: 'Lookup' },
  watchlist: { subtitle: 'Dashboard', title: 'Watchlist' },
  news: { subtitle: 'Dashboard', title: 'News feed' },
  settings: { subtitle: 'Dashboard', title: 'Settings' },
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
  const [activeNav, setActiveNav] = useState('overview')

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
    setActiveNav('heatmap')
  }, [searchResult])

  const onDetailRowUpdate = useCallback((row: SentimentRow) => {
    setDetailRow(row)
    setHeatmapExtras((prev) => upsertHeatmapExtra(prev, row))
  }, [])

  const onDetailModalClose = useCallback(() => {
    setDetailRow(null)
  }, [])

  const headerRight = (
    <Button
      icon="pi pi-bell"
      rounded
      text
      severity="secondary"
      aria-label="Notifications"
    />
  )

  const heatmap = (
    <Heatmap
      apiBase={apiBase}
      extraRows={heatmapExtras}
      onRowsChange={setHeatmapRows}
      onSelectSymbol={setDetailRow}
    />
  )

  const searchPanel = (
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
  )

  const meta = PAGE_META[activeNav] ?? PAGE_META.overview

  let body
  if (activeNav === 'overview') {
    body = (
      <div className="space-y-6">
        <section className="fs-rise">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
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
            </div>
          </div>
          <div className="mt-5">
            <KpiCards rows={heatmapRows} />
          </div>
        </section>
        <section className="grid grid-cols-1 gap-6 xl:grid-cols-5">
          <div className="xl:col-span-3">{heatmap}</div>
          <div className="xl:col-span-2">{searchPanel}</div>
        </section>
      </div>
    )
  } else if (activeNav === 'heatmap') {
    body = (
      <div className="space-y-6">
        <section className="fs-rise">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Heatmap
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[color:var(--color-fs-text)] sm:text-3xl">
            All tracked symbols
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-[color:var(--color-fs-text-subtle)]">
            Full-width grid of every symbol you're tracking. Click a tile for
            headlines and a fresh refresh.
          </p>
          <div className="mt-5">
            <KpiCards rows={heatmapRows} />
          </div>
        </section>
        <section>{heatmap}</section>
      </div>
    )
  } else if (activeNav === 'search') {
    body = (
      <div className="space-y-6">
        <section className="fs-rise">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Lookup
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[color:var(--color-fs-text)] sm:text-3xl">
            Look up any US equity
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-[color:var(--color-fs-text-subtle)]">
            Search any ticker to see its latest sentiment score, supporting
            headlines, and add it to your heatmap.
          </p>
        </section>
        <div className="mx-auto w-full max-w-3xl">{searchPanel}</div>
        {/* Keep the heatmap mounted so it doesn't have to reload its default rows when you switch back. */}
        <div className="hidden">{heatmap}</div>
      </div>
    )
  } else if (activeNav === 'watchlist') {
    body = (
      <div className="space-y-6">
        <section className="fs-rise">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Watchlist
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[color:var(--color-fs-text)] sm:text-3xl">
            Your saved symbols
          </h1>
        </section>
        <PlaceholderView
          icon="pi-bookmark"
          title="Watchlist isn't wired up yet"
          description="Custom watchlists with alerts and notes will live here. For now, search a symbol and use 'Add to heatmap' to keep it in view."
        />
        <div className="hidden">{heatmap}</div>
      </div>
    )
  } else if (activeNav === 'news') {
    body = (
      <div className="space-y-6">
        <section className="fs-rise">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            News feed
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[color:var(--color-fs-text)] sm:text-3xl">
            Latest market coverage
          </h1>
        </section>
        <PlaceholderView
          icon="pi-megaphone"
          title="The news feed is on the way"
          description="A cross-symbol view of the headlines driving today's scores. For now, open any heatmap tile to see the stories behind a symbol."
        />
        <div className="hidden">{heatmap}</div>
      </div>
    )
  } else if (activeNav === 'settings') {
    body = (
      <div className="space-y-6">
        <section className="fs-rise">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Settings
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-[color:var(--color-fs-text)] sm:text-3xl">
            Preferences
          </h1>
        </section>
        <PlaceholderView
          icon="pi-cog"
          title="No settings to configure yet"
          description="API endpoint, default symbols, and theme preferences will live here once they're wired up."
        />
        <div className="hidden">{heatmap}</div>
      </div>
    )
  }

  return (
    <DashboardLayout
      active={activeNav}
      onNavigate={setActiveNav}
      headerRight={headerRight}
      title={meta.title}
      subtitle={meta.subtitle}
    >
      {body}
      <TickerDetailModal
        apiBase={apiBase}
        row={detailRow}
        onClose={onDetailModalClose}
        onRowUpdate={onDetailRowUpdate}
      />
    </DashboardLayout>
  )
}
