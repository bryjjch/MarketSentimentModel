import type { FormEvent } from 'react'
import { formatScore } from '../formatScore'
import { scoreToCardStyle } from '../scoreColor'
import type { SentimentRow } from '../types'

export type SearchResultContext = 'heatmap' | 'saved' | 'fresh' | null

type Props = {
  query: string
  onQueryChange: (q: string) => void
  onSubmit: (e: FormEvent) => void
  searchError: string | null
  searchLoading: boolean
  searchResult: SentimentRow | null
  searchSource: SearchResultContext
  onAddToHeatmap: () => void
}

function contextLine(source: SearchResultContext): string | null {
  if (source === 'heatmap') return 'This symbol is already on your heatmap below.'
  if (source === 'saved') return 'Using the latest saved score for this symbol.'
  if (source === 'fresh') return 'Just finished a full pass over recent coverage.'
  return null
}

export function SearchPanel({
  query,
  onQueryChange,
  onSubmit,
  searchError,
  searchLoading,
  searchResult,
  searchSource,
  onAddToHeatmap,
}: Props) {
  const resultHint =
    searchResult && !searchLoading ? contextLine(searchSource) : null

  return (
    <section className="rounded-2xl border border-slate-700/60 bg-slate-900/50 p-6 text-left shadow-xl backdrop-blur">
      <h2 className="text-lg font-semibold text-slate-100">Search tickers</h2>
      <p className="mt-1 text-sm text-slate-400">
        Enter a US equity symbol to see sentiment and sources. New lookups may take
        up to about half a minute while coverage is gathered and scored.
      </p>

      <form onSubmit={onSubmit} className="mt-4 flex flex-col gap-3 sm:flex-row">
        <label className="sr-only" htmlFor="ticker-search">
          Ticker symbol
        </label>
        <input
          id="ticker-search"
          type="text"
          inputMode="text"
          autoCapitalize="characters"
          autoComplete="off"
          placeholder="e.g. AAPL"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          className="min-h-11 flex-1 rounded-lg border border-slate-600 bg-slate-950/80 px-4 font-mono text-slate-100 placeholder:text-slate-500 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-500/30"
        />
        <button
          type="submit"
          disabled={searchLoading}
          className="min-h-11 rounded-lg bg-violet-600 px-6 font-medium text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {searchLoading ? 'Working...' : 'Search'}
        </button>
      </form>

      {searchError ? (
        <p className="mt-4 text-sm text-red-300" role="alert">
          {searchError}
        </p>
      ) : null}

      {searchLoading ? (
        <div className="mt-6 flex items-center gap-3 text-slate-300">
          <div
            className="h-8 w-8 shrink-0 animate-spin rounded-full border-2 border-violet-400 border-t-transparent"
            aria-hidden
          />
          <span>Gathering recent headlines and running sentiment...</span>
        </div>
      ) : null}

      {searchResult && !searchLoading ? (
        <div className="mt-6 space-y-4">
          {resultHint ? (
            <p className="text-sm text-slate-400">{resultHint}</p>
          ) : null}

          <div
            className="rounded-xl border border-white/10 p-5 shadow-inner"
            style={scoreToCardStyle(searchResult.score)}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="font-mono text-2xl font-bold tracking-tight">
                {searchResult.symbol}
              </span>
              <span className="font-mono text-3xl font-bold tabular-nums">
                {formatScore(searchResult.score)}
              </span>
            </div>
            {searchResult.label ? (
              <p className="mt-2 text-sm capitalize opacity-90">{searchResult.label}</p>
            ) : null}
            {typeof searchResult.article_count === 'number' ? (
              <p className="mt-1 text-sm opacity-80">
                Stories in this score: {searchResult.article_count}
              </p>
            ) : null}
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onAddToHeatmap}
              className="rounded-lg border border-violet-500/50 bg-violet-600/20 px-4 py-2 text-sm font-medium text-violet-200 transition hover:bg-violet-600/30"
            >
              Add to heatmap
            </button>
          </div>

          <div>
            <h3 className="text-sm font-medium text-slate-300">Headlines &amp; sources</h3>
            {searchResult.recent_headlines && searchResult.recent_headlines.length > 0 ? (
              <ul className="mt-2 space-y-2">
                {searchResult.recent_headlines.map((h, i) => (
                  <li key={`${h.url}-${i}`}>
                    <a
                      href={h.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-violet-300 underline decoration-violet-500/50 underline-offset-2 transition hover:text-violet-200"
                    >
                      {h.title || h.url}
                    </a>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-slate-500">No story links returned.</p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}
