import type { FormEvent } from 'react'
import { formatScore } from '../formatScore'
import type { SentimentRow } from '../types'
import { scoreToCardStyle } from '../scoreColor'

type Props = {
  query: string
  onQueryChange: (q: string) => void
  onSubmit: (e: FormEvent) => void
  searchError: string | null
  searchLoading: boolean
  searchResult: SentimentRow | null
  searchSource: 'heatmap' | 'cached' | 'live' | null
}

export function SearchPanel({
  query,
  onQueryChange,
  onSubmit,
  searchError,
  searchLoading,
  searchResult,
  searchSource,
}: Props) {
  return (
    <section className="rounded-2xl border border-slate-700/60 bg-slate-900/50 p-6 text-left shadow-xl backdrop-blur">
      <h2 className="text-lg font-semibold text-slate-100">On-demand search</h2>
      <p className="mt-1 text-sm text-slate-400">
        Look up sentiment for any US ticker. Symbols already on the heatmap use
        the cached snapshot; others call the API (may take up to ~30s).
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
          {searchLoading ? 'Analyzing…' : 'Search'}
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
          <span>Running sentiment pipeline (news + model)…</span>
        </div>
      ) : null}

      {searchResult && !searchLoading ? (
        <div className="mt-6 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            {searchSource === 'heatmap' ? (
              <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-300">
                From heatmap (cached)
              </span>
            ) : searchSource === 'cached' ? (
              <span className="rounded-full bg-sky-500/20 px-3 py-1 text-xs font-medium text-sky-200">
                Cached snapshot
              </span>
            ) : searchSource === 'live' ? (
              <span className="rounded-full bg-violet-500/20 px-3 py-1 text-xs font-medium text-violet-200">
                Live analysis
              </span>
            ) : null}
          </div>

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
                Articles analyzed: {searchResult.article_count}
              </p>
            ) : null}
          </div>

          <div>
            <h3 className="text-sm font-medium text-slate-300">Recent headlines</h3>
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
              <p className="mt-2 text-sm text-slate-500">No headlines returned.</p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}
