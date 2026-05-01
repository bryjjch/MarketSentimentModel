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
    <section className="border border-white/[0.08] bg-white/[0.02] p-8 text-left shadow-[0_0_0_1px_rgb(255_255_255_/_0.03)_inset] backdrop-blur-sm sm:p-10">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-[0.25em] text-zinc-500">
          Lookup
        </h2>
        <span className="font-mono text-[10px] text-zinc-600">
          US equities · 1–5 letters
        </span>
      </div>
      <p className="mt-4 max-w-2xl text-sm leading-relaxed text-zinc-500">
        Full runs can take up to about half a minute while coverage is gathered
        and scored.
      </p>

      <form onSubmit={onSubmit} className="mt-8 flex flex-col gap-3 sm:flex-row">
        <label className="sr-only" htmlFor="ticker-search">
          Ticker symbol
        </label>
        <input
          id="ticker-search"
          type="text"
          inputMode="text"
          autoCapitalize="characters"
          autoComplete="off"
          placeholder="AAPL"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          className="min-h-12 flex-1 border border-white/10 bg-black/50 px-4 font-mono text-sm text-white placeholder:text-zinc-600 focus:border-white/25 focus:outline-none focus:ring-1 focus:ring-white/20"
        />
        <button
          type="submit"
          disabled={searchLoading}
          className="min-h-12 shrink-0 border border-white bg-white px-8 font-medium text-black transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:border-white/20 disabled:bg-zinc-600 disabled:text-zinc-400"
        >
          {searchLoading ? 'Running…' : 'Search'}
        </button>
      </form>

      {searchError ? (
        <p
          className="mt-6 border border-white/10 bg-black/40 px-4 py-3 font-mono text-xs text-zinc-300"
          role="alert"
        >
          {searchError}
        </p>
      ) : null}

      {searchLoading ? (
        <div className="mt-8 flex items-center gap-4 text-sm text-zinc-400">
          <div
            className="h-5 w-5 shrink-0 animate-spin rounded-full border-2 border-zinc-700 border-t-white"
            aria-hidden
          />
          <span>Gathering headlines and scoring…</span>
        </div>
      ) : null}

      {searchResult && !searchLoading ? (
        <div className="mt-10 space-y-8 border-t border-white/[0.06] pt-10">
          {resultHint ? (
            <p className="font-mono text-xs text-zinc-500">{resultHint}</p>
          ) : null}

          <div
            className="border p-6 sm:p-8"
            style={scoreToCardStyle(searchResult.score)}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <span className="font-mono text-2xl font-semibold tracking-tight">
                {searchResult.symbol}
              </span>
              <span className="font-mono text-4xl font-semibold tabular-nums tracking-tight">
                {formatScore(searchResult.score)}
              </span>
            </div>
            {searchResult.label ? (
              <p className="mt-3 text-xs font-medium uppercase tracking-wider opacity-90">
                {searchResult.label}
              </p>
            ) : null}
            {typeof searchResult.article_count === 'number' ? (
              <p className="mt-2 font-mono text-xs opacity-80">
                Stories in this score: {searchResult.article_count}
              </p>
            ) : null}
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onAddToHeatmap}
              className="border border-white/20 bg-transparent px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.15em] text-white transition hover:border-white/40 hover:bg-white/[0.04]"
            >
              Add to heatmap
            </button>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-zinc-500">
              Headlines &amp; sources
            </h3>
            {searchResult.recent_headlines && searchResult.recent_headlines.length > 0 ? (
              <ul className="mt-4 space-y-3 border-l border-white/10 pl-4">
                {searchResult.recent_headlines.map((h, i) => (
                  <li key={`${h.url}-${i}`}>
                    <a
                      href={h.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-zinc-300 underline decoration-white/20 underline-offset-[5px] transition hover:text-white hover:decoration-white/50"
                    >
                      {h.title || h.url}
                    </a>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-4 font-mono text-xs text-zinc-600">
                No story links returned.
              </p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}
