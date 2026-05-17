import { useEffect, useState, type FormEvent } from 'react'
import { AutoComplete, type AutoCompleteChangeEvent, type AutoCompleteSelectEvent } from 'primereact/autocomplete'
import { Button } from 'primereact/button'
import { Card } from 'primereact/card'
import { Message } from 'primereact/message'
import { ProgressSpinner } from 'primereact/progressspinner'
import { Tag } from 'primereact/tag'
import { formatScore } from '../formatScore'
import { scoreToAccent, scoreToCardStyle } from '../scoreColor'
import type { SentimentRow } from '../types'

export type SearchResultContext = 'heatmap' | 'saved' | 'fresh' | null

type Props = {
  className?: string
  query: string
  onQueryChange: (q: string) => void
  onSubmit: (e: FormEvent) => void
  onSuggestionRequest: (q: string) => void
  onSuggestionSelect: (symbol: string) => void
  searchError: string | null
  searchLoading: boolean
  suggestionLoading: boolean
  suggestions: string[]
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
  className,
  query,
  onQueryChange,
  onSubmit,
  onSuggestionRequest,
  onSuggestionSelect,
  searchError,
  searchLoading,
  suggestionLoading,
  suggestions,
  searchResult,
  searchSource,
  onAddToHeatmap,
}: Props) {
  const [debouncedQuery, setDebouncedQuery] = useState(query)

  useEffect(() => {
    const handle = window.setTimeout(() => setDebouncedQuery(query), 180)
    return () => window.clearTimeout(handle)
  }, [query])

  useEffect(() => {
    const normalized = debouncedQuery.trim().toUpperCase()
    if (normalized.length < 2) {
      onSuggestionRequest('')
      return
    }
    onSuggestionRequest(normalized)
  }, [debouncedQuery, onSuggestionRequest])

  // AutoComplete fires on input change; suggestion fetch is debounced via debouncedQuery effect above.
  const onComplete = () => {}

  function onChange(e: AutoCompleteChangeEvent) {
    const v = typeof e.value === 'string' ? e.value : ''
    onQueryChange(v.toUpperCase())
  }

  function onSelect(e: AutoCompleteSelectEvent) {
    const v = typeof e.value === 'string' ? e.value : ''
    if (v) onSuggestionSelect(v)
  }

  const resultHint =
    searchResult && !searchLoading ? contextLine(searchSource) : null
  const accent = searchResult ? scoreToAccent(searchResult.score) : null

  return (
    <Card
      className={[
        'fs-rise border border-[color:var(--color-fs-border)] bg-white/95 shadow-sm',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      pt={{
        body: { className: 'p-6 sm:p-7' },
        content: { className: 'p-0' },
      }}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
            Lookup
          </h2>
          <p className="mt-1 text-lg font-semibold tracking-tight text-[color:var(--color-fs-text)]">
            Look up a symbol
          </p>
        </div>
        <span className="rounded-full bg-[color:var(--color-fs-blue-soft)] px-3 py-1 font-mono text-[11px] font-medium text-[color:var(--color-fs-blue-deep)]">
          US equities · 1–5 letters
        </span>
      </div>

      <form
        onSubmit={onSubmit}
        className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-stretch"
      >
        <label className="sr-only" htmlFor="ticker-search">
          Ticker symbol
        </label>
        <div className="relative flex-1">
          <i className="pi pi-search pointer-events-none absolute left-3 top-1/2 z-10 -translate-y-1/2 text-[color:var(--color-fs-text-subtle)]" />
          <AutoComplete
            inputId="ticker-search"
            value={query}
            suggestions={suggestions}
            completeMethod={onComplete}
            onChange={onChange}
            onSelect={onSelect}
            placeholder="Search ticker (e.g. AAPL)"
            inputClassName="w-full !pl-10 !py-3 !text-sm font-mono uppercase tracking-wider"
            panelClassName="font-mono text-sm"
            className="w-full [&>input]:w-full"
            delay={0}
            inputStyle={{ width: '100%' }}
            style={{ width: '100%' }}
          />
          {suggestionLoading ? (
            <i className="pi pi-spinner pi-spin absolute right-3 top-1/2 -translate-y-1/2 text-xs text-[color:var(--color-fs-text-subtle)]" />
          ) : null}
        </div>
        <Button
          type="submit"
          label={searchLoading ? 'Running…' : 'Search'}
          icon={searchLoading ? 'pi pi-spinner pi-spin' : 'pi pi-arrow-right'}
          iconPos="right"
          disabled={searchLoading}
          className="!px-6"
        />
      </form>

      {searchError ? (
        <div className="mt-5 fs-result-reveal">
          <Message severity="error" text={searchError} className="w-full" />
        </div>
      ) : null}

      {searchLoading ? (
        <div className="mt-7 flex items-center gap-3 text-sm text-[color:var(--color-fs-text-muted)] fs-result-reveal">
          <ProgressSpinner
            style={{ width: '24px', height: '24px' }}
            strokeWidth="4"
          />
          <span>Gathering headlines and scoring…</span>
          <span className="ml-1 flex gap-1" aria-hidden>
            <span className="inline-block h-1 w-1 rounded-full bg-[color:var(--color-fs-blue)] fs-loader-dot" />
            <span className="inline-block h-1 w-1 rounded-full bg-[color:var(--color-fs-blue)] fs-loader-dot" />
            <span className="inline-block h-1 w-1 rounded-full bg-[color:var(--color-fs-blue)] fs-loader-dot" />
          </span>
        </div>
      ) : null}

      {searchResult && !searchLoading ? (
        <div
          key={searchResult.symbol}
          className="mt-7 space-y-6 border-t border-[color:var(--color-fs-border)] pt-6 fs-result-reveal"
        >
          {resultHint ? (
            <p className="font-mono text-xs text-[color:var(--color-fs-text-subtle)]">
              {resultHint}
            </p>
          ) : null}

          <div
            className="rounded-xl border p-6 shadow-sm sm:p-7"
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
            <div className="mt-4 flex flex-wrap items-center gap-2">
              {accent ? (
                <Tag
                  value={accent.label}
                  style={{
                    backgroundColor: accent.bg,
                    color: accent.fg,
                    border: `1px solid ${accent.border}`,
                  }}
                  className="!rounded-full !px-3 !py-1 !text-[11px] !font-semibold uppercase tracking-wider"
                />
              ) : null}
              {searchResult.label ? (
                <span className="rounded-full bg-white/60 px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-slate-700">
                  {searchResult.label}
                </span>
              ) : null}
              {typeof searchResult.article_count === 'number' ? (
                <span className="rounded-full bg-white/60 px-3 py-1 font-mono text-[11px] text-slate-700">
                  {searchResult.article_count} stories
                </span>
              ) : null}
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button
              type="button"
              onClick={onAddToHeatmap}
              label="Add to heatmap"
              icon="pi pi-plus"
              severity="success"
              outlined
              className="!text-sm"
            />
          </div>

          <div>
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
              Headlines &amp; sources
            </h3>
            {searchResult.recent_headlines && searchResult.recent_headlines.length > 0 ? (
              <ul className="mt-4 space-y-3 border-l-2 border-[color:var(--color-fs-blue-soft)] pl-4">
                {searchResult.recent_headlines.map((h, i) => (
                  <li
                    key={`${h.url}-${i}`}
                    className="fs-headline-item"
                    style={{ ['--fs-hl-stagger' as string]: `${i * 50}ms` }}
                  >
                    <a
                      href={h.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-[color:var(--color-fs-text)] underline decoration-[color:var(--color-fs-blue-soft)] decoration-2 underline-offset-[5px] transition-colors hover:text-[color:var(--color-fs-blue-deep)] hover:decoration-[color:var(--color-fs-blue)]"
                    >
                      {h.title || h.url}
                    </a>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-4 font-mono text-xs text-[color:var(--color-fs-text-subtle)]">
                No story links returned.
              </p>
            )}
          </div>
        </div>
      ) : null}
    </Card>
  )
}
