import { useEffect, useState } from 'react'
import { Button } from 'primereact/button'
import { Dialog } from 'primereact/dialog'
import { Message } from 'primereact/message'
import { ProgressSpinner } from 'primereact/progressspinner'
import { Tag } from 'primereact/tag'
import { fetchSentimentCacheSymbol, postSentimentBySymbol } from '../api'
import { formatScore } from '../formatScore'
import { scoreToAccent, scoreToCardStyle } from '../scoreColor'
import type { SentimentRow } from '../types'

type Props = {
  apiBase: string
  row: SentimentRow | null
  onClose: () => void
  onRowUpdate?: (row: SentimentRow) => void
}

export function TickerDetailModal({
  apiBase,
  row,
  onClose,
  onRowUpdate,
}: Props) {
  const [detail, setDetail] = useState<SentimentRow | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!row) {
      setDetail(null)
      setError(null)
      setLoading(false)
      return
    }

    setDetail(row)
    setError(null)

    const hasHeadlines =
      row.recent_headlines && row.recent_headlines.length > 0
    if (hasHeadlines || !apiBase) {
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    fetchSentimentCacheSymbol(apiBase, row.symbol)
      .then((r) => {
        if (cancelled) return
        if (r) setDetail(r)
      })
      .catch(() => {
        if (!cancelled) setError('Could not load story links for this symbol.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [row, apiBase])

  const display = detail ?? row

  async function runLatestAnalysis() {
    if (!apiBase || !display) return
    setRefreshing(true)
    setError(null)
    try {
      const next = await postSentimentBySymbol(apiBase, display.symbol)
      setDetail(next)
      onRowUpdate?.(next)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRefreshing(false)
    }
  }

  const headlines = display?.recent_headlines ?? []
  const accent = display ? scoreToAccent(display.score) : null

  const headerNode = display ? (
    <div className="flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[color:var(--color-fs-blue-soft)] text-[color:var(--color-fs-blue-deep)]">
        <i className="pi pi-chart-line" />
      </div>
      <div className="font-mono text-base font-semibold tracking-tight text-[color:var(--color-fs-text)]">
        {display.symbol}
      </div>
    </div>
  ) : null

  return (
    <Dialog
      visible={!!row}
      onHide={onClose}
      header={headerNode}
      modal
      dismissableMask
      draggable={false}
      resizable={false}
      className="w-[min(560px,calc(100vw-2rem))]"
      contentClassName="!p-0"
      pt={{
        root: { className: 'rounded-2xl overflow-hidden' },
        header: {
          className:
            'border-b border-[color:var(--color-fs-border)] !bg-[color:var(--color-fs-surface)]',
        },
        content: { className: '!bg-[color:var(--color-fs-surface)]' },
      }}
    >
      {display ? (
        <div className="space-y-6 px-6 py-6">
          <div
            className="rounded-xl border p-5 shadow-sm fs-result-reveal"
            style={scoreToCardStyle(display.score)}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-80">
                Sentiment
              </span>
              <span className="font-mono text-3xl font-semibold tabular-nums tracking-tight">
                {formatScore(display.score)}
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
              {display.label ? (
                <span className="rounded-full bg-slate-950/45 px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-slate-100">
                  {display.label}
                </span>
              ) : null}
              {typeof display.article_count === 'number' ? (
                <span className="rounded-full bg-slate-950/45 px-3 py-1 font-mono text-[11px] text-slate-100">
                  {display.article_count} stories
                </span>
              ) : null}
            </div>
          </div>

          {loading ? (
            <div className="flex items-center gap-3 text-sm text-[color:var(--color-fs-text-muted)] fs-result-reveal">
              <ProgressSpinner
                style={{ width: '20px', height: '20px' }}
                strokeWidth="4"
              />
              <span>Loading related stories…</span>
            </div>
          ) : null}

          {error ? (
            <Message severity="error" text={error} className="w-full" />
          ) : null}

          <div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
                Related coverage
              </h3>
              {apiBase ? (
                <Button
                  type="button"
                  disabled={refreshing}
                  onClick={runLatestAnalysis}
                  label={refreshing ? 'Updating…' : 'Refresh from latest news'}
                  icon={refreshing ? 'pi pi-spinner pi-spin' : 'pi pi-refresh'}
                  size="small"
                  text
                  severity="info"
                />
              ) : null}
            </div>
            {headlines.length > 0 ? (
              <ul className="mt-4 space-y-3 border-l-2 border-[color:var(--color-fs-blue-soft)] pl-4">
                {headlines.map((h, i) => (
                  <li
                    key={`${h.url}-${i}`}
                    className="fs-headline-item"
                    style={{ ['--fs-hl-stagger' as string]: `${i * 48}ms` }}
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
            ) : !loading ? (
              <p className="mt-4 font-mono text-xs text-[color:var(--color-fs-text-subtle)]">
                No story links on file for this symbol yet. Try refreshing from
                latest news.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}
    </Dialog>
  )
}
