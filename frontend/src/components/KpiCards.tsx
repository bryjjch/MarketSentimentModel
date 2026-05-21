import { useMemo } from 'react'
import { Card } from 'primereact/card'
import { Skeleton } from 'primereact/skeleton'
import { formatScore } from '../formatScore'
import type { SentimentRow } from '../types'

type Props = {
  rows: SentimentRow[]
  loading?: boolean
}

type Kpi = {
  key: string
  label: string
  value: string
  hint: string
  icon: string
  accent: 'blue' | 'green' | 'red' | 'slate'
}

const ACCENT_CLASSES: Record<Kpi['accent'], { ring: string; icon: string; chip: string }> = {
  blue: {
    ring: 'from-[color:var(--color-fs-blue-soft)] to-white',
    icon: 'bg-[color:var(--color-fs-blue-soft)] text-[color:var(--color-fs-blue-deep)]',
    chip: 'text-[color:var(--color-fs-blue-deep)]',
  },
  green: {
    ring: 'from-[color:var(--color-fs-green-soft)] to-white',
    icon: 'bg-[color:var(--color-fs-green-soft)] text-[color:var(--color-fs-green-deep)]',
    chip: 'text-[color:var(--color-fs-green-deep)]',
  },
  red: {
    ring: 'from-[color:var(--color-fs-red-soft)] to-white',
    icon: 'bg-[color:var(--color-fs-red-soft)] text-red-700',
    chip: 'text-red-700',
  },
  slate: {
    ring: 'from-slate-100 to-white',
    icon: 'bg-slate-100 text-slate-700',
    chip: 'text-slate-700',
  },
}

export function KpiCards({ rows, loading }: Props) {
  const kpis = useMemo<Kpi[]>(() => {
    const total = rows.length
    if (total === 0) {
      return [
        {
          key: 'tracking',
          label: 'Symbols tracked',
          value: '0',
          hint: 'Search to add tickers',
          icon: 'pi-database',
          accent: 'blue',
        },
        {
          key: 'avg',
          label: 'Average sentiment',
          value: '—',
          hint: 'Awaiting data',
          icon: 'pi-chart-bar',
          accent: 'slate',
        },
        {
          key: 'bullish',
          label: 'Bullish',
          value: '0',
          hint: 'No symbols yet',
          icon: 'pi-arrow-up-right',
          accent: 'green',
        },
        {
          key: 'bearish',
          label: 'Bearish',
          value: '0',
          hint: 'No symbols yet',
          icon: 'pi-arrow-down-right',
          accent: 'red',
        },
      ]
    }

    const sum = rows.reduce((acc, r) => acc + (Number.isFinite(r.score) ? r.score : 0), 0)
    const avg = sum / total
    const bullish = rows.filter((r) => r.score > 0.05)
    const bearish = rows.filter((r) => r.score < -0.05)
    const topBull = [...bullish].sort((a, b) => b.score - a.score)[0]
    const topBear = [...bearish].sort((a, b) => a.score - b.score)[0]

    return [
      {
        key: 'tracking',
        label: 'Symbols tracked',
        value: String(total),
        hint: 'On your heatmap',
        icon: 'pi-database',
        accent: 'blue',
      },
      {
        key: 'avg',
        label: 'Average sentiment',
        value: formatScore(avg, 3),
        hint: avg >= 0 ? 'Tilted bullish overall' : 'Tilted bearish overall',
        icon: 'pi-chart-bar',
        accent: avg >= 0 ? 'green' : 'red',
      },
      {
        key: 'bullish',
        label: 'Bullish',
        value: String(bullish.length),
        hint: topBull ? `Top: ${topBull.symbol} ${formatScore(topBull.score, 2)}` : 'None today',
        icon: 'pi-arrow-up-right',
        accent: 'green',
      },
      {
        key: 'bearish',
        label: 'Bearish',
        value: String(bearish.length),
        hint: topBear ? `Lowest: ${topBear.symbol} ${formatScore(topBear.score, 2)}` : 'None today',
        icon: 'pi-arrow-down-right',
        accent: 'red',
      },
    ]
  }, [rows])

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {kpis.map((kpi, i) => {
        const accent = ACCENT_CLASSES[kpi.accent]
        return (
          <div
            key={kpi.key}
            className="fs-tile-in"
            style={{ ['--fs-stagger' as string]: `${i * 60}ms` }}
          >
            <Card
              className={[
                'h-full border border-[color:var(--color-fs-border)] bg-gradient-to-br',
                accent.ring,
                'shadow-sm',
              ].join(' ')}
              pt={{
                body: { className: 'p-5' },
                content: { className: 'p-0' },
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--color-fs-text-subtle)]">
                    {kpi.label}
                  </p>
                  <p className="mt-3 font-mono text-3xl font-semibold tabular-nums tracking-tight text-[color:var(--color-fs-text)]">
                    {loading ? (
                      <Skeleton width="6rem" height="2rem" />
                    ) : (
                      kpi.value
                    )}
                  </p>
                  <p className={['mt-2 truncate text-xs font-medium', accent.chip].join(' ')}>
                    {kpi.hint}
                  </p>
                </div>
                <div
                  className={[
                    'flex h-10 w-10 items-center justify-center rounded-xl',
                    accent.icon,
                  ].join(' ')}
                >
                  <i className={['pi text-base', kpi.icon].join(' ')} />
                </div>
              </div>
            </Card>
          </div>
        )
      })}
    </div>
  )
}
