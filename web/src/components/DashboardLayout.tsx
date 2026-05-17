import { useState, type ReactNode } from 'react'

type NavItem = {
  key: string
  label: string
  icon: string
  badge?: string
}

const NAV_ITEMS: NavItem[] = [
  { key: 'overview', label: 'Overview', icon: 'pi-th-large' },
  { key: 'heatmap', label: 'Heatmap', icon: 'pi-table' },
  { key: 'search', label: 'Lookup', icon: 'pi-search' },
  { key: 'watchlist', label: 'Watchlist', icon: 'pi-bookmark' },
  { key: 'news', label: 'News feed', icon: 'pi-megaphone', badge: 'soon' },
  { key: 'settings', label: 'Settings', icon: 'pi-cog' },
]

type Props = {
  active?: string
  onNavigate?: (key: string) => void
  headerRight?: ReactNode
  title?: string
  subtitle?: string
  children: ReactNode
}

export function DashboardLayout({
  active = 'overview',
  onNavigate,
  headerRight,
  title = 'Market Sentiment',
  subtitle = 'Dashboard',
  children,
}: Props) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="flex min-h-svh bg-[color:var(--color-fs-bg)] text-[color:var(--color-fs-text)]">
      <aside
        className={[
          'hidden shrink-0 flex-col border-r border-[color:var(--color-fs-border)] bg-white/85 backdrop-blur-md transition-[width] duration-200 ease-out md:flex',
          collapsed ? 'w-[72px]' : 'w-[244px]',
        ].join(' ')}
        aria-label="Primary"
      >
        <div className="flex h-16 items-center gap-3 border-b border-[color:var(--color-fs-border)] px-4">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[color:var(--color-fs-blue)] to-[color:var(--color-fs-green)] text-white shadow-sm">
            <i className="pi pi-chart-line text-base" />
          </div>
          {!collapsed ? (
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold tracking-tight">
                FinSense
              </div>
              <div className="truncate text-[11px] font-medium uppercase tracking-[0.18em] text-[color:var(--color-fs-text-subtle)]">
                Sentiment
              </div>
            </div>
          ) : null}
        </div>

        <nav className="flex-1 overflow-y-auto px-2 py-4">
          <ul className="space-y-1">
            {NAV_ITEMS.map((item) => {
              const isActive = item.key === active
              return (
                <li key={item.key}>
                  <button
                    type="button"
                    onClick={() => onNavigate?.(item.key)}
                    title={collapsed ? item.label : undefined}
                    className={[
                      'group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-[color:var(--color-fs-blue-soft)] text-[color:var(--color-fs-blue-deep)]'
                        : 'text-[color:var(--color-fs-text-muted)] hover:bg-[color:var(--color-fs-surface-muted)] hover:text-[color:var(--color-fs-text)]',
                    ].join(' ')}
                  >
                    <i
                      className={[
                        'pi text-base shrink-0',
                        item.icon,
                        isActive
                          ? 'text-[color:var(--color-fs-blue)]'
                          : 'text-[color:var(--color-fs-text-subtle)] group-hover:text-[color:var(--color-fs-text)]',
                      ].join(' ')}
                    />
                    {!collapsed ? (
                      <span className="flex-1 truncate">{item.label}</span>
                    ) : null}
                    {!collapsed && item.badge ? (
                      <span className="rounded-full bg-[color:var(--color-fs-green-soft)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[color:var(--color-fs-green-deep)]">
                        {item.badge}
                      </span>
                    ) : null}
                  </button>
                </li>
              )
            })}
          </ul>
        </nav>

        <div className="border-t border-[color:var(--color-fs-border)] px-2 py-3">
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-[color:var(--color-fs-text-muted)] transition-colors hover:bg-[color:var(--color-fs-surface-muted)] hover:text-[color:var(--color-fs-text)]"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <i
              className={[
                'pi shrink-0 text-base',
                collapsed ? 'pi-angle-double-right' : 'pi-angle-double-left',
              ].join(' ')}
            />
            {!collapsed ? <span>Collapse</span> : null}
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-[color:var(--color-fs-border)] bg-white/90 px-4 backdrop-blur-md sm:px-6">
          <div className="flex items-center gap-2 md:hidden">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-[color:var(--color-fs-blue)] to-[color:var(--color-fs-green)] text-white">
              <i className="pi pi-chart-line text-base" />
            </div>
            <span className="text-sm font-semibold tracking-tight">FinSense</span>
          </div>

          <div className="hidden flex-col md:flex">
            <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[color:var(--color-fs-text-subtle)]">
              {subtitle}
            </span>
            <span className="text-sm font-semibold tracking-tight text-[color:var(--color-fs-text)]">
              {title}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-3">
            {headerRight}
            <div className="hidden h-9 w-9 items-center justify-center rounded-full bg-[color:var(--color-fs-blue-soft)] text-sm font-semibold text-[color:var(--color-fs-blue-deep)] sm:flex">
              FS
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-x-hidden px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
          <div className="mx-auto max-w-[1440px]">{children}</div>
        </main>
      </div>
    </div>
  )
}
