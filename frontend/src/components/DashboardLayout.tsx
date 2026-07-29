import type { ReactNode } from 'react'
import { BullMark } from './BullMark'

type Props = {
  children: ReactNode
}

export function DashboardLayout({ children }: Props) {
  return (
    <div className="flex min-h-svh flex-col bg-[color:var(--color-fs-bg)] text-[color:var(--color-fs-text)]">
      <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-[color:var(--color-fs-border)] bg-[color:var(--color-fs-surface)]/80 px-4 backdrop-blur-md sm:px-6">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[color:var(--color-fs-blue)] to-[color:var(--color-fs-green)] text-[color:var(--color-fs-bg)] shadow-sm">
          <BullMark className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold tracking-tight">
            FinSense
          </div>
          <div className="truncate text-[11px] font-medium uppercase tracking-[0.18em] text-[color:var(--color-fs-text-subtle)]">
            Sentiment
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-x-hidden px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
        <div className="mx-auto max-w-[1440px]">{children}</div>
      </main>
    </div>
  )
}
