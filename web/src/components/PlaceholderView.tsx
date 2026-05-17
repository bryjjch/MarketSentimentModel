import { Card } from 'primereact/card'

type Props = {
  icon: string
  title: string
  description: string
}

export function PlaceholderView({ icon, title, description }: Props) {
  return (
    <Card
      className="fs-rise border border-dashed border-[color:var(--color-fs-border-strong)] bg-white/80 shadow-sm"
      pt={{
        body: { className: 'p-10 sm:p-14' },
        content: { className: 'p-0' },
      }}
    >
      <div className="mx-auto flex max-w-md flex-col items-center text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[color:var(--color-fs-blue-soft)] to-[color:var(--color-fs-green-soft)] text-[color:var(--color-fs-blue-deep)] shadow-sm">
          <i className={['pi text-xl', icon].join(' ')} />
        </div>
        <h2 className="mt-5 text-xl font-semibold tracking-tight text-[color:var(--color-fs-text)]">
          {title}
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-[color:var(--color-fs-text-subtle)]">
          {description}
        </p>
        <span className="mt-5 inline-flex items-center gap-2 rounded-full bg-[color:var(--color-fs-green-soft)] px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-[color:var(--color-fs-green-deep)]">
          <i className="pi pi-clock text-[10px]" />
          Coming soon
        </span>
      </div>
    </Card>
  )
}
