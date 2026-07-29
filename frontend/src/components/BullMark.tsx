type Props = {
  className?: string
}

/**
 * FinSense logo mark: a bull's head, for "bullish".
 *
 * One path so the horns union with the head under the default nonzero fill
 * rule — every solid subpath is wound clockwise and the eyes/nostrils
 * counter-clockwise so they knock through to the background.
 */
const BULL_PATH = [
  // Left horn.
  'M5.87 10.23 C3.6 9.6 2.2 7.2 2.6 3.4 C3.2 6 4.6 8 8.13 8.57 Z',
  // Right horn.
  'M15.87 8.57 C19.4 8 20.8 6 21.4 3.4 C21.8 7.2 20.4 9.6 18.13 10.23 Z',
  // Head.
  'M8.5 8 h7 a3 3 0 0 1 3 3 v1 a6.5 6.5 0 0 1 -13 0 v-1 a3 3 0 0 1 3 -3 Z',
  // Eyes.
  'M9.4 10.75 a1.15 1.15 0 1 0 0 2.3 a1.15 1.15 0 1 0 0 -2.3 Z',
  'M14.6 10.75 a1.15 1.15 0 1 0 0 2.3 a1.15 1.15 0 1 0 0 -2.3 Z',
  // Nostrils.
  'M10.6 15.52 a.78 .78 0 1 0 0 1.56 a.78 .78 0 1 0 0 -1.56 Z',
  'M13.4 15.52 a.78 .78 0 1 0 0 1.56 a.78 .78 0 1 0 0 -1.56 Z',
].join(' ')

export function BullMark({ className }: Props) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="currentColor"
      aria-hidden="true"
      focusable="false"
    >
      <path d={BULL_PATH} />
    </svg>
  )
}
