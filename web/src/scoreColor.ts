import type { CSSProperties } from 'react'

type Hsl = { h: number; s: number; l: number }

/** Neutral surface used at score = 0 so we don't pass through purple/blue. */
const NEUTRAL_HI: Hsl = { h: 220, s: 30, l: 97 }
const NEUTRAL_LO: Hsl = { h: 220, s: 25, l: 93 }

/** Bearish surface — soft red, light-mode-friendly. */
const RED_HI: Hsl = { h: 4, s: 84, l: 94 }
const RED_LO: Hsl = { h: 4, s: 70, l: 84 }

/** Bullish surface — soft green to match the FinSense accent. */
const GREEN_HI: Hsl = { h: 146, s: 65, l: 92 }
const GREEN_LO: Hsl = { h: 148, s: 55, l: 80 }

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function hsl({ h, s, l }: Hsl): string {
  return `hsl(${h.toFixed(1)} ${s.toFixed(1)}% ${l.toFixed(1)}%)`
}

function blend(a: Hsl, b: Hsl, t: number): Hsl {
  return {
    h: lerp(a.h, b.h, t),
    s: lerp(a.s, b.s, t),
    l: lerp(a.l, b.l, t),
  }
}

/**
 * Map sentiment score (roughly -1 … +1) to a card surface tinted between
 * soft red (bearish) and soft green (bullish), passing through a neutral
 * cool gray at 0 so the gradient never reads as a third hue.
 */
export function scoreToCardStyle(score: number): CSSProperties {
  const t = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))

  let hi: Hsl
  let lo: Hsl
  if (t <= 0) {
    const k = t + 1
    hi = blend(RED_HI, NEUTRAL_HI, k)
    lo = blend(RED_LO, NEUTRAL_LO, k)
  } else {
    hi = blend(NEUTRAL_HI, GREEN_HI, t)
    lo = blend(NEUTRAL_LO, GREEN_LO, t)
  }

  return {
    background: `linear-gradient(155deg, ${hsl(hi)} 0%, ${hsl(lo)} 100%)`,
    color: '#0f172a',
    borderColor: 'rgb(15 23 42 / 8%)',
  }
}

/** Semantic color (text/icon) for the sentiment score on a light surface. */
export function scoreToAccent(score: number): {
  fg: string
  bg: string
  border: string
  label: 'Bullish' | 'Bearish' | 'Neutral'
} {
  const t = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))
  if (t > 0.05) {
    return {
      fg: '#15803d',
      bg: '#dcfce7',
      border: '#86efac',
      label: 'Bullish',
    }
  }
  if (t < -0.05) {
    return {
      fg: '#b91c1c',
      bg: '#fee2e2',
      border: '#fca5a5',
      label: 'Bearish',
    }
  }
  return {
    fg: '#1d4ed8',
    bg: '#dbeafe',
    border: '#93c5fd',
    label: 'Neutral',
  }
}
