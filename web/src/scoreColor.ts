import type { CSSProperties } from 'react'

type Hsl = { h: number; s: number; l: number }

const BEARISH_HI: Hsl = { h: 357, s: 46, l: 27 }
const BEARISH_LO: Hsl = { h: 352, s: 40, l: 17 }

const NEUTRAL_HI: Hsl = { h: 220, s: 11, l: 20 }
const NEUTRAL_LO: Hsl = { h: 228, s: 8, l: 13 }

const BULLISH_HI: Hsl = { h: 154, s: 44, l: 30 }
const BULLISH_LO: Hsl = { h: 158, s: 38, l: 20 }

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Shortest path on the hue wheel (degrees). */
function lerpHue(a: number, b: number, t: number): number {
  let d = b - a
  if (d > 180) d -= 360
  if (d < -180) d += 360
  const h = a + d * t
  return ((h % 360) + 360) % 360
}

function lerpHsl(from: Hsl, to: Hsl, t: number): Hsl {
  return {
    h: lerpHue(from.h, to.h, t),
    s: lerp(from.s, to.s, t),
    l: lerp(from.l, to.l, t),
  }
}

function hsl({ h, s, l }: Hsl): string {
  return `hsl(${h.toFixed(1)} ${s.toFixed(1)}% ${l.toFixed(1)}%)`
}

/**
 * Map sentiment score (roughly -1 … +1) to a card surface: dusty rose for
 * bearish, cool slate at neutral, muted emerald for bullish — tuned for a
 * dark UI without neon saturation.
 */
export function scoreToCardStyle(score: number): CSSProperties {
  const t = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))

  let hi: Hsl
  let lo: Hsl
  if (t <= 0) {
    const k = t + 1
    hi = lerpHsl(BEARISH_HI, NEUTRAL_HI, k)
    lo = lerpHsl(BEARISH_LO, NEUTRAL_LO, k)
  } else {
    hi = lerpHsl(NEUTRAL_HI, BULLISH_HI, t)
    lo = lerpHsl(NEUTRAL_LO, BULLISH_LO, t)
  }

  const avgL = (hi.l + lo.l) / 2
  const useDarkText = avgL > 44

  return {
    background: `linear-gradient(155deg, ${hsl(hi)} 0%, ${hsl(lo)} 100%)`,
    color: useDarkText ? '#0c0c0d' : '#f4f4f5',
    borderColor: useDarkText
      ? 'rgb(0 0 0 / 12%)'
      : 'rgb(255 255 255 / 14%)',
    textShadow: useDarkText ? 'none' : '0 1px 2px rgb(0 0 0 / 45%)',
  }
}
