import type { CSSProperties } from 'react'

type Hsl = { h: number; s: number; l: number }

/** Achromatic neutral — avoids hue interpolation through purple/blue. */
const NEUTRAL_L_HI = 21
const NEUTRAL_L_LO = 14

/** Strong red family (both stops); gradient depth from lightness only. */
const RED_HI: Hsl = { h: 3, s: 56, l: 30 }
const RED_LO: Hsl = { h: 3, s: 48, l: 18 }

/** Strong green family — hue kept in true green, not teal/cyan. */
const GREEN_HI: Hsl = { h: 146, s: 52, l: 34 }
const GREEN_LO: Hsl = { h: 148, s: 46, l: 21 }

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function hsl({ h, s, l }: Hsl): string {
  return `hsl(${h.toFixed(1)} ${s.toFixed(1)}% ${l.toFixed(1)}%)`
}

/**
 * Map sentiment score (roughly -1 … +1) to a card surface. Bearish fades from
 * clear red through gray; bullish fades from gray into green — never blending
 * hue “across” the wheel (which reads as purple / slate-blue).
 */
export function scoreToCardStyle(score: number): CSSProperties {
  const t = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))

  let hi: Hsl
  let lo: Hsl
  if (t <= 0) {
    const k = t + 1
    hi = {
      h: RED_HI.h,
      s: lerp(RED_HI.s, 0, k),
      l: lerp(RED_HI.l, NEUTRAL_L_HI, k),
    }
    lo = {
      h: RED_LO.h,
      s: lerp(RED_LO.s, 0, k),
      l: lerp(RED_LO.l, NEUTRAL_L_LO, k),
    }
  } else {
    hi = {
      h: GREEN_HI.h,
      s: lerp(0, GREEN_HI.s, t),
      l: lerp(NEUTRAL_L_HI, GREEN_HI.l, t),
    }
    lo = {
      h: GREEN_LO.h,
      s: lerp(0, GREEN_LO.s, t),
      l: lerp(NEUTRAL_L_LO, GREEN_LO.l, t),
    }
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
