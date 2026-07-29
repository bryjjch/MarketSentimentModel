import type { CSSProperties } from 'react'

type Rgb = { r: number; g: number; b: number }

/**
 * Surfaces are blended in sRGB rather than HSL on purpose: a hue lerp from red
 * (~4°) to a cool gray (~220°) sweeps through yellow and green on the way, so
 * mid-strength bearish tiles came out looking bullish.
 */

/** Neutral surface at score 0 — cool gray, so "no signal" reads as no colour. */
const NEUTRAL_HI: Rgb = { r: 246, g: 248, b: 251 }
const NEUTRAL_LO: Rgb = { r: 231, g: 236, b: 243 }

/** Bearish surface at score −1. */
const RED_HI: Rgb = { r: 254, g: 160, b: 156 }
const RED_LO: Rgb = { r: 238, g: 88, b: 92 }

/** Bullish surface at score +1. */
const GREEN_HI: Rgb = { r: 140, g: 232, b: 178 }
const GREEN_LO: Rgb = { r: 74, g: 203, b: 138 }

/**
 * Sub-linear ramp from neutral to the full-strength surface. Real scores rarely
 * sit near ±1, so a straight lerp leaves almost every tile washed out; this
 * gives a score of ±0.3 roughly half the available tint.
 */
const INTENSITY_GAMMA = 0.6

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function mix(a: Rgb, b: Rgb, t: number): Rgb {
  return {
    r: lerp(a.r, b.r, t),
    g: lerp(a.g, b.g, t),
    b: lerp(a.b, b.b, t),
  }
}

function css({ r, g, b }: Rgb): string {
  return `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)})`
}

function clampScore(score: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))
}

/**
 * Map sentiment score (roughly -1 … +1) to a card surface tinted from a neutral
 * gray at 0 toward red (bearish) or green (bullish).
 */
export function scoreToCardStyle(score: number): CSSProperties {
  const t = clampScore(score)
  const strength = Math.pow(Math.abs(t), INTENSITY_GAMMA)
  const [hi, lo] = t < 0 ? [RED_HI, RED_LO] : [GREEN_HI, GREEN_LO]

  return {
    background: `linear-gradient(155deg, ${css(mix(NEUTRAL_HI, hi, strength))} 0%, ${css(mix(NEUTRAL_LO, lo, strength))} 100%)`,
    color: '#0f172a',
    borderColor: 'rgb(15 23 42 / 12%)',
  }
}

/** Semantic color (text/icon) for the sentiment score on a light surface. */
export function scoreToAccent(score: number): {
  fg: string
  bg: string
  border: string
  label: 'Bullish' | 'Bearish' | 'Neutral'
} {
  const t = clampScore(score)
  if (t > 0.05) {
    return {
      fg: '#166534',
      bg: '#dcfce7',
      border: '#4ade80',
      label: 'Bullish',
    }
  }
  if (t < -0.05) {
    return {
      fg: '#991b1b',
      bg: '#fee2e2',
      border: '#f87171',
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
