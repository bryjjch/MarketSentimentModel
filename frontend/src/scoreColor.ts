import type { CSSProperties } from 'react'

type Rgb = { r: number; g: number; b: number }

/**
 * Surfaces are blended in sRGB rather than HSL on purpose: a hue lerp from red
 * (~4°) to a cool gray (~220°) sweeps through yellow and green on the way, so
 * mid-strength bearish tiles came out looking bullish.
 */

/** Neutral surface at score 0 — cool slate, so "no signal" reads as no colour. */
const NEUTRAL_HI: Rgb = { r: 33, g: 43, b: 61 }
const NEUTRAL_LO: Rgb = { r: 22, g: 30, b: 44 }

/** Bearish surface at score −1. */
const RED_HI: Rgb = { r: 155, g: 44, b: 51 }
const RED_LO: Rgb = { r: 102, g: 28, b: 34 }

/** Bullish surface at score +1. */
const GREEN_HI: Rgb = { r: 26, g: 122, b: 84 }
const GREEN_LO: Rgb = { r: 13, g: 79, b: 54 }

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
    color: '#eef2f9',
    borderColor: 'rgb(255 255 255 / 12%)',
  }
}

/** Semantic color (text/icon) for the sentiment score on a dark surface. */
export function scoreToAccent(score: number): {
  fg: string
  bg: string
  border: string
  label: 'Bullish' | 'Bearish' | 'Neutral'
} {
  const t = clampScore(score)
  if (t > 0.05) {
    return {
      fg: '#6ee7b7',
      bg: '#0f2f22',
      border: '#1f6b4f',
      label: 'Bullish',
    }
  }
  if (t < -0.05) {
    return {
      fg: '#fca5a5',
      bg: '#3a1519',
      border: '#8b3a3f',
      label: 'Bearish',
    }
  }
  return {
    fg: '#93c5fd',
    bg: '#152540',
    border: '#2f4d80',
    label: 'Neutral',
  }
}
