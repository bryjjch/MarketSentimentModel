import type { CSSProperties } from 'react'

/** Map sentiment score (roughly -1 … +1) to a background gradient color. */
export function scoreToCardStyle(score: number): CSSProperties {
  const t = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))
  // Hue: red (0°) → amber → green (120°)
  const hue = ((t + 1) / 2) * 120
  const lightness = 42 - Math.abs(t) * 8
  const saturation = 62 + Math.abs(t) * 12
  return {
    background: `linear-gradient(145deg, hsl(${hue} ${saturation}% ${lightness}%) 0%, hsl(${hue} ${saturation - 10}% ${lightness - 12}%) 100%)`,
    color: Math.abs(t) < 0.15 ? '#0f172a' : '#f8fafc',
    textShadow: Math.abs(t) < 0.15 ? 'none' : '0 1px 2px rgb(0 0 0 / 35%)',
  }
}
