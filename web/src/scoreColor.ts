import type { CSSProperties } from 'react'

/**
 * Map sentiment score (roughly -1 … +1) to a monochrome card surface.
 * Bearish: darker gray; bullish: lighter gray / near-white.
 */
export function scoreToCardStyle(score: number): CSSProperties {
  const t = Math.max(-1, Math.min(1, Number.isFinite(score) ? score : 0))
  const u = (t + 1) / 2
  const hi = 8 + u * 82
  const lo = Math.max(5, hi - 16)
  const useDarkText = hi > 48
  return {
    background: `linear-gradient(155deg, hsl(0 0% ${hi}%) 0%, hsl(0 0% ${lo}%) 100%)`,
    color: useDarkText ? '#0a0a0a' : '#f4f4f5',
    borderColor: useDarkText ? 'rgb(0 0 0 / 10%)' : 'rgb(255 255 255 / 12%)',
    textShadow: useDarkText ? 'none' : '0 1px 2px rgb(0 0 0 / 45%)',
  }
}
