/** Format a sentiment score with a leading sign and fixed decimal places. */
export function formatScore(score: number, decimals = 4): string {
  if (!Number.isFinite(score)) return '-'
  const s = score.toFixed(decimals)
  return score >= 0 ? `+${s}` : s
}
