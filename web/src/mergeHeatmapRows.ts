import type { SentimentRow } from './types'

/** Replace or append a row in the user's heatmap overlay list. */
export function upsertHeatmapExtra(
  prev: SentimentRow[],
  row: SentimentRow,
): SentimentRow[] {
  const sym = row.symbol.toUpperCase()
  const rest = prev.filter((r) => r.symbol.toUpperCase() !== sym)
  return [...rest, row]
}

/** Server rows with user-added overlays; extras win on duplicate symbols. */
export function mergeHeatmapRows(
  server: SentimentRow[],
  extras: SentimentRow[],
): SentimentRow[] {
  const map = new Map<string, SentimentRow>()
  for (const r of server) map.set(r.symbol.toUpperCase(), r)
  for (const r of extras) map.set(r.symbol.toUpperCase(), r)
  return [...map.values()].sort((a, b) =>
    a.symbol.localeCompare(b.symbol),
  )
}
