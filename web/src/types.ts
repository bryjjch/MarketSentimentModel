export type Headline = {
  title: string
  url: string
}

export type SentimentRow = {
  symbol: string
  score: number
  label?: string
  article_count?: number
  recent_headlines?: Headline[]
  updated_at?: number
  expires_at?: number
  error?: string
  message?: string
  detail?: string
}
