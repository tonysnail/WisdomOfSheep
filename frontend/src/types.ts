export type PostListAsset = {
  ticker?: string | null
  name_or_description?: string | null
  exchange_or_market?: string | null
}

export type InterestRecord = {
  status?: string | null
  ticker?: string | null
  interest_score?: number | null
  interest_label?: string | null
  interest_why?: string | null
  council_recommended?: boolean | null
  council_priority?: string | null
  calculated_at?: string | null
  error_code?: string | null
  error_message?: string | null
  metrics?: Record<string, unknown> | null
}

export type PostListItem = {
  post_id: string
  title: string
  platform: string
  source: string
  url: string
  scraped_at?: string | null
  posted_at?: string | null
  preview: string
  markets: string[]
  signal: Record<string, any>
  has_summary: boolean
  has_analysis: boolean
  summary_bullets: string[]
  assets_mentioned: PostListAsset[]
  spam_likelihood_pct?: number | string | null
  spam_why?: string | string[] | null
  interest?: InterestRecord | null
  chairman_plain_english?: string | null
  chairman_direction?: string | null
}

export type PostDetail = {
  post: {
    post_id: string
    platform?: string | null
    source?: string | null
    url?: string | null
    title?: string | null
    author?: string | null
    scraped_at?: string | null
    posted_at?: string | null
    score?: number | null
    text?: string | null
  }
  extras: Record<string, any>
  stages: Record<string, any>
  interest?: InterestRecord | null
}

export type ListResponse = {
  items: PostListItem[]
  total: number
  page: number
  page_size: number
}

export type PostsCalendarDay = {
  date: string
  count: number
  analysed_count: number
}

export type RunStageRequest = {
  post_id: string
  stages: string[]
  overwrite: boolean
  refresh_from_csv?: boolean
  echo_post?: boolean
}

export type RunStageResponse = {
  ok: boolean
  post_id: string
  stages_run: string[]
  stdout?: string
  stderr?: string
}

export type ResearchInsight = {
  tool: string
  text: string
  status?: string
}

export type ResearchTechnical = {
  steps?: any[]
  results?: any[]
  insights?: ResearchInsight[]
  summary_lines?: string[]
  status?: string
  error?: string
}

export type ResearchTickerPayload = {
  ticker: string
  article_time: string
  hypotheses: Record<string, any>[]
  rationale: string
  plan: Record<string, any>
  technical: ResearchTechnical
  sentiment: Record<string, any>
  summary_text: string
  updated_at: string
  session_id?: string
  log?: string
}

export type ResearchPayload = {
  article_time: string
  updated_at: string
  ordered_tickers: string[]
  tickers: Record<string, ResearchTickerPayload>
}

export type ResearchResponse = {
  ok: boolean
  research: ResearchPayload
}

