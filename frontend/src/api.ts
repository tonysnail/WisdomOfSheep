import type {
  ListResponse,
  PostDetail,
  PostsCalendarDay,
  RunStageRequest,
  RunStageResponse,
  ResearchResponse,
} from './types'

const API = import.meta.env.VITE_API ?? 'http://127.0.0.1:8000'

const encodePostIdForPath = (postId: string) => encodeURIComponent(postId)

const buildPostUrl = (postId: string, suffix = '') =>
  `${API}/api/posts/${encodePostIdForPath(postId)}${suffix}`

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(`HTTP ${res.status}: ${txt}`)
  }
  return res.json() as Promise<T>
}

export type ListPostsParams = {
  q?: string
  platform?: string
  source?: string
  page?: number
  page_size?: number
  date_from?: string
  date_to?: string
  interest_min?: number
}

export async function listPosts(params: ListPostsParams): Promise<ListResponse> {
  const url = new URL(`${API}/api/posts`)
  if (params.q) url.searchParams.set('q', params.q)
  if (params.platform) url.searchParams.set('platform', params.platform)
  if (params.source) url.searchParams.set('source', params.source)
  url.searchParams.set('page', String(params.page ?? 1))
  url.searchParams.set('page_size', String(params.page_size ?? 20))
  if (params.date_from) url.searchParams.set('date_from', params.date_from)
  if (params.date_to) url.searchParams.set('date_to', params.date_to)
  if (typeof params.interest_min === 'number') {
    url.searchParams.set('interest_min', String(params.interest_min))
  }

  const res = await fetch(url.toString())
  return json<ListResponse>(res)
}

export type PostsCalendarResponse = {
  days?: PostsCalendarDay[]
}

export async function getPostsCalendar(year: number, month: number): Promise<PostsCalendarDay[]> {
  const url = new URL(`${API}/api/posts/calendar`)
  url.searchParams.set('year', String(year))
  url.searchParams.set('month', String(month))

  const res = await fetch(url.toString())
  const data = await json<PostsCalendarResponse>(res)
  if (!data.days) return []
  return data.days
    .map((day) => {
      const rawCount = typeof day.count === 'number' ? day.count : Number.parseInt(String(day.count ?? 0), 10)
      const rawAnalysed =
        typeof day.analysed_count === 'number'
          ? day.analysed_count
          : Number.parseInt(String(day.analysed_count ?? 0), 10)
      return {
        date: day.date,
        count: Number.isFinite(rawCount) && rawCount > 0 ? rawCount : 0,
        analysed_count: Number.isFinite(rawAnalysed) && rawAnalysed > 0 ? rawAnalysed : 0,
      }
    })
    .filter((day): day is PostsCalendarDay => Boolean(day.date) && day.count >= 0)
}

export async function getPost(postId: string): Promise<PostDetail> {
  const res = await fetch(buildPostUrl(postId))
  return json<PostDetail>(res)
}

export type StockWindowRow = {
  timestamp: string
  Open?: number | null
  High?: number | null
  Low?: number | null
  Close?: number | null
  Volume?: number | null
  seconds_from_center?: number | null
  return_from_t0?: number | null
}

export type StockWindowResponse = {
  ticker: string
  interval: string
  start_utc: string
  end_utc: string
  center_utc: string
  t0_timestamp_utc: string | null
  t0_close: number | null
  rows: number
  note?: string | null
  data?: StockWindowRow[] | null
}

export type GetStockWindowParams = {
  ticker: string
  center?: string | null
  before?: string
  after?: string
  interval?: string
}

export async function getStockWindow(params: GetStockWindowParams): Promise<StockWindowResponse> {
  const url = new URL(`${API}/api/stocks/window`)
  url.searchParams.set('ticker', params.ticker)
  if (params.center) url.searchParams.set('center', params.center)
  if (params.before) url.searchParams.set('before', params.before)
  if (params.after) url.searchParams.set('after', params.after)
  if (params.interval) url.searchParams.set('interval', params.interval)

  const res = await fetch(url.toString())
  return json<StockWindowResponse>(res)
}

export async function runStage(body: RunStageRequest): Promise<RunStageResponse> {
  const res = await fetch(`${API}/api/run-stage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return json<RunStageResponse>(res)
}

export async function runResearch(postId: string): Promise<ResearchResponse> {
  const res = await fetch(buildPostUrl(postId, '/research'), {
    method: 'POST',
  })
  return json<ResearchResponse>(res)
}

export async function clearAnalysis(postId: string): Promise<{
  ok: boolean
  deleted_stages: number
  removed_research: boolean
}> {
  const res = await fetch(buildPostUrl(postId, '/clear-analysis'), {
    method: 'POST',
  })
  return json(res)
}

export async function eraseAllCouncilAnalysis(): Promise<{
  ok: boolean
  deleted_stages: number
  cleared_research_posts: number
}> {
  const res = await fetch(`${API}/api/council-analysis/erase-all`, {
    method: 'POST',
  })
  return json(res)
}

export async function repairDatabase(options?: RepairDatabaseOptions): Promise<RepairDatabaseResponse> {
  const payload: Record<string, unknown> = {}
  if (typeof options?.allowReset === 'boolean') {
    payload.allow_reset = options.allowReset
  }
  if (options?.restoreBackup) {
    payload.restore_backup = options.restoreBackup
  }
  const res = await fetch(`${API}/api/database/repair`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return json(res)
}

export type RefreshSummariesMode = 'full' | 'new_only'

export async function startRefreshSummaries(options?: {
  mode?: RefreshSummariesMode
  collectNewPosts?: boolean
  oracleOnline?: boolean
  oracleBaseUrl?: string
}): Promise<{
  ok: boolean
  job_id: string
  total: number
  conflict?: boolean
}> {
  const payload: Record<string, unknown> = {}
  if (options?.mode) payload.mode = options.mode
  if (typeof options?.collectNewPosts === 'boolean') {
    payload.collect_new_posts = options.collectNewPosts
  }
  if (typeof options?.oracleOnline === 'boolean') {
    payload.oracle_online = options.oracleOnline
  }
  if (options?.oracleBaseUrl) {
    payload.oracle_base_url = options.oracleBaseUrl
  }
  const body = Object.keys(payload).length > 0 ? JSON.stringify(payload) : null
  const headers = body ? { 'Content-Type': 'application/json' } : undefined
  const res = await fetch(`${API}/api/refresh-summaries/start`, {
    method: 'POST',
    headers,
    body: body ?? undefined,
  })
  if (res.ok) {
    return json(res)
  }

  if (res.status === 409) {
    const data = (await res.json().catch(() => ({}))) as {
      job_id?: string
      total?: number
      ok?: boolean
    }
    if (!data.job_id) {
      throw new Error('HTTP 409: Missing job id in response')
    }
    return {
      ok: data.ok ?? true,
      job_id: data.job_id,
      total: data.total ?? 0,
      conflict: true,
    }
  }

  const txt = await res.text()
  throw new Error(`HTTP ${res.status}: ${txt}`)
}

export type RefreshJob = {
  id: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled' | 'cancelling'
  total: number
  done: number
  phase: string
  current: string
  message: string
  log_tail: string[]
  new_posts?: {
    post_id: string
    title?: string | null
    interest_score?: number | null
  }[]
  oracle_online?: boolean
  oracle_base_url?: string | null
  oracle_status?: string | null
  oracle_cursor?: {
    platform?: string | null
    post_id?: string | null
    scraped_at?: string | null
  } | null
  oracle_poll_seconds?: number | null
  oracle_idle_since?: number | null
  oracle_progress_total?: number | null
  oracle_progress_index?: number | null
  oracle_progress_stage?: string | null
  oracle_progress_message?: string | null
}

export async function getRefreshJob(jobId: string): Promise<RefreshJob> {
  const res = await fetch(`${API}/api/refresh-summaries/${jobId}`)
  return json(res)
}

export async function getActiveRefreshJob(): Promise<RefreshJob | null> {
  const res = await fetch(`${API}/api/refresh-summaries/active`)
  if (res.status === 404) return null
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(`HTTP ${res.status}: ${txt}`)
  }
  return json(res)
}

export async function stopRefreshJob(jobId: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${API}/api/refresh-summaries/${jobId}/stop`, {
    method: 'POST',
  })
  return json(res)
}

export type CouncilJobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled' | 'cancelling'

export type CouncilJob = {
  id: string
  status: CouncilJobStatus
  total: number
  done: number
  remaining?: number
  current?: string
  log_tail?: string[]
  message?: string
  interest_min?: number
  error?: string
  repairs_total?: number
  repairs_done?: number
  repair_missing?: boolean
  current_mode?: string | null
  current_eta_seconds?: number | null
  current_started_at?: number | null
  current_article_tokens?: number | null
  current_summary_tokens?: number | null
  current_stage?: string | null
  current_stage_label?: string | null
  current_stage_detail?: string | null
}

export type DatabaseBackupInfo = {
  name: string
  path: string
  size: number
  modified: string
  kind: string
}

export type RepairDatabaseResponse = {
  ok: boolean
  message: string
  actions: string[]
  warnings: string[]
  reset_performed: boolean
  restored_backup?: string | null
  backup_path?: string | null
  wal_cleaned: boolean
  integrity_ok?: boolean | null
  needs_reset: boolean
  backups: DatabaseBackupInfo[]
}

export type RepairDatabaseOptions = {
  allowReset?: boolean
  restoreBackup?: string
}

export async function startCouncilAnalysis(options: {
  interest_min: number
  repair_missing?: boolean
  postIds?: string[]
}): Promise<{
  ok: boolean
  job_id: string
  total: number
  interest_min: number
  repairs_total: number
  conflict?: boolean
}> {
  const body: Record<string, unknown> = {
    interest_min: options.interest_min,
    repair_missing: Boolean(options.repair_missing),
  }
  if (options.postIds && options.postIds.length > 0) {
    body.post_ids = options.postIds
  }
  const res = await fetch(`${API}/api/council-analysis/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (res.ok) {
    return json(res)
  }

  if (res.status === 409) {
    const data = (await res.json().catch(() => ({}))) as {
      job_id?: string
      total?: number
      interest_min?: number
      repairs_total?: number
    }
    if (!data.job_id) {
      throw new Error('HTTP 409: Missing job id in response')
    }
    return {
      ok: true,
      job_id: data.job_id,
      total: data.total ?? 0,
      interest_min: typeof data.interest_min === 'number' ? data.interest_min : options.interest_min,
      repairs_total: typeof data.repairs_total === 'number' ? data.repairs_total : 0,
      conflict: true,
    }
  }

  const txt = await res.text()
  throw new Error(`HTTP ${res.status}: ${txt}`)
}

export async function getCouncilJob(jobId: string): Promise<CouncilJob> {
  const res = await fetch(`${API}/api/council-analysis/${jobId}`)
  return json(res)
}

export async function getActiveCouncilJob(): Promise<{
  job_id: string
  status: CouncilJobStatus
  total?: number
  done?: number
  remaining?: number
  interest_min?: number
} | null> {
  const res = await fetch(`${API}/api/council-analysis/active`)
  if (res.status === 204) return null
  return json(res)
}

export async function stopCouncilJob(jobId: string): Promise<{ ok: boolean; status: CouncilJobStatus }> {
  const res = await fetch(`${API}/api/council-analysis/${jobId}/stop`, {
    method: 'POST',
  })
  return json(res)
}
