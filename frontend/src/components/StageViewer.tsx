import { clampSpamPct, spamClassForPct } from '../utils/spam'
import JsonViewer from './JsonViewer'

type StageViewerProps = {
  name: string
  payload: unknown
  placeholder?: string
  ticker?: string | null
}

export type SummariserAsset = {
  ticker?: string | null
  name_or_description?: string | null
  exchange_or_market?: string | null
}

export type SummariserNumber = {
  label?: string | null
  value?: string | number | null
  unit?: string | null
}

export type QualityFlags = Record<string, boolean>

type SummariserSpamMeta = {
  hasData: boolean
  pct: number
  display: string
  trackClass: string
  reason: string
  reasons: string[]
}

export type SummariserData = {
  summaryBullets: string[]
  catalysts: string[]
  risks: string[]
  numbers: SummariserNumber[]
  assets: SummariserAsset[]
  stance: string | null
  qualityFlags: QualityFlags | null
  why: string
  extraFields: Record<string, unknown> | null
  spam: SummariserSpamMeta
  raw: Record<string, unknown>
}

export type ConversationHubData = {
  tickers: string[]
  appended: number
  skippedOld: number
  reason: string
  summary: string
  direction: string
  impact: string
  why: string
  categories: string[]
  timestamp: string
  url: string
  source: string
  postId: string
  channel?: string
  ingestedAt?: string
}

const STANCE_LABELS: Record<string, string> = {
  bullish: 'Bullish',
  bearish: 'Bearish',
  neutral: 'Neutral',
  uncertain: 'Uncertain',
}

function tryParse(value: unknown): unknown {
  if (typeof value === 'string') {
    try {
      return JSON.parse(value)
    } catch (err) {
      return undefined
    }
  }
  return value
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => (item == null ? '' : String(item).trim()))
    .filter((item) => item.length > 0)
}

function asAssetArray(value: unknown): SummariserAsset[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => (isPlainObject(item) ? (item as SummariserAsset) : undefined))
    .filter((item): item is SummariserAsset => !!item)
}

function asNumberArray(value: unknown): SummariserNumber[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => (isPlainObject(item) ? (item as SummariserNumber) : undefined))
    .filter((item): item is SummariserNumber => !!item)
}

function asQualityFlags(value: unknown): QualityFlags | null {
  if (!isPlainObject(value)) return null
  const entries = Object.entries(value).filter(([, v]) => typeof v === 'boolean')
  if (!entries.length) return null
  return Object.fromEntries(entries) as QualityFlags
}

function formatKey(label: string): string {
  return label
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function extractSpamMeta(data: Record<string, unknown>): SummariserSpamMeta {
  const spamReasonsRaw = data.spam_reasons ?? data.spam_why
  const spamReasons = Array.isArray(spamReasonsRaw)
    ? spamReasonsRaw
        .map((reason) => (reason == null ? '' : String(reason).trim()))
        .filter((reason) => reason.length > 0)
    : (() => {
        if (spamReasonsRaw == null) return []
        const reason = String(spamReasonsRaw).trim()
        return reason.length > 0 ? [reason] : []
      })()

  const spamPctRaw = data.spam_likelihood_pct
  let spamPctValue: number | null = null
  if (typeof spamPctRaw === 'number' && Number.isFinite(spamPctRaw)) {
    spamPctValue = spamPctRaw
  } else if (typeof spamPctRaw === 'string') {
    const parsed = Number.parseFloat(spamPctRaw)
    if (Number.isFinite(parsed)) {
      spamPctValue = parsed
    }
  }

  let hasSpamData = false
  let spamPct = 0
  if (typeof spamPctValue === 'number') {
    hasSpamData = true
    spamPct = clampSpamPct(spamPctValue)
  }

  const trackClass = spamClassForPct(hasSpamData ? spamPct : null)
  const display = hasSpamData ? `${Math.round(spamPct)}%` : '—'
  const reason = spamReasons.length > 0 ? spamReasons.join('; ') : 'not assessed'

  return {
    hasData: hasSpamData,
    pct: spamPct,
    display,
    trackClass,
    reason,
    reasons: spamReasons,
  }
}

const toStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => (item == null ? '' : String(item).trim()))
    .filter((item) => item.length > 0)
}

const toNumber = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number.parseInt(value, 10)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return 0
}

const toFloat = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

const toInteger = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value)
  }
  if (typeof value === 'string') {
    const parsed = Number.parseInt(value, 10)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

const normalizeTicker = (value?: string | null): string => {
  if (typeof value !== 'string') return ''
  return value.trim().toUpperCase()
}

type TickerSelection = {
  ticker: string
  record: Record<string, unknown>
  container: Record<string, unknown> | null
}

function selectTickerRecord(payload: unknown, ticker?: string | null): TickerSelection | null {
  const parsed = tryParse(payload)
  if (isPlainObject(parsed)) {
    const parsedRecord = parsed as Record<string, unknown>
    const tickersValue = parsedRecord.tickers
    if (isPlainObject(tickersValue)) {
      const entries = Object.entries(tickersValue as Record<string, unknown>).filter(([, value]) =>
        isPlainObject(value),
      )
      if (entries.length > 0) {
        const normalizedMap = new Map<string, Record<string, unknown>>()
        entries.forEach(([key, value]) => {
          normalizedMap.set(normalizeTicker(key), value as Record<string, unknown>)
        })

        const orderedRaw = parsedRecord.ordered ?? parsedRecord.ordered_tickers
        const orderedKeys = Array.isArray(orderedRaw)
          ? orderedRaw
              .map((item) => {
                if (typeof item === 'string') return item
                if (item == null) return ''
                return String(item)
              })
              .map((item) => normalizeTicker(item))
              .filter((item) => item.length > 0)
          : []

        const normalizedTicker = normalizeTicker(ticker)
        const fallbackKeys = [...orderedKeys, ...normalizedMap.keys()]
        const selectedKey =
          (normalizedTicker && normalizedMap.has(normalizedTicker) ? normalizedTicker : undefined) ||
          fallbackKeys.find((candidate) => normalizedMap.has(candidate))

        if (selectedKey) {
          return {
            ticker: selectedKey,
            record: normalizedMap.get(selectedKey) as Record<string, unknown>,
            container: parsedRecord,
          }
        }
      }
    }

    if (typeof parsedRecord.ticker === 'string' || ticker) {
      const selectedTicker = normalizeTicker(
        typeof parsedRecord.ticker === 'string' ? parsedRecord.ticker : ticker ?? undefined,
      )
      return {
        ticker: selectedTicker,
        record: parsedRecord,
        container: parsedRecord,
      }
    }
  }

  if (isPlainObject(payload)) {
    const record = payload as Record<string, unknown>
    return {
      ticker: normalizeTicker(ticker),
      record,
      container: null,
    }
  }

  return null
}

export type ResearchPlanStep = {
  index: number
  tool: string
  covers: string[]
  tests: number[]
  why?: string
  passIf?: string
  failIf?: string
}

export type ResearcherStageData = {
  ticker: string
  hypotheses: { text: string; type?: string }[]
  rationale: string
  stageWhy?: string
  planWhy?: string
  steps: ResearchPlanStep[]
  runAt?: string
  sessionId?: string
  log?: string
}

export type ResearchInsightEntry = {
  tool: string
  text: string
  status?: string
}

export type TechnicalResultEntry = {
  tool: string
  status: string
  error?: string
  result?: Record<string, unknown> | null
  why?: string
  passIf?: string
  failIf?: string
}

export type TechnicalResearchStageData = {
  ticker: string
  steps: ResearchPlanStep[]
  results: TechnicalResultEntry[]
  insights: ResearchInsightEntry[]
  summaryLines: string[]
  status?: string
  error?: string
  runAt?: string
}

export type SentimentDelta = {
  timestamp?: string
  summary?: string
  direction?: string
  impact?: string
  why?: string
  channel?: string
  categories?: string[]
  source?: string
  url?: string
}

export function parseConversationHubData(payload: unknown): ConversationHubData | null {
  if (!isPlainObject(payload)) return null
  const record = payload as Record<string, unknown>
  const delta = isPlainObject(record.delta) ? (record.delta as Record<string, unknown>) : null

  const tickers = toStringArray(record.tickers)
  const categories = toStringArray(delta?.cat)
  const appended = toNumber(record.appended)
  const skippedOld = toNumber(record.skipped_old ?? record.skippedOld)
  const reason = typeof record.reason === 'string' ? record.reason : ''

  const summary = typeof delta?.sum === 'string' ? delta.sum.trim() : ''
  const direction = typeof delta?.dir === 'string' ? delta.dir.trim().toLowerCase() : ''
  const impact = typeof delta?.impact === 'string' ? delta.impact.trim().toLowerCase() : ''
  const why = typeof delta?.why === 'string' ? delta.why.trim() : ''
  const timestamp =
    typeof record.ts === 'string'
      ? record.ts
      : typeof delta?.t === 'string'
        ? delta.t
        : ''
  const url = typeof record.url === 'string' ? record.url : typeof delta?.url === 'string' ? delta.url : ''
  const source = typeof record.source === 'string' ? record.source : ''
  const postId =
    typeof record.post_id === 'string'
      ? record.post_id
      : typeof delta?.post_id === 'string'
        ? delta.post_id
        : ''
  const channel = typeof delta?.chan === 'string' ? delta.chan : undefined
  const ingestedAt = typeof record.ingested_at === 'string' ? record.ingested_at : undefined

  if (!tickers.length && !summary && !why && !direction && !impact) {
    return null
  }

  return {
    tickers,
    appended,
    skippedOld,
    reason,
    summary,
    direction,
    impact,
    why,
    categories,
    timestamp,
    url,
    source,
    postId,
    channel,
    ingestedAt,
  }
}

export type SentimentResearchData = {
  ticker: string
  asOf: string
  channel?: string
  windowDays?: number | null
  desRaw?: number | null
  desSector?: number | null
  desIdio?: number | null
  confidence?: number | null
  nDeltas?: number | null
  desAdj?: number | null
  narrative?: string
  narrativeError?: string
  error?: string
  runAt?: string
  raw: Record<string, unknown> | null
  deltas: SentimentDelta[]
  latestDelta?: SentimentDelta
  articleDelta?: SentimentDelta
}

const toSentimentDelta = (value: unknown): SentimentDelta | null => {
  if (!isPlainObject(value)) return null
  const record = value as Record<string, unknown>
  const timestamp =
    typeof record.t === 'string'
      ? record.t
      : typeof record.timestamp === 'string'
        ? record.timestamp
        : undefined
  const summary =
    typeof record.sum === 'string'
      ? record.sum
      : typeof record.summary === 'string'
        ? record.summary
        : undefined
  const why = typeof record.why === 'string' ? record.why : undefined
  const direction = typeof record.dir === 'string' ? record.dir : undefined
  const impact = typeof record.impact === 'string' ? record.impact : undefined
  const channel = typeof record.chan === 'string' ? record.chan : undefined
  const categoriesRaw = record.cat
  const categories = Array.isArray(categoriesRaw)
    ? categoriesRaw
        .map((item) => {
          if (item == null) return ''
          return String(item).trim()
        })
        .filter((item) => item.length > 0)
    : []
  const source = typeof record.src === 'string' ? record.src : undefined
  const url = typeof record.url === 'string' ? record.url : undefined

  return {
    timestamp,
    summary,
    why,
    direction,
    impact,
    channel,
    categories,
    source,
    url,
  }
}

function pickSentimentRecord(payload: unknown): Record<string, unknown> | null {
  if (!isPlainObject(payload)) return null
  const base = payload as Record<string, unknown>
  if (isPlainObject(base['result'])) {
    return base['result'] as Record<string, unknown>
  }
  return base
}

export function parseSentimentResearchData(
  payload: unknown,
  ticker?: string | null,
): SentimentResearchData | null {
  const selection = selectTickerRecord(payload, ticker)
  const parsed = tryParse(payload)
  const baseRecord = selection?.record
    ? selection.record
    : isPlainObject(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  if (!baseRecord) return null

  const sentimentRecord = pickSentimentRecord(baseRecord) ?? baseRecord

  const tickerSource =
    sentimentRecord?.['ticker'] ?? baseRecord['ticker'] ?? selection?.ticker ?? normalizeTicker(ticker)
  const asOfSource = sentimentRecord?.['as_of'] ?? baseRecord['as_of']
  const channelSource = sentimentRecord?.['channel'] ?? baseRecord['channel']
  const windowSource = sentimentRecord?.['window_days'] ?? baseRecord['window_days']

  const desRawSource = sentimentRecord?.['des_raw']
  const desSectorSource = sentimentRecord?.['des_sector']
  const desIdioSource = sentimentRecord?.['des_idio']
  const confidenceSource = sentimentRecord?.['confidence']
  const nDeltasSource = sentimentRecord?.['n_deltas']
  const desAdjSource = sentimentRecord?.['DES_adj'] ?? sentimentRecord?.['des_adj']

  const tickerValue = typeof tickerSource === 'string' ? tickerSource : ''
  const asOf = typeof asOfSource === 'string' ? asOfSource : ''
  const runAtCandidate =
    typeof baseRecord.run_at === 'string'
      ? (baseRecord.run_at as string)
      : typeof selection?.container?.updated_at === 'string'
        ? (selection?.container?.updated_at as string)
        : undefined

  const narrative =
    typeof sentimentRecord?.['narrative'] === 'string'
      ? (sentimentRecord?.['narrative'] as string).trim()
      : typeof baseRecord['narrative'] === 'string'
        ? (baseRecord['narrative'] as string).trim()
        : undefined
  const narrativeError =
    typeof sentimentRecord?.['narrative_error'] === 'string'
      ? (sentimentRecord?.['narrative_error'] as string)
      : typeof baseRecord['narrative_error'] === 'string'
        ? (baseRecord['narrative_error'] as string)
        : undefined

  const error =
    typeof sentimentRecord?.['error'] === 'string'
      ? (sentimentRecord?.['error'] as string)
      : typeof baseRecord['error'] === 'string'
        ? (baseRecord['error'] as string)
        : undefined

  const deltasRaw = sentimentRecord?.['deltas'] ?? baseRecord['deltas']
  const deltas = Array.isArray(deltasRaw)
    ? (deltasRaw as unknown[])
        .map((item) => toSentimentDelta(item))
        .filter((item): item is SentimentDelta => !!item)
    : []

  const latestDelta =
    toSentimentDelta(sentimentRecord?.['latest_delta'] ?? baseRecord['latest_delta']) ||
    (deltas.length > 0 ? deltas[deltas.length - 1] : undefined)

  const articleDeltaRaw = toSentimentDelta(
    sentimentRecord?.['article_delta'] ?? baseRecord['article_delta'],
  )
  const articleDelta = articleDeltaRaw ?? undefined

  return {
    ticker: tickerValue,
    asOf,
    channel: typeof channelSource === 'string' ? channelSource : undefined,
    windowDays: toInteger(windowSource),
    desRaw: toFloat(desRawSource),
    desSector: toFloat(desSectorSource),
    desIdio: toFloat(desIdioSource),
    confidence: toFloat(confidenceSource),
    nDeltas: toInteger(nDeltasSource),
    desAdj: toFloat(desAdjSource),
    narrative,
    narrativeError,
    error,
    runAt: runAtCandidate,
    raw: sentimentRecord ?? baseRecord,
    deltas,
    latestDelta,
    articleDelta,
  }
}

type ChairmanNextCheck = {
  metric: string
  op?: string | null
  value?: number | string | null
  action?: string | null
}

export type ChairmanStageData = {
  plainEnglish: string
  ticker: string
  timeframe: string
  direction: string
  directionStrength: number | null
  riskLevel: string
  conviction: number | null
  tradability: number | null
  uncertainty: number | null
  staleRisk: number | null
  catalysts: string[]
  watchouts: string[]
  blockingIssues: string[]
  dataGaps: string[]
  nextChecks: ChairmanNextCheck[]
  technical: Record<string, unknown>
  sentiment: Record<string, unknown>
  timestamps: Record<string, unknown>
  why?: string
  verifierUsed: boolean | null
  verifierNotes: string[]
  raw: Record<string, unknown>
}

function parseChairmanNextChecks(value: unknown): ChairmanNextCheck[] {
  if (!Array.isArray(value)) return []
  const checks: ChairmanNextCheck[] = []
  for (const item of value as unknown[]) {
    if (!isPlainObject(item)) continue
    const entry = item as Record<string, unknown>
    const metric = typeof entry.metric === 'string' ? entry.metric.trim() : ''
    if (!metric) continue
    const op = typeof entry.op === 'string' ? entry.op : null
    const action = typeof entry.action === 'string' ? entry.action : null
    const rawVal = entry.value
    const valueOut =
      typeof rawVal === 'number'
        ? rawVal
        : typeof rawVal === 'string'
          ? rawVal
          : rawVal == null
            ? null
            : String(rawVal)
    checks.push({ metric, op: op ?? null, value: valueOut, action })
  }
  return checks
}

const toRecord = (value: unknown): Record<string, unknown> =>
  isPlainObject(value) ? (value as Record<string, unknown>) : {}

export function parseChairmanStageData(payload: unknown): ChairmanStageData | null {
  const parsed = tryParse(payload)
  if (!isPlainObject(parsed)) return null
  const record = parsed as Record<string, unknown>
  const finalMetricsRaw = record.final_metrics
  const finalMetrics = toRecord(finalMetricsRaw)

  const plainEnglishRaw = record.plain_english_result
  const plainEnglish =
    typeof plainEnglishRaw === 'string' ? plainEnglishRaw.trim() : ''

  const technical = toRecord(finalMetrics.technical)
  const sentiment = toRecord(finalMetrics.sentiment)
  const timestamps = toRecord(finalMetrics.timestamps)
  const verifierBlock = toRecord(finalMetrics.verifier)

  return {
    plainEnglish,
    ticker: typeof finalMetrics.ticker === 'string' ? finalMetrics.ticker : '',
    timeframe: typeof finalMetrics.timeframe === 'string' ? finalMetrics.timeframe : '',
    direction:
      typeof finalMetrics.implied_direction === 'string'
        ? finalMetrics.implied_direction
        : '',
    directionStrength: toInteger(finalMetrics.direction_strength),
    riskLevel: typeof finalMetrics.risk_level === 'string' ? finalMetrics.risk_level : '',
    conviction: toInteger(finalMetrics.conviction_0to100),
    tradability: toInteger(finalMetrics.tradability_score_0to100),
    uncertainty: toInteger(finalMetrics.uncertainty_0to3),
    staleRisk: toInteger(finalMetrics.stale_risk_0to3),
    catalysts: toStringArray(finalMetrics.catalysts),
    watchouts: toStringArray(finalMetrics.watchouts),
    blockingIssues: toStringArray(finalMetrics.blocking_issues),
    dataGaps: toStringArray(finalMetrics.data_gaps),
    nextChecks: parseChairmanNextChecks(finalMetrics.next_checks),
    technical,
    sentiment,
    timestamps,
    why: typeof finalMetrics.why === 'string' ? finalMetrics.why : undefined,
    verifierUsed:
      typeof verifierBlock.used === 'boolean'
        ? (verifierBlock.used as boolean)
        : null,
    verifierNotes: toStringArray(verifierBlock.notes),
    raw: record,
  }
}

export function parseResearcherStageData(
  payload: unknown,
  ticker?: string | null,
): ResearcherStageData | null {
  const selection = selectTickerRecord(payload, ticker)
  if (!selection) return null

  const record = selection.record
  const stage1 = isPlainObject(record.stage1) ? (record.stage1 as Record<string, unknown>) : {}
  const hypothesesRaw = Array.isArray(stage1.hypotheses) ? (stage1.hypotheses as unknown[]) : []
  const hypotheses = hypothesesRaw
    .map((item) => (isPlainObject(item) ? (item as Record<string, unknown>) : null))
    .filter((item): item is Record<string, unknown> => !!item)
    .map((item) => ({
      text: typeof item.text === 'string' ? item.text : '',
      type: typeof item.type === 'string' ? item.type : undefined,
    }))
    .filter((item) => item.text.length > 0)

  const rationale = typeof stage1.rationale === 'string' ? stage1.rationale : ''
  const stageWhy = typeof stage1.why === 'string' ? stage1.why : undefined

  const plan = isPlainObject(record.plan) ? (record.plan as Record<string, unknown>) : {}
  const planWhy = typeof plan.why === 'string' ? plan.why : undefined
  const stepsRaw = Array.isArray(plan.steps) ? (plan.steps as unknown[]) : []
  const steps: ResearchPlanStep[] = []
  stepsRaw.forEach((item, idx) => {
    if (!isPlainObject(item)) return
    const step = item as Record<string, unknown>
    const tool = typeof step.tool === 'string' ? step.tool : `step-${idx + 1}`
    const covers = Array.isArray(step.covers)
      ? (step.covers as unknown[])
          .map((cover) => {
            if (cover == null) return ''
            return String(cover).trim()
          })
          .filter((cover) => cover.length > 0)
      : []
    const testsRaw = Array.isArray(step.tests) ? (step.tests as unknown[]) : []
    const tests = testsRaw
      .map((val) => toInteger(val))
      .filter((val): val is number => typeof val === 'number')
    const why = typeof step.why === 'string' ? step.why : undefined
    const passIf = typeof step.pass_if === 'string' ? step.pass_if : undefined
    const failIf = typeof step.fail_if === 'string' ? step.fail_if : undefined
    steps.push({ index: idx + 1, tool, covers, tests, why, passIf, failIf })
  })

  const runAt =
    typeof record.run_at === 'string'
      ? (record.run_at as string)
      : typeof selection.container?.updated_at === 'string'
        ? (selection.container?.updated_at as string)
        : undefined
  const sessionId = typeof record.session_id === 'string' ? (record.session_id as string) : undefined
  const log = typeof record.log === 'string' ? (record.log as string) : undefined

  return {
    ticker: selection.ticker || normalizeTicker(ticker),
    hypotheses,
    rationale,
    stageWhy,
    planWhy,
    steps,
    runAt,
    sessionId,
    log,
  }
}

export function parseTechnicalResearchData(
  payload: unknown,
  ticker?: string | null,
): TechnicalResearchStageData | null {
  const selection = selectTickerRecord(payload, ticker)
  if (!selection) return null

  const record = selection.record
  const stepsRaw = Array.isArray(record.steps) ? (record.steps as unknown[]) : []
  const steps: ResearchPlanStep[] = []
  stepsRaw.forEach((item, idx) => {
    if (!isPlainObject(item)) return
    const step = item as Record<string, unknown>
    const tool = typeof step.tool === 'string' ? step.tool : `step-${idx + 1}`
    const covers = Array.isArray(step.covers)
      ? (step.covers as unknown[])
          .map((cover) => (cover == null ? '' : String(cover).trim()))
          .filter((cover) => cover.length > 0)
      : []
    const testsRaw = Array.isArray(step.tests) ? (step.tests as unknown[]) : []
    const tests = testsRaw
      .map((val) => toInteger(val))
      .filter((val): val is number => typeof val === 'number')
    const why = typeof step.why === 'string' ? step.why : undefined
    const passIf = typeof step.pass_if === 'string' ? step.pass_if : undefined
    const failIf = typeof step.fail_if === 'string' ? step.fail_if : undefined
    steps.push({ index: idx + 1, tool, covers, tests, why, passIf, failIf })
  })

  const resultsRaw = Array.isArray(record.results) ? (record.results as unknown[]) : []
  const results: TechnicalResultEntry[] = []
  resultsRaw.forEach((item) => {
    if (!isPlainObject(item)) return
    const res = item as Record<string, unknown>
    const tool = typeof res.tool === 'string' ? res.tool : 'unknown'
    const status = typeof res.status === 'string' ? res.status : 'unknown'
    const error = typeof res.error === 'string' ? res.error : undefined
    const resultData = isPlainObject(res.result) ? (res.result as Record<string, unknown>) : null
    const why = typeof res.why === 'string' ? res.why : undefined
    const passIf = typeof res.pass_if === 'string' ? res.pass_if : undefined
    const failIf = typeof res.fail_if === 'string' ? res.fail_if : undefined
    results.push({ tool, status, error, result: resultData, why, passIf, failIf })
  })

  const insightsRaw = Array.isArray(record.insights) ? (record.insights as unknown[]) : []
  const insights: ResearchInsightEntry[] = []
  insightsRaw.forEach((item) => {
    if (!isPlainObject(item)) return
    const entry = item as Record<string, unknown>
    const tool = typeof entry.tool === 'string' ? entry.tool : ''
    const text = typeof entry.text === 'string' ? entry.text : ''
    const status = typeof entry.status === 'string' ? entry.status : undefined
    if (!tool || !text) return
    insights.push({ tool, text, status })
  })

  const summaryLinesRaw = Array.isArray(record.summary_lines)
    ? (record.summary_lines as unknown[])
    : []
  const summaryLines = summaryLinesRaw
    .map((item) => {
      if (item == null) return ''
      return String(item).trim()
    })
    .filter((item) => item.length > 0)

  const status = typeof record.status === 'string' ? (record.status as string) : undefined
  const error = typeof record.error === 'string' ? (record.error as string) : undefined

  const runAt =
    typeof record.run_at === 'string'
      ? (record.run_at as string)
      : typeof selection.container?.updated_at === 'string'
        ? (selection.container?.updated_at as string)
        : undefined

  return {
    ticker: selection.ticker || normalizeTicker(ticker),
    steps,
    results,
    insights,
    summaryLines,
    status,
    error,
    runAt,
  }
}

export function parseSummariserData(payload: unknown): SummariserData | null {
  const parsed = tryParse(payload)
  if (!isPlainObject(parsed)) return null

  const record = parsed as Record<string, unknown>
  const summaryBullets = asStringArray(record.summary_bullets)
  const catalysts = asStringArray(record.claimed_catalysts)
  const risks = asStringArray(record.claimed_risks)
  const numbers = asNumberArray(record.numbers_mentioned)
  const assets = asAssetArray(record.assets_mentioned)
  const stance = typeof record.author_stance === 'string' ? record.author_stance : null
  const qualityFlags = asQualityFlags(record.quality_flags)
  const why = typeof record.why === 'string' ? record.why.trim() : ''
  const spam = extractSpamMeta(record)

  const knownKeys = new Set([
    'summary_bullets',
    'assets_mentioned',
    'claimed_catalysts',
    'claimed_risks',
    'numbers_mentioned',
    'author_stance',
    'quality_flags',
    'why',
    'spam_likelihood_pct',
    'spam_why',
    'spam_reasons',
  ])
  const extraEntries = Object.entries(record).filter(([key]) => !knownKeys.has(key))
  const extraFields = extraEntries.length > 0 ? Object.fromEntries(extraEntries) : null

  return {
    summaryBullets,
    catalysts,
    risks,
    numbers,
    assets,
    stance,
    qualityFlags,
    why,
    extraFields,
    spam,
    raw: record,
  }
}

const DIRECTION_LABELS: Record<string, string> = {
  up: 'Bullish',
  down: 'Bearish',
  neutral: 'Neutral',
}

const DIRECTION_STRENGTH_LABELS = ['None', 'Low', 'Medium', 'High']
const SCALE_LABELS = ['Low', 'Moderate', 'High', 'Severe']

const CHAIRMAN_TECH_FIELDS: { key: string; label: string; format: 'number' | 'percent' | 'boolean' | 'string' }[] = [
  { key: 'close', label: 'Close', format: 'number' },
  { key: 'price_window_close_pct', label: 'Price Window %', format: 'percent' },
  { key: 'rsi14', label: 'RSI (14)', format: 'number' },
  { key: 'macd_hist', label: 'MACD Histogram', format: 'number' },
  { key: 'trend_direction', label: 'Trend Direction', format: 'string' },
  { key: 'trend_strength', label: 'Trend Strength', format: 'number' },
  { key: 'price_above_sma20', label: 'Above 20DMA', format: 'boolean' },
  { key: 'price_above_sma50', label: 'Above 50DMA', format: 'boolean' },
  { key: 'price_above_sma200', label: 'Above 200DMA', format: 'boolean' },
  { key: 'vol_state', label: 'Volatility State', format: 'string' },
  { key: 'bollinger_last_event', label: 'Bollinger Event', format: 'string' },
  { key: 'mfi', label: 'Money Flow Index', format: 'number' },
]

const CHAIRMAN_SENTIMENT_FIELDS: { key: string; label: string; format: 'number' | 'string' }[] = [
  { key: 'des_raw', label: 'DES (Raw)', format: 'number' },
  { key: 'idio', label: 'DES Idio', format: 'number' },
  { key: 'conf', label: 'Confidence', format: 'number' },
  { key: 'deltas', label: 'Delta Signal', format: 'string' },
]

const capitalizeWord = (value: string): string => {
  if (!value) return ''
  return value.charAt(0).toUpperCase() + value.slice(1)
}

const formatDirectionLabel = (direction: string, strength: number | null): string | null => {
  if (!direction) return null
  const normalized = direction.toLowerCase()
  const base = DIRECTION_LABELS[normalized] ?? capitalizeWord(direction.replace(/_/g, ' '))
  if (typeof strength === 'number' && Number.isFinite(strength)) {
    const idx = Math.max(0, Math.min(DIRECTION_STRENGTH_LABELS.length - 1, Math.round(strength)))
    const suffix = DIRECTION_STRENGTH_LABELS[idx]
    if (suffix && suffix !== 'None') {
      return `${base} · ${suffix}`
    }
  }
  return base
}

const formatTimeframeLabel = (timeframe: string): string | null => {
  if (!timeframe) return null
  return timeframe
    .split(/[_\s]+/)
    .filter((part) => part.length > 0)
    .map((part) => capitalizeWord(part))
    .join(' ')
}

const formatScaleLabel = (value: number | null | undefined): string | null => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  const idx = Math.max(0, Math.min(SCALE_LABELS.length - 1, Math.round(value)))
  return SCALE_LABELS[idx] ?? value.toString()
}

const formatPercentLabel = (value: number | null | undefined): string | null => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return `${Math.round(value)}%`
}

const formatBooleanLabel = (value: unknown): string => {
  if (value === true) return 'Yes'
  if (value === false) return 'No'
  return '—'
}

const formatNumericLabel = (value: unknown, fractionDigits = 2): string => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toLocaleString(undefined, { maximumFractionDigits: fractionDigits })
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : '—'
  }
  return '—'
}

const formatTableValue = (
  value: unknown,
  format: 'number' | 'percent' | 'boolean' | 'string',
): string => {
  if (format === 'boolean') {
    return formatBooleanLabel(value)
  }
  if (format === 'percent') {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}%`
    }
    if (typeof value === 'string') {
      const trimmed = value.trim()
      if (trimmed.length === 0) return '—'
      return trimmed.endsWith('%') ? trimmed : `${trimmed}%`
    }
    return '—'
  }
  if (format === 'number') {
    return formatNumericLabel(value)
  }
  if (value == null) return '—'
  if (typeof value === 'string') {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : '—'
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return '—'
    const joined = value
      .map((item) => (item == null ? '' : String(item).trim()))
      .filter((item) => item.length > 0)
      .join(', ')
    return joined.length > 0 ? joined : value.length.toString()
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch (err) {
      return '—'
    }
  }
  return String(value)
}

const formatNextCheckCondition = (check: ChairmanNextCheck): string => {
  const pieces: string[] = []
  if (check.op) {
    pieces.push(check.op.replace(/_/g, ' '))
  }
  if (check.value !== undefined && check.value !== null) {
    if (typeof check.value === 'number' && Number.isFinite(check.value)) {
      pieces.push(check.value.toLocaleString(undefined, { maximumFractionDigits: 2 }))
    } else {
      pieces.push(String(check.value))
    }
  }
  return pieces.length > 0 ? pieces.join(' ') : '—'
}

function ChairmanStageView({ data }: { data: ChairmanStageData }) {
  const headlineItems = [
    { label: 'Ticker', value: data.ticker || null },
    { label: 'Direction', value: formatDirectionLabel(data.direction, data.directionStrength) },
    { label: 'Risk Level', value: data.riskLevel ? capitalizeWord(data.riskLevel) : null },
    { label: 'Timeframe', value: formatTimeframeLabel(data.timeframe) },
    { label: 'Conviction', value: formatPercentLabel(data.conviction) },
    { label: 'Tradability', value: formatPercentLabel(data.tradability) },
    { label: 'Uncertainty', value: formatScaleLabel(data.uncertainty) },
    { label: 'Stale Risk', value: formatScaleLabel(data.staleRisk) },
  ].filter((item) => item.value)

  const technicalRows = CHAIRMAN_TECH_FIELDS.map(({ key, label, format }) => ({
    label,
    value: formatTableValue((data.technical as Record<string, unknown>)[key], format),
  })).filter((row) => row.value !== '—')

  const sentimentRows = CHAIRMAN_SENTIMENT_FIELDS.map(({ key, label, format }) => ({
    label,
    value: formatTableValue((data.sentiment as Record<string, unknown>)[key], format === 'number' ? 'number' : 'string'),
  })).filter((row) => row.value !== '—')

  const timestampEntries = Object.entries(data.timestamps)
    .map(([key, value]) => ({
      label: formatKey(key),
      value: typeof value === 'string' ? value : typeof value === 'number' ? value.toString() : '',
    }))
    .filter((entry) => entry.value.trim().length > 0)

  return (
    <div className="stage-content chairman-stage">
      {data.plainEnglish && (
        <section className="stage-section">
          <h4>Plain English Verdict</h4>
          <p className="chairman-plain">{data.plainEnglish}</p>
        </section>
      )}

      {headlineItems.length > 0 && (
        <section className="stage-section">
          <h4>Headline Metrics</h4>
          <dl className="chairman-metric-grid">
            {headlineItems.map((item) => (
              <div key={item.label} className="chairman-metric">
                <dt>{item.label}</dt>
                <dd>{item.value}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {data.why && (
        <section className="stage-section">
          <h4>Why</h4>
          <p>{data.why}</p>
        </section>
      )}

      {(data.catalysts.length > 0 || data.watchouts.length > 0) && (
        <section className="stage-section stage-flex">
          {data.catalysts.length > 0 && (
            <div className="stage-column">
              <h4>Catalysts</h4>
              <ul className="stage-list">
                {data.catalysts.map((item, idx) => (
                  <li key={`catalyst-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}
          {data.watchouts.length > 0 && (
            <div className="stage-column">
              <h4>Watchouts</h4>
              <ul className="stage-list">
                {data.watchouts.map((item, idx) => (
                  <li key={`watchout-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {(data.blockingIssues.length > 0 || data.dataGaps.length > 0) && (
        <section className="stage-section stage-flex">
          {data.blockingIssues.length > 0 && (
            <div className="stage-column">
              <h4>Blocking Issues</h4>
              <ul className="stage-list">
                {data.blockingIssues.map((item, idx) => (
                  <li key={`block-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}
          {data.dataGaps.length > 0 && (
            <div className="stage-column">
              <h4>Data Gaps</h4>
              <ul className="stage-list">
                {data.dataGaps.map((item, idx) => (
                  <li key={`gap-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {data.nextChecks.length > 0 && (
        <section className="stage-section">
          <h4>Next Checks</h4>
          <table className="stage-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Condition</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {data.nextChecks.map((check, idx) => (
                <tr key={`next-${idx}`}>
                  <td>{check.metric}</td>
                  <td>{formatNextCheckCondition(check)}</td>
                  <td>{check.action ? capitalizeWord(check.action) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {(data.verifierUsed !== null || data.verifierNotes.length > 0) && (
        <section className="stage-section">
          <h4>Verifier</h4>
          {data.verifierUsed !== null && (
            <p>{data.verifierUsed ? 'Verifier insights were incorporated.' : 'Verifier not used in synthesis.'}</p>
          )}
          {data.verifierNotes.length > 0 && (
            <ul className="stage-list">
              {data.verifierNotes.map((note, idx) => (
                <li key={`verifier-${idx}`}>{note}</li>
              ))}
            </ul>
          )}
        </section>
      )}

      {technicalRows.length > 0 && (
        <section className="stage-section">
          <h4>Technical Snapshot</h4>
          <table className="stage-table stage-kv">
            <tbody>
              {technicalRows.map((row) => (
                <tr key={row.label}>
                  <th>{row.label}</th>
                  <td>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {sentimentRows.length > 0 && (
        <section className="stage-section">
          <h4>Sentiment Snapshot</h4>
          <table className="stage-table stage-kv">
            <tbody>
              {sentimentRows.map((row) => (
                <tr key={row.label}>
                  <th>{row.label}</th>
                  <td>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {timestampEntries.length > 0 && (
        <section className="stage-section">
          <h4>Timestamps</h4>
          <table className="stage-table stage-kv">
            <tbody>
              {timestampEntries.map((entry) => (
                <tr key={entry.label}>
                  <th>{entry.label}</th>
                  <td>{entry.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <details className="stage-log">
        <summary>Raw Payload</summary>
        <JsonViewer value={data.raw} />
      </details>
    </div>
  )
}

function SummariserView({ data }: { data: SummariserData }) {
  const hasStructuredData =
    data.summaryBullets.length > 0 ||
    data.assets.length > 0 ||
    data.catalysts.length > 0 ||
    data.risks.length > 0 ||
    data.numbers.length > 0 ||
    !!data.stance ||
    !!data.qualityFlags ||
    !!data.why ||
    !!data.extraFields

  if (!hasStructuredData) {
    return <JsonViewer value={data.raw} />
  }

  return (
    <div className="stage-content">
      {data.summaryBullets.length > 0 && (
        <section className="stage-section">
          <h4>Summary</h4>
          <ul className="stage-list stage-bullets">
            {data.summaryBullets.map((bullet, idx) => (
              <li key={idx}>{bullet}</li>
            ))}
          </ul>
        </section>
      )}

      {data.assets.length > 0 && (
        <section className="stage-section">
          <h4>Assets Mentioned</h4>
          <table className="stage-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Name / Description</th>
                <th>Market</th>
              </tr>
            </thead>
            <tbody>
              {data.assets.map((asset, idx) => {
                const name = asset.name_or_description
                const ticker = asset.ticker && String(asset.ticker).trim()
                const market = asset.exchange_or_market && String(asset.exchange_or_market).trim()
                return (
                  <tr key={idx}>
                    <td>{ticker || '—'}</td>
                    <td>
                      {name && String(name).trim().length
                        ? String(name)
                        : <span className="asset-null">null</span>}
                    </td>
                    <td>{market || '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}

      {(data.catalysts.length > 0 || data.risks.length > 0) && (
        <section className="stage-section stage-flex">
          {data.catalysts.length > 0 && (
            <div className="stage-column">
              <h4>Catalysts</h4>
              <ul className="stage-list">
                {data.catalysts.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}
          {data.risks.length > 0 && (
            <div className="stage-column">
              <h4>Risks</h4>
              <ul className="stage-list">
                {data.risks.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {data.numbers.length > 0 && (
        <section className="stage-section">
          <h4>Numbers Mentioned</h4>
          <table className="stage-table">
            <thead>
              <tr>
                <th>Label</th>
                <th>Value</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {data.numbers.map((num, idx) => (
                <tr key={idx}>
                  <td>{num.label ?? '—'}</td>
                  <td>{num.value ?? '—'}</td>
                  <td>{num.unit ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {data.stance && (
        <section className="stage-section">
          <h4>Stance</h4>
          <span className={`stage-stance stance-${data.stance}`}>
            {STANCE_LABELS[data.stance] || formatKey(data.stance)}
          </span>
        </section>
      )}

      {data.qualityFlags && (
        <section className="stage-section">
          <h4>Quality Flags</h4>
          <ul className="stage-flag-list">
            {Object.entries(data.qualityFlags).map(([key, value]) => {
              const keyLower = key.toLowerCase()
              const highlightWhenTrue =
                keyLower !== 'vague_claims' && keyLower !== 'repetition_or_template'
              const isHighlighted = highlightWhenTrue ? value : !value
              return (
                <li key={key} className={`stage-flag ${isHighlighted ? 'on' : 'off'}`}>
                  <span>{formatKey(key)}</span>
                  <span>{value ? 'Yes' : 'No'}</span>
                </li>
              )
            })}
          </ul>
        </section>
      )}

      {data.why && (
        <section className="stage-section">
          <h4>Why</h4>
          <p className="stage-why">{data.why}</p>
        </section>
      )}

      {data.extraFields && (
        <section className="stage-section">
          <h4>Other Fields</h4>
          <JsonViewer value={data.extraFields} />
        </section>
      )}
    </div>
  )
}

const formatChannel = (value?: string) => {
  if (!value) return 'all'
  return value.replace(/_/g, ' ')
}

const formatDecimal = (value: number | null | undefined, digits = 2) => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

const formatInteger = (value: number | null | undefined) => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return Math.trunc(value).toString()
}

function SentimentResearchView({ data }: { data: SentimentResearchData }) {
  const hasScores = typeof data.desIdio === 'number' && Number.isFinite(data.desIdio)

  if (!hasScores) {
    const message = data.error || 'Sentiment unavailable.'
    return <div className="stage-placeholder">{message}</div>
  }

  const channelLabel = formatChannel(data.channel)
  const windowLabel = data.windowDays != null ? `${Math.trunc(data.windowDays)}d` : '—'

  return (
    <div className="stage-content">
      <section className="stage-section">
        <p className="stage-leading">
          DES ({channelLabel}, {windowLabel}) = {formatDecimal(data.desRaw)}, idio=
          {formatDecimal(data.desIdio)}, conf={formatDecimal(data.confidence)}, n=
          {formatInteger(data.nDeltas)}
        </p>
        <table className="stage-table stage-kv">
          <tbody>
            <tr>
              <th>DES Raw</th>
              <td>{formatDecimal(data.desRaw)}</td>
            </tr>
            <tr>
              <th>DES Sector</th>
              <td>{formatDecimal(data.desSector)}</td>
            </tr>
            <tr>
              <th>DES Idio</th>
              <td>{formatDecimal(data.desIdio)}</td>
            </tr>
            <tr>
              <th>Confidence</th>
              <td>{formatDecimal(data.confidence)}</td>
            </tr>
            <tr>
              <th>DES × Conf</th>
              <td>{formatDecimal(data.desAdj)}</td>
            </tr>
            <tr>
              <th>Channel</th>
              <td>{channelLabel}</td>
            </tr>
            <tr>
              <th>Window</th>
              <td>{windowLabel}</td>
            </tr>
            <tr>
              <th>Deltas</th>
              <td>{formatInteger(data.nDeltas)}</td>
            </tr>
          </tbody>
        </table>
        {data.error && <p className="muted small">Hub note: {data.error}</p>}
      </section>

      {data.narrative && (
        <section className="stage-section">
          <h4>Narrative</h4>
          <p>{data.narrative}</p>
        </section>
      )}

      {data.narrativeError && <p className="muted small">Narrative note: {data.narrativeError}</p>}

      {data.articleDelta && (
        <section className="stage-section">
          <h4>Article Delta</h4>
          <DeltaView delta={data.articleDelta} />
        </section>
      )}

      {data.deltas.length > 0 && (
        <section className="stage-section">
          <h4>Recent Conversation</h4>
          <ul className="stage-list">
            {data.deltas.map((delta, idx) => (
              <li key={`${delta.timestamp ?? idx}-${idx}`}>
                <DeltaView delta={delta} />
              </li>
            ))}
          </ul>
        </section>
      )}

      {data.runAt && <p className="muted small">Updated {data.runAt}</p>}
    </div>
  )
}

function DeltaView({ delta }: { delta: SentimentDelta }) {
  const meta: string[] = []
  if (delta.direction) meta.push(`dir: ${delta.direction}`)
  if (delta.impact) meta.push(`impact: ${delta.impact}`)
  if (delta.channel) meta.push(delta.channel)
  if (delta.categories && delta.categories.length > 0) {
    meta.push(delta.categories.join(', '))
  }

  return (
    <div className="stage-delta">
      {delta.timestamp && <div className="muted small">{delta.timestamp}</div>}
      {delta.summary && <div>{delta.summary}</div>}
      {delta.why && <div className="muted small">Why: {delta.why}</div>}
      {meta.length > 0 && <div className="muted small">{meta.join(' · ')}</div>}
      {delta.url && (
        <div className="muted small">
          <a href={delta.url} target="_blank" rel="noreferrer">
            Source
          </a>
        </div>
      )}
    </div>
  )
}

function ResearcherStageView({ data }: { data: ResearcherStageData }) {
  return (
    <div className="stage-content">
      {data.hypotheses.length > 0 && (
        <section className="stage-section">
          <h4>Hypotheses</h4>
          <ol className="stage-list stage-numbered">
            {data.hypotheses.map((hypothesis, idx) => (
              <li key={`${hypothesis.text}-${idx}`}>
                {hypothesis.type ? <strong>{hypothesis.type}: </strong> : null}
                {hypothesis.text}
              </li>
            ))}
          </ol>
        </section>
      )}

      {data.rationale && (
        <section className="stage-section">
          <h4>Rationale</h4>
          <p>{data.rationale}</p>
        </section>
      )}

      {data.stageWhy && (
        <section className="stage-section">
          <h4>Stage Why</h4>
          <p className="stage-why">{data.stageWhy}</p>
        </section>
      )}

      {data.planWhy && (
        <section className="stage-section">
          <h4>Plan Why</h4>
          <p className="stage-why">{data.planWhy}</p>
        </section>
      )}

      {data.steps.length > 0 && (
        <section className="stage-section">
          <h4>Plan Steps</h4>
          <table className="stage-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Tool</th>
                <th>Covers</th>
                <th>Tests</th>
                <th>Why</th>
                <th>Pass If</th>
                <th>Fail If</th>
              </tr>
            </thead>
            <tbody>
              {data.steps.map((step) => (
                <tr key={step.index}>
                  <td>{step.index}</td>
                  <td>{step.tool}</td>
                  <td>{step.covers.length > 0 ? step.covers.join(', ') : '—'}</td>
                  <td>{step.tests.length > 0 ? step.tests.join(', ') : '—'}</td>
                  <td>{step.why || '—'}</td>
                  <td>{step.passIf || '—'}</td>
                  <td>{step.failIf || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {(data.sessionId || data.log) && (
        <section className="stage-section">
          {data.sessionId && <p className="muted small">Session {data.sessionId}</p>}
          {data.log && (
            <details className="stage-log">
              <summary>View LLM log</summary>
              <pre>{data.log}</pre>
            </details>
          )}
        </section>
      )}

      {data.runAt && <p className="muted small">Updated {data.runAt}</p>}
    </div>
  )
}

function TechnicalResearchView({ data }: { data: TechnicalResearchStageData }) {
  return (
    <div className="stage-content">
      {data.summaryLines.length > 0 && (
        <section className="stage-section">
          <h4>Summary</h4>
          <ul className="stage-list">
            {data.summaryLines.map((line, idx) => (
              <li key={`${line}-${idx}`}>{line}</li>
            ))}
          </ul>
        </section>
      )}

      {data.insights.length > 0 && (
        <section className="stage-section">
          <h4>Insights</h4>
          <ul className="stage-list">
            {data.insights.map((entry, idx) => (
              <li key={`${entry.tool}-${idx}`}>
                <strong>{entry.tool}:</strong> {entry.text}
                {entry.status && <span className="muted small"> ({entry.status})</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {data.steps.length > 0 && (
        <section className="stage-section">
          <h4>Plan Steps</h4>
          <table className="stage-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Tool</th>
                <th>Covers</th>
                <th>Tests</th>
                <th>Why</th>
                <th>Pass If</th>
                <th>Fail If</th>
              </tr>
            </thead>
            <tbody>
              {data.steps.map((step) => (
                <tr key={`tech-step-${step.index}`}>
                  <td>{step.index}</td>
                  <td>{step.tool}</td>
                  <td>{step.covers.length > 0 ? step.covers.join(', ') : '—'}</td>
                  <td>{step.tests.length > 0 ? step.tests.join(', ') : '—'}</td>
                  <td>{step.why || '—'}</td>
                  <td>{step.passIf || '—'}</td>
                  <td>{step.failIf || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {data.results.length > 0 && (
        <section className="stage-section">
          <h4>Execution Results</h4>
          <table className="stage-table">
            <thead>
              <tr>
                <th>Tool</th>
                <th>Status</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((result, idx) => (
                <tr key={`tech-result-${result.tool}-${idx}`}>
                  <td>{result.tool}</td>
                  <td>{result.status}</td>
                  <td>
                    {result.error ? (
                      <span className="stage-error">{result.error}</span>
                    ) : (
                      <div className="stage-result-detail">
                        {result.why && <div>Why: {result.why}</div>}
                        {result.passIf && <div>Pass if: {result.passIf}</div>}
                        {result.failIf && <div>Fail if: {result.failIf}</div>}
                        {result.result && (
                          <details>
                            <summary>View output</summary>
                            <JsonViewer value={result.result} />
                          </details>
                        )}
                        {!result.why && !result.passIf && !result.failIf && !result.result && '—'}
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {data.error && <p className="muted small">Error: {data.error}</p>}
      {data.runAt && <p className="muted small">Updated {data.runAt}</p>}
    </div>
  )
}

function formatDirection(value: string): string {
  if (!value) return '—'
  return value.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function ConversationHubView({ data }: { data: ConversationHubData }) {
  const tickersDisplay = data.tickers.join(', ') || '—'
  const directionLabel = formatDirection(data.direction)
  const impactLabel = formatDirection(data.impact)
  const categoriesDisplay = data.categories.join(', ') || '—'
  const showReason = data.reason && data.reason.toLowerCase() !== 'ok'

  return (
    <div className="stage-content">
      {data.summary && (
        <section className="stage-section">
          <h4>Summary</h4>
          <p className="stage-why">{data.summary}</p>
        </section>
      )}
      <section className="stage-section">
        <h4>Details</h4>
        <table className="stage-table stage-kv">
          <tbody>
            <tr>
              <th>Tickers</th>
              <td>{tickersDisplay}</td>
            </tr>
            <tr>
              <th>Direction</th>
              <td>{directionLabel}</td>
            </tr>
            <tr>
              <th>Impact</th>
              <td>{impactLabel}</td>
            </tr>
            {data.channel && (
              <tr>
                <th>Channel</th>
                <td>{formatDirection(data.channel)}</td>
              </tr>
            )}
            <tr>
              <th>Categories</th>
              <td>{categoriesDisplay}</td>
            </tr>
            <tr>
              <th>Timestamp</th>
              <td>{data.timestamp || '—'}</td>
            </tr>
            <tr>
              <th>Source</th>
              <td>{data.source || '—'}</td>
            </tr>
            {data.url && (
              <tr>
                <th>URL</th>
                <td>
                  <a href={data.url} target="_blank" rel="noreferrer">
                    {data.url}
                  </a>
                </td>
              </tr>
            )}
            <tr>
              <th>Appended</th>
              <td>{data.appended}</td>
            </tr>
            <tr>
              <th>Skipped (old)</th>
              <td>{data.skippedOld}</td>
            </tr>
            {showReason && (
              <tr>
                <th>Status</th>
                <td>{formatDirection(data.reason)}</td>
              </tr>
            )}
            {data.ingestedAt && (
              <tr>
                <th>Ingested</th>
                <td>{data.ingestedAt}</td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
      {data.why && (
        <section className="stage-section">
          <h4>Why</h4>
          <p className="stage-why">{data.why}</p>
        </section>
      )}
    </div>
  )
}

function renderPrimitive(value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  return String(value)
}

function renderArray(value: unknown[]) {
  if (!value.length) {
    return <div className="muted">(empty)</div>
  }

  if (value.every((item) => isPlainObject(item))) {
    const columns = Array.from(
      value.reduce<Set<string>>((set, item) => {
        Object.keys(item as Record<string, unknown>).forEach((key) => set.add(key))
        return set
      }, new Set())
    )

    return (
      <table className="stage-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{formatKey(col)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {value.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col}>{renderPrimitive((row as Record<string, unknown>)[col])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <ul className="stage-list">
      {value.map((item, idx) => (
        <li key={idx}>{renderPrimitive(item)}</li>
      ))}
    </ul>
  )
}

function renderObject(value: Record<string, unknown>) {
  const entries = Object.entries(value)
  if (!entries.length) {
    return <div className="muted">(empty)</div>
  }

  return (
    <table className="stage-table stage-kv">
      <tbody>
        {entries.map(([key, val]) => (
          <tr key={key}>
            <th>{formatKey(key)}</th>
            <td>{renderValue(val)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function renderValue(value: unknown): any {
  if (Array.isArray(value)) {
    return renderArray(value)
  }
  if (isPlainObject(value)) {
    return renderObject(value)
  }
  return renderPrimitive(value)
}

function GenericStageView({ data }: { data: unknown }) {
  if (Array.isArray(data)) {
    return <div className="stage-content">{renderArray(data)}</div>
  }

  if (!isPlainObject(data)) {
    return <JsonViewer value={data} />
  }

  const entries = Object.entries(data)
  return (
    <div className="stage-content">
      {entries.map(([key, value]) => (
        <section key={key} className="stage-section">
          <h4>{formatKey(key)}</h4>
          {renderValue(value)}
        </section>
      ))}
    </div>
  )
}

function hasRenderableContent(value: unknown): boolean {
  if (value == null) return false
  if (typeof value === 'string') return value.trim().length > 0
  if (Array.isArray(value)) return value.length > 0
  if (isPlainObject(value)) return Object.keys(value).length > 0
  return true
}

export default function StageViewer({
  name,
  payload,
  placeholder = 'ready to analyse',
  ticker,
}: StageViewerProps) {
  if (name === 'chairman') {
    const chairmanData = parseChairmanStageData(payload)
    if (chairmanData) {
      return <ChairmanStageView data={chairmanData} />
    }
  }

  if (name === 'conversation_hub') {
    const hubData = parseConversationHubData(payload)
    if (hubData) {
      return <ConversationHubView data={hubData} />
    }
  }

  if (name === 'researcher') {
    const researcherData = parseResearcherStageData(payload, ticker)
    if (researcherData) {
      return <ResearcherStageView data={researcherData} />
    }
  }

  if (name === 'technical_research') {
    const technicalData = parseTechnicalResearchData(payload, ticker)
    if (technicalData) {
      return <TechnicalResearchView data={technicalData} />
    }
  }

  if (name === 'sentiment_research') {
    const sentimentData = parseSentimentResearchData(payload, ticker)
    if (sentimentData) {
      return <SentimentResearchView data={sentimentData} />
    }
  }

  if (name === 'summariser') {
    const summariserData = parseSummariserData(payload)
    if (summariserData) {
      return <SummariserView data={summariserData} />
    }
  }

  const parsed = tryParse(payload)
  const data = parsed ?? payload

  if (!hasRenderableContent(data)) {
    return <div className="stage-placeholder">{placeholder}</div>
  }

  return <GenericStageView data={data} />
}
