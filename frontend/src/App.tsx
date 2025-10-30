import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  listPosts,
  getPost,
  runStage,
  runResearch,
  clearAnalysis,
  eraseAllCouncilAnalysis,
  startRefreshSummaries,
  getRefreshJob,
  getActiveRefreshJob,
  stopRefreshJob,
  startCouncilAnalysis,
  getCouncilJob,
  getActiveCouncilJob,
  stopCouncilJob,
} from './api'
import type { CouncilJob, ListPostsParams, RefreshJob } from './api'
import type { ListResponse, PostDetail, ResearchPayload, ResearchTickerPayload } from './types'
import PostList from './components/PostList'
import PostDetailView from './components/PostDetail'
import TraderBar from './components/TraderBar'
import { parseChairmanStageData, parseSummariserData } from './components/StageViewer'
import { formatDateKey, parseDateKey } from './utils/dates'
import './styles.css'

const COUNCIL_STAGE_LABELS: Record<string, string> = {
  entity: 'Entity Council',
  claims: 'Claims Council',
  context: 'Context Council',
  verifier: 'Verifier Council',
  for: 'Bull Council',
  against: 'Bear Council',
  direction: 'Direction Council',
  researcher: 'Researcher',
  chairman: 'Chairman Verdict',
}

const normalizeTicker = (value?: string | null): string => {
  if (typeof value !== 'string') return ''
  return value.trim().toUpperCase()
}

const parseInterestScore = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

const normalizeOracleStatusValue = (value: unknown): OracleStatus => {
  const allowed: OracleStatus[] = [
    'offline',
    'connecting',
    'warmup',
    'processing',
    'idle',
    'error',
    'unauthorized',
  ]
  if (typeof value === 'string' && allowed.includes(value as OracleStatus)) {
    return value as OracleStatus
  }
  return 'offline'
}

const STORAGE_KEYS = {
  selectedDate: 'wos.selectedDate',
  selectedPost: 'wos.selectedPostId',
  postsPage: 'wos.postsPage',
  interestMin: 'wos.interestMin',
  councilJob: 'wos.councilJobId',
  councilPrimed: 'wos.councilPrimedMissing',
  analyseNewActive: 'wos.analyseNewActive',
  analyseNewThreshold: 'wos.analyseNewThreshold',
  oracleOnline: 'wos.oracleOnline',
  oracleBaseUrl: 'wos.oracleBaseUrl',
} as const

type CouncilProgressState = {
  stages: string[]
  currentStageIndex: number
  completed: number
}

type OracleStatus =
  | 'offline'
  | 'connecting'
  | 'warmup'
  | 'processing'
  | 'idle'
  | 'error'
  | 'unauthorized'

export default function App() {
  const [q, setQ] = useState('')
  const [platform, setPlatform] = useState('')
  const [source, setSource] = useState('')
  const [page, setPage] = useState<number>(() => {
    if (typeof window === 'undefined') return 1
    const raw = window.localStorage.getItem(STORAGE_KEYS.postsPage)
    const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 1
  })
  const [pageSize, setPageSize] = useState(20)
  const [interestFilter, setInterestFilter] = useState<number>(() => {
    if (typeof window === 'undefined') return 0
    const raw = window.localStorage.getItem(STORAGE_KEYS.interestMin)
    const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN
    return Number.isFinite(parsed) && parsed >= 0 && parsed <= 100 ? parsed : 0
  })

  const [oracleOnline, setOracleOnline] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(STORAGE_KEYS.oracleOnline) === '1'
  })
  const [oracleBaseUrl, setOracleBaseUrl] = useState<string>(() => {
    if (typeof window === 'undefined') return ''
    return window.localStorage.getItem(STORAGE_KEYS.oracleBaseUrl) ?? ''
  })
  const [oracleStatus, setOracleStatus] = useState<OracleStatus>('offline')
  const [oracleBanner, setOracleBanner] = useState<string | null>(null)
  const [oracleIdleInfo, setOracleIdleInfo] = useState<{ pollSeconds?: number | null; idleSince?: number | null } | null>(null)
  const [oracleCursor, setOracleCursor] = useState<{ platform?: string | null; post_id?: string | null; scraped_at?: string | null } | null>(null)

  const [selectedDate, setSelectedDate] = useState<Date | null>(() => {
    if (typeof window === 'undefined') return null
    return parseDateKey(window.localStorage.getItem(STORAGE_KEYS.selectedDate))
  })

  const [list, setList] = useState<ListResponse | null>(null)
  const [loadingList, setLoadingList] = useState(false)

  const [pickedId, setPickedId] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    const stored = window.localStorage.getItem(STORAGE_KEYS.selectedPost)
    return stored && stored.trim() ? stored : null
  })
  const [detail, setDetail] = useState<PostDetail | null>(null)
  const [log, setLog] = useState<string>('')
  const [logSnippet, setLogSnippet] = useState<string>('')
  const [logExpanded, setLogExpanded] = useState(false)
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  const [busy, setBusy] = useState(false)

  const [articleTickers, setArticleTickers] = useState<string[]>([])
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null)

  const [refreshJobId, setRefreshJobId] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    return window.localStorage.getItem('refreshJobId')
  })
  const [refreshJob, setRefreshJob] = useState<RefreshJob | null>(null)
  const [jobBusy, setJobBusy] = useState(false)
  const lastListRefresh = useRef<number | null>(null)
  const lastChairmanVerdict = useRef<{ postId: string | null; signature: string | null }>({
    postId: null,
    signature: null,
  })
  const [councilProgress, setCouncilProgress] = useState<CouncilProgressState | null>(null)
  const [councilJobId, setCouncilJobId] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null
    const stored = window.localStorage.getItem(STORAGE_KEYS.councilJob)
    return stored && stored.trim() ? stored : null
  })
  const [councilJob, setCouncilJob] = useState<CouncilJob | null>(null)
  const [councilJobBusy, setCouncilJobBusy] = useState(false)
  const [councilLogEntry, setCouncilLogEntry] = useState('')
  const [councilLog, setCouncilLog] = useState<string>('')
  const [councilCopyStatus, setCouncilCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  const [councilPrimed, setCouncilPrimed] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(STORAGE_KEYS.councilPrimed) === '1'
  })
  const [analyseNewActive, setAnalyseNewActive] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(STORAGE_KEYS.analyseNewActive) === '1'
  })
  const [analyseNewThreshold, setAnalyseNewThreshold] = useState<number | null>(() => {
    if (typeof window === 'undefined') return null
    const stored = window.localStorage.getItem(STORAGE_KEYS.analyseNewThreshold)
    if (!stored) return null
    const parsed = Number.parseFloat(stored)
    return Number.isFinite(parsed) ? parsed : null
  })
  const [analyseNewCouncilStarted, setAnalyseNewCouncilStarted] = useState(false)
  const analyseNewPendingRef = useRef<Set<string>>(new Set())
  const analyseNewInFlightRef = useRef<Set<string>>(new Set())
  const analyseNewProcessedRef = useRef<Set<string>>(new Set())
  const [analyseNewPendingIds, setAnalyseNewPendingIds] = useState<string[]>([])
  const [analyseNewInFlightIds, setAnalyseNewInFlightIds] = useState<string[]>([])
  const analyseNewStartInFlightRef = useRef(false)
  const [analyseNewRefreshComplete, setAnalyseNewRefreshComplete] = useState(false)
  const [analyseNewHadEligible, setAnalyseNewHadEligible] = useState(false)

  const syncAnalyseNewPending = useCallback(() => {
    setAnalyseNewPendingIds(Array.from(analyseNewPendingRef.current))
  }, [])

  const syncAnalyseNewInFlight = useCallback(() => {
    setAnalyseNewInFlightIds(Array.from(analyseNewInFlightRef.current))
  }, [])

  const clearAnalyseNewQueues = useCallback(() => {
    analyseNewPendingRef.current.clear()
    analyseNewInFlightRef.current.clear()
    analyseNewProcessedRef.current.clear()
    analyseNewStartInFlightRef.current = false
    syncAnalyseNewPending()
    syncAnalyseNewInFlight()
  }, [syncAnalyseNewPending, syncAnalyseNewInFlight])

  const resetAnalyseNewState = useCallback(() => {
    clearAnalyseNewQueues()
    setAnalyseNewActive(false)
    setAnalyseNewThreshold(null)
    setAnalyseNewCouncilStarted(false)
    setAnalyseNewRefreshComplete(false)
    setAnalyseNewHadEligible(false)
    setOracleIdleInfo(null)
    setOracleCursor(null)
    setOracleBanner((prev) =>
      prev === 'Oracle auth invalid. Set WOS_ORACLE_USER and WOS_ORACLE_PASS.' ? prev : null,
    )
    setOracleStatus((prev) => (prev === 'unauthorized' ? prev : 'offline'))
  }, [clearAnalyseNewQueues])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedDate) {
      window.localStorage.removeItem(STORAGE_KEYS.selectedDate)
      return
    }
    window.localStorage.setItem(STORAGE_KEYS.selectedDate, formatDateKey(selectedDate))
  }, [selectedDate])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(STORAGE_KEYS.postsPage, String(page))
  }, [page])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(STORAGE_KEYS.interestMin, String(interestFilter))
  }, [interestFilter])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (councilPrimed) {
      window.localStorage.setItem(STORAGE_KEYS.councilPrimed, '1')
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.councilPrimed)
    }
  }, [councilPrimed])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!pickedId) {
      window.localStorage.removeItem(STORAGE_KEYS.selectedPost)
      return
    }
    window.localStorage.setItem(STORAGE_KEYS.selectedPost, pickedId)
  }, [pickedId])

  const batchTotal = refreshJob?.total ?? 0
  const batchDone = refreshJob?.done ?? 0
  const progressPct = batchTotal ? Math.round((batchDone / batchTotal) * 100) : 0
  const batchMsg = [refreshJob?.phase, refreshJob?.current].filter(Boolean).join(' – ') || refreshJob?.message || ''
  const jobActive = refreshJob ? refreshJob.status === 'queued' || refreshJob.status === 'running' : Boolean(refreshJobId)
  const working = busy || jobBusy
  const councilTotal = councilProgress?.stages.length ?? 0
  const councilCompleted = councilProgress?.completed ?? 0
  const councilDone = councilTotal ? Math.min(councilCompleted, councilTotal) : councilCompleted
  const councilPercent = councilTotal ? Math.round((councilDone / councilTotal) * 100) : 0
  const councilStageKey =
    councilProgress && councilProgress.currentStageIndex < councilProgress.stages.length
      ? councilProgress.stages[councilProgress.currentStageIndex]
      : null
  const councilStageLabel = councilStageKey
    ? COUNCIL_STAGE_LABELS[councilStageKey] ?? councilStageKey
    : councilTotal > 0 && councilDone >= councilTotal
      ? 'Finalising'
      : ''
  const councilJobTotal = councilJob?.total ?? 0
  const councilJobDone = councilJob?.done ?? 0
  const councilJobPercent = councilJobTotal ? Math.round((councilJobDone / councilJobTotal) * 100) : 0
  const councilJobRemaining = councilJob?.remaining ?? Math.max(councilJobTotal - councilJobDone, 0)
  const councilJobMessage = (councilJob?.message ?? '').trim()
  const councilJobActive = councilJob
    ? councilJob.status === 'queued' || councilJob.status === 'running' || councilJob.status === 'cancelling'
    : Boolean(councilJobId)
  const interestSliderDisabled = councilJobActive || analyseNewActive
  const etaBaseSeconds =
    typeof councilJob?.current_eta_seconds === 'number' && councilJob.current_eta_seconds > 0
      ? councilJob.current_eta_seconds
      : null
  const etaStartedAt =
    typeof councilJob?.current_started_at === 'number' && councilJob.current_started_at > 0
      ? councilJob.current_started_at
      : null
  const nowSeconds = Date.now() / 1000
  const hasEtaTiming = etaBaseSeconds != null && etaStartedAt != null
  const etaElapsedSeconds = hasEtaTiming ? Math.max(nowSeconds - etaStartedAt, 0) : 0
  const etaRemainingSeconds = hasEtaTiming && etaBaseSeconds != null
    ? Math.max(etaBaseSeconds - etaElapsedSeconds, 0)
    : null
  const etaPercent = hasEtaTiming && etaBaseSeconds != null && etaBaseSeconds > 0
    ? Math.max(0, Math.min(100, (etaElapsedSeconds / etaBaseSeconds) * 100))
    : 0
  const etaDisplayText = councilJobActive ? formatEta(etaRemainingSeconds) : 'idle'
  const councilLastLine = councilLog
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .pop() || ''
  const councilDisplayLog = (councilLogEntry || councilJobMessage || councilLastLine).trim()
  const panelDisabled = busy || jobBusy || councilJobBusy || analyseNewActive
  const oracleControlsDisabled = working || jobActive || councilJobActive || councilJobBusy || analyseNewActive
  const oracleStatusVariant = useMemo(() => {
    switch (oracleStatus) {
      case 'processing':
        return 'positive'
      case 'idle':
      case 'warmup':
      case 'connecting':
        return 'warning'
      case 'unauthorized':
      case 'error':
        return 'negative'
      default:
        return 'neutral'
    }
  }, [oracleStatus])
  const oracleStatusText = useMemo(() => {
    const cursorBits: string[] = []
    if (oracleCursor) {
      const parts: string[] = []
      if (oracleCursor.scraped_at) parts.push(String(oracleCursor.scraped_at))
      const idPart = [oracleCursor.platform, oracleCursor.post_id].filter(Boolean).join(':')
      if (idPart) parts.push(idPart)
      if (parts.length > 0) {
        cursorBits.push(`last ${parts.join(' ')}`.trim())
      }
    }
    switch (oracleStatus) {
      case 'connecting':
        return 'Oracle: Connecting…'
      case 'warmup':
        return 'Oracle: Warm-up scan…'
      case 'processing':
        return cursorBits.length > 0
          ? `Oracle: Processing (${cursorBits.join(' · ')})`
          : 'Oracle: Processing…'
      case 'idle': {
        const pollText =
          oracleIdleInfo?.pollSeconds != null
            ? `${oracleIdleInfo.pollSeconds.toFixed(1)}s`
            : '?s'
        let idleSegment = ''
        if (oracleIdleInfo?.idleSince != null) {
          const elapsed = Math.max(0, Date.now() / 1000 - oracleIdleInfo.idleSince)
          idleSegment = ` • Idle for ${formatEta(elapsed)}`
        }
        const cursorSegment = cursorBits.length > 0 ? ` • ${cursorBits.join(' · ')}` : ''
        return `Oracle: Idle (poll ${pollText})${idleSegment}${cursorSegment}`
      }
      case 'unauthorized':
        return 'Oracle: Unauthorized'
      case 'error':
        return 'Oracle: Error'
      default:
        return 'Oracle: Offline'
    }
  }, [oracleStatus, oracleIdleInfo, oracleCursor])

  const deriveLogSnippet = (value: string): string => {
    const trimmed = value.trim()
    if (!trimmed) return ''
    const lines = trimmed
      .split(/\r?\n+/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
    if (lines.length === 0) return ''
    const contentLines = lines.filter((line) => !/^ran:/i.test(line))
    const chosen = contentLines.length > 0 ? contentLines : lines
    const snippet = chosen.slice(0, 2).join(' · ')
    if (snippet.length <= 200) return snippet
    return `${snippet.slice(0, 197)}…`
  }

  function formatEta(seconds: number | null): string {
    if (seconds == null) return 'estimating…'
    if (!Number.isFinite(seconds)) return 'estimating…'
    if (seconds <= 0.5) return '<1sec'
    const total = Math.round(seconds)
    const hours = Math.floor(total / 3600)
    const minutes = Math.floor((total % 3600) / 60)
    const secs = total % 60
    const parts: string[] = []
    if (hours > 0) {
      parts.push(`${hours}hr${hours === 1 ? '' : 's'}`)
      if (minutes > 0) {
        parts.push(`${minutes}min`)
      }
      if (minutes === 0 && secs > 0) {
        parts.push(`${secs}sec${secs === 1 ? '' : 's'}`)
      }
    } else {
      if (minutes > 0) {
        parts.push(`${minutes}min`)
      }
      if (minutes < 5 && secs > 0) {
        parts.push(`${secs}sec${secs === 1 ? '' : 's'}`)
      } else if (minutes === 0) {
        parts.push(`${secs}sec${secs === 1 ? '' : 's'}`)
      }
    }
    if (parts.length === 0) {
      parts.push(`${secs}sec${secs === 1 ? '' : 's'}`)
    }
    return parts.join(' ')
  }

  const updateLog = (value: string | ((prev: string) => string)) => {
    if (typeof value === 'function') {
      setLog((prev) => {
        const next = value(prev)
        setLogSnippet(deriveLogSnippet(next))
        return next
      })
      return
    }
    setLog(value)
    setLogSnippet(deriveLogSnippet(value))
  }

  useEffect(() => {
    if (copyStatus === 'idle') return undefined
    if (typeof window === 'undefined') return undefined
    const timer = window.setTimeout(() => setCopyStatus('idle'), 2000)
    return () => window.clearTimeout(timer)
  }, [copyStatus])

  useEffect(() => {
    if (councilCopyStatus === 'idle') return undefined
    if (typeof window === 'undefined') return undefined
    const timer = window.setTimeout(() => setCouncilCopyStatus('idle'), 2000)
    return () => window.clearTimeout(timer)
  }, [councilCopyStatus])

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    if (analyseNewActive) {
      window.localStorage.setItem(STORAGE_KEYS.analyseNewActive, '1')
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.analyseNewActive)
    }
    return undefined
  }, [analyseNewActive])

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    if (analyseNewThreshold != null && Number.isFinite(analyseNewThreshold)) {
      window.localStorage.setItem(STORAGE_KEYS.analyseNewThreshold, String(analyseNewThreshold))
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.analyseNewThreshold)
    }
    return undefined
  }, [analyseNewThreshold])

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    if (oracleOnline) {
      window.localStorage.setItem(STORAGE_KEYS.oracleOnline, '1')
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.oracleOnline)
    }
    return undefined
  }, [oracleOnline])

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    const trimmed = oracleBaseUrl.trim()
    if (trimmed) {
      window.localStorage.setItem(STORAGE_KEYS.oracleBaseUrl, trimmed)
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.oracleBaseUrl)
    }
    return undefined
  }, [oracleBaseUrl])

  useEffect(() => {
    if (!analyseNewActive) {
      setAnalyseNewCouncilStarted(false)
    }
  }, [analyseNewActive])

  const appendLog = (entry: string) => {
    const trimmedEntry = entry.trim()
    if (!trimmedEntry) return
    updateLog((prev) => {
      const prefix = prev.trim()
      return prefix ? `${prefix}\n\n${trimmedEntry}` : trimmedEntry
    })
  }

  const formatStageLabel = (key: string): string => COUNCIL_STAGE_LABELS[key] ?? key

  const logRunOutput = (
    stages: string[],
    stdout: string,
    stderr: string,
    options?: { append?: boolean }
  ) => {
    const trimmedStdout = stdout.trim()
    const trimmedStderr = stderr.trim()
    const labeledStages = stages.map((stage) => formatStageLabel(stage))
    const parts = [
      `Ran: ${labeledStages.join(', ')}`,
      trimmedStderr && `stderr:\n${trimmedStderr}`,
      trimmedStdout && `stdout:\n${trimmedStdout}`,
    ].filter(Boolean) as string[]
    const entry = parts.join('\n\n')
    if (options?.append) {
      appendLog(entry)
      return
    }
    updateLog(entry)
  }

  const formatResearchLog = (payload: ResearchPayload | null | undefined): string | null => {
    if (!payload) return null

    const tickersMap = payload.tickers ?? {}
    const normalized = new Map<string, ResearchTickerPayload>()
    Object.entries(tickersMap).forEach(([key, value]) => {
      if (value && typeof value === 'object') {
        normalized.set(normalizeTicker(key), value as ResearchTickerPayload)
      }
    })

    const ordered = (payload.ordered_tickers ?? []).map((item) => normalizeTicker(item))
    const sequence = ordered.length > 0 ? ordered : Array.from(normalized.keys())

    const tickerSummaries = sequence
      .map((symbol) => {
        const entry = normalized.get(symbol)
        if (!entry) return null
        const lines: string[] = []
        if (typeof entry.summary_text === 'string' && entry.summary_text.trim()) {
          lines.push(entry.summary_text.trim())
        } else if (typeof entry.rationale === 'string' && entry.rationale.trim()) {
          lines.push(entry.rationale.trim())
        }

        const technicalSummary = Array.isArray(entry.technical?.summary_lines)
          ? entry.technical.summary_lines
              .map((line) => (typeof line === 'string' ? line.trim() : ''))
              .filter((line) => line.length > 0)
              .join(' | ')
          : ''
        if (technicalSummary) {
          lines.push(`Technical: ${technicalSummary}`)
        }

        const sentimentRecord =
          entry.sentiment && typeof entry.sentiment === 'object'
            ? (entry.sentiment as Record<string, unknown>)
            : null
        if (sentimentRecord) {
          const deltaCandidate =
            (sentimentRecord['article_delta'] as Record<string, unknown> | undefined) ??
            (sentimentRecord['articleDelta'] as Record<string, unknown> | undefined) ??
            (sentimentRecord['latest_delta'] as Record<string, unknown> | undefined) ??
            (sentimentRecord['latestDelta'] as Record<string, unknown> | undefined) ??
            null
          if (deltaCandidate) {
            const delta = deltaCandidate as Record<string, unknown>
            const deltaSummary =
              typeof delta['sum'] === 'string'
                ? (delta['sum'] as string).trim()
                : typeof delta['summary'] === 'string'
                  ? (delta['summary'] as string).trim()
                  : ''
            const deltaWhy = typeof delta['why'] === 'string' ? (delta['why'] as string).trim() : ''
            const deltaDir = typeof delta['dir'] === 'string' ? (delta['dir'] as string).trim() : ''
            const deltaImpact =
              typeof delta['impact'] === 'string' ? (delta['impact'] as string).trim() : ''
            const sentimentParts: string[] = []
            if (deltaSummary) sentimentParts.push(deltaSummary)
            if (deltaWhy) sentimentParts.push(`why: ${deltaWhy}`)
            const dirImpact = [deltaDir, deltaImpact].filter((part) => part.length > 0)
            if (dirImpact.length > 0) sentimentParts.push(dirImpact.join('/'))
            if (sentimentParts.length > 0) {
              lines.push(`Sentiment: ${sentimentParts.join(' · ')}`)
            }
          }
        }

        if (lines.length === 0) return null
        return `${symbol}:\n${lines.join('\n')}`
      })
      .filter((entry): entry is string => !!entry)

    const updatedAt = payload.updated_at || new Date().toISOString()
    const header = `Research updated ${updatedAt}`
    if (tickerSummaries.length === 0) return header
    return [header, ...tickerSummaries].join('\n\n')
  }

  const queryKey = useMemo<ListPostsParams>(() => {
    const base: ListPostsParams = {
      q,
      platform,
      source,
      page,
      page_size: pageSize,
    }

    if (interestFilter > 0) {
      base.interest_min = interestFilter
    }

    if (!selectedDate) {
      return base
    }

    const start = new Date(selectedDate)
    start.setHours(0, 0, 0, 0)
    const end = new Date(selectedDate)
    end.setHours(23, 59, 59, 999)

    return {
      ...base,
      date_from: start.toISOString(),
      date_to: end.toISOString(),
    }
  }, [q, platform, source, page, pageSize, selectedDate, interestFilter])

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoadingList(true)
      try {
        const res = await listPosts(queryKey)
        if (!cancelled) setList(res)
      } catch (e: any) {
        if (!cancelled) updateLog(`List error: ${e.message}`)
      } finally {
        if (!cancelled) setLoadingList(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [
    queryKey.q,
    queryKey.platform,
    queryKey.source,
    queryKey.page,
    queryKey.page_size,
    queryKey.date_from,
    queryKey.date_to,
    queryKey.interest_min,
  ])

  useEffect(() => {
    if (!councilJobId) {
      setCouncilJob(null)
      setCouncilJobBusy(false)
      return
    }

    let cancelled = false
    const activeJobId = councilJobId!
    const currentPickedId = pickedId
    const currentQuery: ListPostsParams = { ...queryKey }

    async function poll() {
      try {
        const job = await getCouncilJob(activeJobId)
        if (cancelled) return
        setCouncilJob(job)
        const active = job.status === 'queued' || job.status === 'running' || job.status === 'cancelling'
        setCouncilJobBusy(active)
        const tail = Array.isArray(job.log_tail) ? job.log_tail : []
        const normalizedTail = tail.map((entry) => {
          if (typeof entry === 'string') return entry
          if (entry == null) return ''
          return String(entry)
        })
        const joinedTail = normalizedTail.join('\n')
        const lastEntry = normalizedTail.length > 0 ? normalizedTail[normalizedTail.length - 1] : ''
        setCouncilLogEntry(lastEntry.trim() ? lastEntry : '')
        setCouncilLog(joinedTail)
        if (!active) {
          if (job.status === 'done') {
            try {
              const [detailRes, listRes] = await Promise.all([
                currentPickedId ? getPost(currentPickedId) : Promise.resolve(null),
                listPosts(currentQuery),
              ])
              if (!cancelled) {
                if (detailRes) setDetail(detailRes)
                setList(listRes)
              }
            } catch (err: any) {
              if (!cancelled) {
                updateLog((prev) => prev || `Council list refresh error: ${err.message}`)
              }
            }
          }
          if (!cancelled) {
            if (analyseNewInFlightRef.current.size > 0) {
              analyseNewInFlightRef.current.forEach((id) => {
                analyseNewProcessedRef.current.add(id)
              })
            }
            analyseNewInFlightRef.current.clear()
            syncAnalyseNewInFlight()
            analyseNewStartInFlightRef.current = false

            if (analyseNewCouncilStarted) {
              if (job.status === 'done') {
                setAnalyseNewCouncilStarted(false)
              } else if (analyseNewActive) {
                const statusMessage =
                  job.status === 'cancelled'
                    ? 'Analyse New Articles cancelled.'
                    : job.status === 'error'
                      ? 'Analyse New Articles encountered an error.'
                      : ''
                if (statusMessage) {
                  updateLog((prev) => prev || statusMessage)
                }
                resetAnalyseNewState()
              }
            }

            setCouncilJobId(null)
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          updateLog((prev) => prev || `Council job error: ${err.message}`)
          setCouncilJobBusy(false)
          setCouncilJobId(null)
        }
      }
    }

    poll()
    const timer = window.setInterval(poll, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [
    councilJobId,
    queryKey.q,
    queryKey.platform,
    queryKey.source,
    queryKey.page,
    queryKey.page_size,
    queryKey.date_from,
    queryKey.date_to,
    queryKey.interest_min,
    pickedId,
    analyseNewActive,
    analyseNewCouncilStarted,
    resetAnalyseNewState,
    syncAnalyseNewInFlight,
  ])

  useEffect(() => {
    if (!pickedId) return setDetail(null)
    let cancelled = false
    const id = pickedId!
    async function load() {
      try {
        const d = await getPost(id)
        if (!cancelled) setDetail(d)
      } catch (e: any) {
        if (!cancelled) updateLog(`Detail error: ${e.message}`)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [pickedId])

  useEffect(() => {
    if (!detail) {
      setArticleTickers([])
      setSelectedTicker(null)
      return
    }

    const summariserPayload = detail.stages?.summariser
    const summariser = parseSummariserData(summariserPayload)
    const summariserTickers =
      summariser?.assets
        ?.map((asset) => (typeof asset.ticker === 'string' ? normalizeTicker(asset.ticker) : ''))
        .filter((ticker): ticker is string => Boolean(ticker)) ?? []

    const extras = detail.extras && typeof detail.extras === 'object' ? detail.extras : null
    const researchTickers: string[] = []
    if (extras && extras.research && typeof extras.research === 'object') {
      const researchRecord = extras.research as Record<string, unknown>
      const orderedField =
        (researchRecord.ordered_tickers as unknown) ?? (researchRecord.ordered as unknown)
      if (Array.isArray(orderedField)) {
        orderedField.forEach((value) => {
          const ticker = normalizeTicker(typeof value === 'string' ? value : String(value ?? ''))
          if (ticker) researchTickers.push(ticker)
        })
      }
      const mapField = researchRecord.tickers
      if (mapField && typeof mapField === 'object') {
        Object.keys(mapField as Record<string, unknown>)
          .map((key) => normalizeTicker(key))
          .forEach((ticker) => {
            if (ticker) researchTickers.push(ticker)
          })
      }
    }

    const combined = Array.from(new Set([...summariserTickers, ...researchTickers]))
    setArticleTickers(combined)
    setSelectedTicker((prev) => {
      if (prev && combined.includes(prev)) return prev
      return combined.length > 0 ? combined[0] : null
    })
  }, [detail])

  useEffect(() => {
    const postId = detail?.post?.post_id ?? null
    if (!postId) {
      lastChairmanVerdict.current = { postId: null, signature: null }
      return
    }

    const chairmanData = parseChairmanStageData(detail?.stages?.chairman)
    const plainEnglish = chairmanData?.plainEnglish?.trim() ?? ''
    const normalizedPlain = plainEnglish.replace(/\s+/g, ' ').trim()
    const verdictSignature = normalizedPlain
      ? [normalizedPlain, chairmanData?.direction ?? '', chairmanData?.timeframe ?? '', chairmanData?.conviction ?? '']
          .map((part) => (part == null ? '' : String(part)))
          .join('|')
      : null

    const prev = lastChairmanVerdict.current
    if (!verdictSignature) {
      if (prev.postId !== postId || prev.signature !== null) {
        lastChairmanVerdict.current = { postId, signature: null }
      }
      return
    }

    const listItem = list?.items?.find((item) => item.post_id === postId)
    const listVerdict = (listItem?.chairman_plain_english ?? '').replace(/\s+/g, ' ').trim()
    const alreadySynced = listVerdict.length > 0 && listVerdict === normalizedPlain

    if (alreadySynced) {
      lastChairmanVerdict.current = { postId, signature: verdictSignature }
      return
    }

    if (prev.postId === postId && prev.signature === verdictSignature) {
      return
    }

    lastChairmanVerdict.current = { postId, signature: verdictSignature }
    let cancelled = false

    const refreshList = async () => {
      try {
        const res = await listPosts(queryKey)
        if (!cancelled) {
          setList(res)
        }
      } catch (err: any) {
        if (!cancelled) {
          setLog((prevLog) => (prevLog ? prevLog : `List refresh error: ${err?.message ?? err}`))
        }
      }
    }

    void refreshList()

    return () => {
      cancelled = true
    }
  }, [detail?.post?.post_id, detail?.stages?.chairman, list, queryKey, setList, setLog])

  const researchExtras = useMemo<ResearchPayload | null>(() => {
    if (!detail) return null
    const extras = detail.extras
    if (!extras || typeof extras !== 'object') return null
    const research = (extras as Record<string, unknown>).research
    if (!research || typeof research !== 'object') return null
    return research as ResearchPayload
  }, [detail])

  const researchSelection = useMemo(() => {
    if (!researchExtras) return null
    const tickersMap = researchExtras.tickers ?? {}
    const entries = Object.entries(tickersMap).filter(([, value]) =>
      value && typeof value === 'object',
    )
    if (entries.length === 0) return null

    const normalizedMap = new Map<string, ResearchTickerPayload>()
    entries.forEach(([key, value]) => {
      normalizedMap.set(normalizeTicker(key), value as ResearchTickerPayload)
    })

    const orderedFallback = (researchExtras as unknown as { ordered?: unknown }).ordered
    const orderedField = Array.isArray(researchExtras.ordered_tickers)
      ? researchExtras.ordered_tickers
      : Array.isArray(orderedFallback)
        ? (orderedFallback as unknown[])
        : []

    const orderedKeys = orderedField
      .map((item) => normalizeTicker(typeof item === 'string' ? item : String(item ?? '')))
      .filter((key) => key.length > 0)

    const desired = normalizeTicker(selectedTicker)
    const selectedKey =
      (desired && normalizedMap.has(desired) ? desired : undefined) ||
      orderedKeys.find((key) => normalizedMap.has(key)) ||
      [...normalizedMap.keys()][0]

    if (!selectedKey) return null

    return { ticker: selectedKey, data: normalizedMap.get(selectedKey)! }
  }, [researchExtras, selectedTicker])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (refreshJobId) {
      window.localStorage.setItem('refreshJobId', refreshJobId)
    } else {
      window.localStorage.removeItem('refreshJobId')
    }
  }, [refreshJobId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (councilJobId) {
      window.localStorage.setItem(STORAGE_KEYS.councilJob, councilJobId)
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.councilJob)
    }
  }, [councilJobId])

  useEffect(() => {
    let cancelled = false
    async function loadActive() {
      try {
        const active = await getActiveRefreshJob()
        if (!cancelled && active) {
          setRefreshJob(active)
          setRefreshJobId(active.id)
          setJobBusy(active.status === 'queued' || active.status === 'running')
          if (active.log_tail?.length) updateLog(active.log_tail.join('\n'))
        }
      } catch (e: any) {
        if (!cancelled) updateLog((prev) => prev || `Active job error: ${e.message}`)
      }
    }
    if (!refreshJobId) {
      loadActive()
    }
    return () => {
      cancelled = true
    }
  }, [refreshJobId])

  useEffect(() => {
    let cancelled = false
    async function loadCouncilActive() {
      try {
        const active = await getActiveCouncilJob()
        if (!cancelled && active) {
          setCouncilJobId(active.job_id)
          setCouncilJob((prev) =>
            prev && prev.id === active.job_id
              ? prev
              : {
                  id: active.job_id,
                  status: active.status,
                  total: active.total ?? 0,
                  done: active.done ?? 0,
                  remaining: active.remaining ?? undefined,
                  interest_min: active.interest_min,
                  log_tail: [],
                  message: '',
                  error: undefined,
                }
          )
          const activeStatus = active.status === 'queued' || active.status === 'running' || active.status === 'cancelling'
          setCouncilJobBusy(activeStatus)
        }
      } catch (e: any) {
        if (!cancelled) updateLog((prev) => prev || `Council active job error: ${e.message}`)
      }
    }
    if (!councilJobId) {
      loadCouncilActive()
    }
    return () => {
      cancelled = true
    }
  }, [councilJobId])

  useEffect(() => {
    if (!refreshJobId) {
      setJobBusy(false)
      lastListRefresh.current = null
      if (!analyseNewActive) {
        setOracleIdleInfo(null)
        if (oracleStatus !== 'unauthorized') {
          setOracleStatus('offline')
        }
      }
      return
    }

    let cancelled = false
    const currentQuery = { ...queryKey }
    const jobId = refreshJobId!

    async function poll() {
      try {
        const job = await getRefreshJob(jobId)
        if (cancelled) return
        setRefreshJob(job)
        const statusValue = normalizeOracleStatusValue(job.oracle_status)
        if (job.oracle_status) {
          setOracleStatus(statusValue)
        } else if (!analyseNewActive) {
          setOracleStatus('offline')
        }
        if (statusValue === 'unauthorized') {
          setOracleBanner('Oracle auth invalid. Set WOS_ORACLE_USER and WOS_ORACLE_PASS.')
        } else if (job.oracle_status) {
          setOracleBanner(null)
        }
        if (job.oracle_cursor && typeof job.oracle_cursor === 'object') {
          setOracleCursor({
            platform: typeof job.oracle_cursor.platform === 'string' ? job.oracle_cursor.platform : null,
            post_id: typeof job.oracle_cursor.post_id === 'string' ? job.oracle_cursor.post_id : null,
            scraped_at: typeof job.oracle_cursor.scraped_at === 'string' ? job.oracle_cursor.scraped_at : null,
          })
        } else {
          setOracleCursor(null)
        }
        if (
          typeof job.oracle_poll_seconds === 'number' ||
          typeof job.oracle_idle_since === 'number'
        ) {
          setOracleIdleInfo({
            pollSeconds: typeof job.oracle_poll_seconds === 'number' ? job.oracle_poll_seconds : null,
            idleSince: typeof job.oracle_idle_since === 'number' ? job.oracle_idle_since : null,
          })
        } else {
          setOracleIdleInfo(null)
        }
        if (job.log_tail?.length) updateLog(job.log_tail.join('\n'))

        const jobIsActive = job.status === 'queued' || job.status === 'running'
        setJobBusy(jobIsActive)

        if (analyseNewActive) {
          const thresholdBase = analyseNewThreshold ?? interestFilter
          const thresholdValue = Math.max(0, Math.min(thresholdBase, 100))
          const newPosts = Array.isArray(job.new_posts) ? job.new_posts : []
          let added = false
          for (const entry of newPosts) {
            const postId = typeof entry?.post_id === 'string' ? entry.post_id.trim() : ''
            if (!postId) continue
            if (
              analyseNewPendingRef.current.has(postId) ||
              analyseNewInFlightRef.current.has(postId) ||
              analyseNewProcessedRef.current.has(postId)
            ) {
              continue
            }
            const score = parseInterestScore(entry?.interest_score)
            if (score == null || score < thresholdValue) continue
            analyseNewPendingRef.current.add(postId)
            added = true
          }
          if (added) {
            setAnalyseNewHadEligible(true)
            syncAnalyseNewPending()
          }
        }

        const doneCount = job.done ?? 0
        const terminal = job.status === 'done' || job.status === 'error' || job.status === 'cancelled'

        const shouldRefreshList = (doneCount > 0 && doneCount !== lastListRefresh.current && doneCount % 5 === 0) || terminal
        if (shouldRefreshList) {
          try {
            const res = await listPosts(currentQuery)
            if (!cancelled) setList(res)
          } catch {}
          lastListRefresh.current = doneCount
        }

        if (terminal && !cancelled) {
          if (analyseNewActive) {
            if (job.status === 'done') {
              setAnalyseNewRefreshComplete(true)
            } else {
              const statusMessage =
                job.status === 'cancelled'
                  ? 'Analyse New Articles cancelled.'
                  : job.status === 'error'
                    ? 'Analyse New Articles encountered an error.'
                    : ''
              if (statusMessage) {
                updateLog((prev) => prev || statusMessage)
              }
              resetAnalyseNewState()
            }
          }
          setRefreshJobId(null)
        }
      } catch (e: any) {
        if (!cancelled) {
          setJobBusy(false)
          setRefreshJobId(null)
          updateLog(`Job poll error: ${e.message}`)
        }
      }
    }

    poll()
    const timer = window.setInterval(poll, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [
    refreshJobId,
    queryKey.q,
    queryKey.platform,
    queryKey.source,
    queryKey.page,
    queryKey.page_size,
    queryKey.date_from,
    queryKey.date_to,
    analyseNewActive,
    analyseNewThreshold,
    interestFilter,
    oracleStatus,
    resetAnalyseNewState,
    syncAnalyseNewPending,
  ])

  useEffect(() => {
    if (!analyseNewActive) return
    if (analyseNewPendingIds.length === 0) return
    if (councilJobActive || councilJobBusy || councilJobId) return
    if (analyseNewStartInFlightRef.current) return

    const targetIds = Array.from(analyseNewPendingRef.current)
    if (targetIds.length === 0) return

    const thresholdBase = analyseNewThreshold ?? interestFilter
    const thresholdValue = Math.max(0, Math.min(thresholdBase, 100))

    analyseNewStartInFlightRef.current = true
    ;(async () => {
      try {
        const res = await startCouncilAnalysis({
          interest_min: thresholdValue,
          postIds: targetIds,
        })

        if (res.conflict) {
          setCouncilJobBusy(true)
          setCouncilLogEntry((prev) => prev || 'Council analysis already running.')
          updateLog((prev) => prev || 'Council analysis already running.')
          resetAnalyseNewState()
          return
        }

        setAnalyseNewCouncilStarted(true)
        targetIds.forEach((id) => {
          analyseNewPendingRef.current.delete(id)
          analyseNewInFlightRef.current.add(id)
        })
        syncAnalyseNewPending()
        syncAnalyseNewInFlight()

        setCouncilJobId(res.job_id)
        setCouncilLog('')
        setCouncilLogEntry('')
        setCouncilCopyStatus('idle')
        setCouncilJob({
          id: res.job_id,
          status: res.total > 0 ? 'queued' : 'done',
          total: res.total,
          done: 0,
          remaining: res.total,
          interest_min: res.interest_min,
          repairs_total: res.repairs_total ?? 0,
          repairs_done: 0,
          log_tail: [],
          message: '',
          error: undefined,
          current_eta_seconds: null,
          current_started_at: null,
          current_mode: null,
          current_article_tokens: null,
          current_summary_tokens: null,
        })
        setCouncilJobBusy(res.total > 0)
      } catch (err: any) {
        updateLog(`Council start error: ${err.message}`)
      } finally {
        analyseNewStartInFlightRef.current = false
      }
    })()
  }, [
    analyseNewActive,
    analyseNewPendingIds,
    councilJobActive,
    councilJobBusy,
    councilJobId,
    analyseNewThreshold,
    interestFilter,
    resetAnalyseNewState,
    syncAnalyseNewPending,
    syncAnalyseNewInFlight,
  ])

  useEffect(() => {
    if (!analyseNewActive) return
    if (!analyseNewRefreshComplete) return
    if (analyseNewPendingIds.length > 0) return
    if (analyseNewInFlightIds.length > 0) return
    if (councilJobActive || councilJobBusy || councilJobId) return
    if (refreshJobId) return

    const finalMessage = analyseNewHadEligible
      ? 'Analyse New Articles completed.'
      : 'No new articles met the interest threshold; council analysis skipped.'
    updateLog((prev) => prev || finalMessage)
    resetAnalyseNewState()
  }, [
    analyseNewActive,
    analyseNewRefreshComplete,
    analyseNewPendingIds.length,
    analyseNewInFlightIds.length,
    councilJobActive,
    councilJobBusy,
    councilJobId,
    refreshJobId,
    analyseNewHadEligible,
    resetAnalyseNewState,
  ])

  async function onClearAnalysis() {
    if (!pickedId) return
    setBusy(true)
    updateLog('')
    try {
      const res = await clearAnalysis(pickedId)
      updateLog(`Clear analysis: ${JSON.stringify(res)}`)
      const [detailRes, listRes] = await Promise.all([getPost(pickedId), listPosts(queryKey)])
      setDetail(detailRes)
      setList(listRes)
    } catch (e: any) {
      updateLog(`Clear analysis error: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function onEraseAllCouncilAnalysis() {
    if (working || councilJobActive || councilJobBusy) return
    if (typeof window !== 'undefined') {
      const confirmErase = window.confirm(
        'This will erase all stored council analysis results. Summaries will remain. Continue?',
      )
      if (!confirmErase) return
    }

    setBusy(true)
    updateLog('')
    try {
      const res = await eraseAllCouncilAnalysis()
      updateLog(`Erase all council analysis: ${JSON.stringify(res)}`)
      const [listRes, detailRes] = await Promise.all([
        listPosts(queryKey),
        pickedId ? getPost(pickedId) : Promise.resolve<PostDetail | null>(null),
      ])
      setList(listRes)
      if (detailRes) {
        setDetail(detailRes)
      }
    } catch (e: any) {
      updateLog(`Erase council analysis error: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function onRefreshSummaries() {
    updateLog('')
    setRefreshJob(null)
    setJobBusy(true)
    try {
      const start = await startRefreshSummaries()
      setRefreshJobId(start.job_id)
      if (start.conflict) {
        updateLog((prev) => prev || 'A refresh job is already running.')
      }
    } catch (e: any) {
      setJobBusy(false)
      updateLog(`Start error: ${e.message}`)
    }
  }

  async function onSummariseNewArticles() {
    if (jobActive) return
    updateLog('')
    setRefreshJob(null)
    setJobBusy(true)
    try {
      const start = await startRefreshSummaries({ mode: 'new_only' })
      setRefreshJobId(start.job_id)
      if (start.conflict) {
        updateLog((prev) => prev || 'A refresh job is already running.')
      }
    } catch (e: any) {
      setJobBusy(false)
      updateLog(`Start error: ${e.message}`)
    }
  }

  async function handleAnalyseNewToggle(next: boolean) {
    if (next) {
      if (working || jobActive || councilJobActive || councilJobBusy) {
        updateLog((prev) => prev || 'Finish current jobs before analysing new articles.')
        resetAnalyseNewState()
        return
      }
      if (oracleOnline && !oracleBaseUrl.trim()) {
        const message = 'Set Oracle base URL before enabling online article retrieval.'
        updateLog((prev) => prev || message)
        setOracleBanner(message)
        resetAnalyseNewState()
        return
      }
      setOracleBanner(null)
      setOracleStatus(oracleOnline ? 'connecting' : 'offline')
      clearAnalyseNewQueues()
      setAnalyseNewRefreshComplete(false)
      setAnalyseNewHadEligible(false)
      const thresholdValue = Math.max(0, Math.min(interestFilter, 100))
      setAnalyseNewActive(true)
      setAnalyseNewCouncilStarted(false)
      setAnalyseNewThreshold(thresholdValue)
      setRefreshJob(null)
      setJobBusy(true)
      try {
        const start = await startRefreshSummaries({
          mode: 'new_only',
          collectNewPosts: true,
          oracleOnline: oracleOnline && Boolean(oracleBaseUrl.trim()),
          oracleBaseUrl: oracleOnline ? oracleBaseUrl.trim() : undefined,
        })
        setRefreshJobId(start.job_id)
        if (start.conflict) {
          setJobBusy(false)
          updateLog((prev) => prev || 'A refresh job is already running.')
          resetAnalyseNewState()
          return
        }
        updateLog((prev) => prev || 'Analyse New Articles started.')
      } catch (e: any) {
        setJobBusy(false)
        updateLog(`Start error: ${e.message}`)
        resetAnalyseNewState()
      }
      return
    }

    const refreshId = refreshJobId
    const councilId = councilJobId
    resetAnalyseNewState()
    if (refreshId) {
      try {
        await stopRefreshJob(refreshId)
      } catch (e: any) {
        updateLog((prev) => `${prev ? `${prev}\n` : ''}Stop error: ${e.message}`)
      }
    }
    if (councilId) {
      try {
        await stopCouncilJob(councilId)
      } catch (e: any) {
        updateLog((prev) => `${prev ? `${prev}\n` : ''}Council stop error: ${e.message}`)
      }
    }
  }

  async function onStopRefreshSummaries() {
    if (!refreshJobId) return
    try {
      await stopRefreshJob(refreshJobId)
    } catch (e: any) {
      updateLog(`Stop error: ${e.message}`)
    }
  }

  const handleInterestThresholdChange = (value: number) => {
    const bounded = Math.max(0, Math.min(100, value))
    setInterestFilter(bounded)
    setPage(1)
  }

  async function onStartCouncilProcessing() {
    try {
      const repairMissing = !councilPrimed
      const res = await startCouncilAnalysis({
        interest_min: interestFilter,
        repair_missing: repairMissing,
      })
      setCouncilJobId(res.job_id)
      if (!res.conflict) {
        setCouncilLog('')
        setCouncilLogEntry('')
        setCouncilCopyStatus('idle')
        setCouncilJob({
          id: res.job_id,
          status: res.total > 0 ? 'queued' : 'done',
          total: res.total,
          done: 0,
          remaining: res.total,
          interest_min: res.interest_min,
          repairs_total: res.repairs_total ?? 0,
          repairs_done: 0,
          log_tail: [],
          message: '',
          error: undefined,
          current_eta_seconds: null,
          current_started_at: null,
          current_mode: null,
          current_article_tokens: null,
          current_summary_tokens: null,
        })
        setCouncilJobBusy(res.total > 0)
        if (repairMissing && !councilPrimed) {
          setCouncilPrimed(true)
        }
      } else {
        setCouncilJobBusy(true)
        setCouncilLogEntry((prev) => prev || 'Council analysis already running.')
        updateLog((prev) => prev || 'Council analysis already running.')
      }
    } catch (e: any) {
      updateLog(`Council start error: ${e.message}`)
    }
  }

  async function onStopCouncilProcessing() {
    if (!councilJobId) return
    try {
      await stopCouncilJob(councilJobId)
    } catch (e: any) {
      updateLog(`Council stop error: ${e.message}`)
    }
  }

  async function onAnalyseSelected() {
    if (!pickedId) {
      updateLog('Pick a post first.')
      return
    }
    setBusy(true)
    const startedAt = new Date().toLocaleTimeString()
    updateLog(`Council analysis started at ${startedAt}`)
    const postId = pickedId as string
    const createStageRunner = (
      key: string,
      action: () => Promise<void>,
    ): { key: string; run: () => Promise<void> } => ({
      key,
      run: async () => {
        const label = formatStageLabel(key)
        const startStamp = new Date().toLocaleTimeString()
        appendLog(`[${startStamp}] Starting ${label}…`)
        try {
          await action()
          const endStamp = new Date().toLocaleTimeString()
          appendLog(`[${endStamp}] Completed ${label}.`)
        } catch (err: any) {
          const failStamp = new Date().toLocaleTimeString()
          const message = err?.message ?? String(err)
          appendLog(`[${failStamp}] ${label} failed: ${message}`)
          throw err
        }
      },
    })
    const stageRunners: { key: string; run: () => Promise<void> }[] = [
      createStageRunner('entity', async () => {
        const res = await runStage({ post_id: postId, stages: ['entity'], overwrite: true })
        logRunOutput(['entity'], res.stdout ?? '', res.stderr ?? '', { append: true })
      }),
      createStageRunner('researcher', async () => {
        const researchRes = await runResearch(postId)
        const researchLog = formatResearchLog(researchRes.research)
        const researchEntry = researchLog ?? 'Research completed, but no payload returned.'
        appendLog(researchEntry)
      }),
      ...['claims', 'context', 'verifier', 'for', 'against', 'direction', 'chairman'].map((stage) =>
        createStageRunner(stage, async () => {
          const res = await runStage({ post_id: postId, stages: [stage], overwrite: true })
          logRunOutput([stage], res.stdout ?? '', res.stderr ?? '', { append: true })
        }),
      ),
    ]
    const stageKeys = stageRunners.map((stage) => stage.key)
    setCouncilProgress({ stages: stageKeys, currentStageIndex: 0, completed: 0 })

    try {
      const cleared = await clearAnalysis(postId)
      appendLog(`Cleared analysis: ${JSON.stringify(cleared)}`)
    } catch (e: any) {
      setCouncilProgress(null)
      setBusy(false)
      appendLog(`Clear analysis error: ${e.message}`)
      return
    }

    let lastStage: string | null = null
    try {
      for (let i = 0; i < stageRunners.length; i += 1) {
        const { key, run } = stageRunners[i]
        lastStage = key
        setCouncilProgress((prev) =>
          prev
            ? { ...prev, currentStageIndex: i }
            : { stages: stageKeys, currentStageIndex: i, completed: i }
        )
        await run()
        setCouncilProgress((prev) =>
          prev
            ? { ...prev, completed: i + 1 }
            : { stages: stageKeys, currentStageIndex: i, completed: i + 1 }
        )
      }
        const [d, l] = await Promise.all([getPost(postId), listPosts(queryKey)])
      setDetail(d)
      setList(l)
    } catch (e: any) {
      const stageLabel = lastStage ? formatStageLabel(lastStage) : ''
      const prefix = stageLabel ? `Analyse error during ${stageLabel}: ` : 'Analyse error: '
      appendLog(`${prefix}${e.message}`)
    } finally {
      setCouncilProgress(null)
      setBusy(false)
    }
  }

  async function onSummariseArticle() {
    if (!pickedId) return
    const stages = ['entity', 'summariser', 'claims', 'context', 'for', 'against', 'direction']
    setBusy(true)
    updateLog('Summarisation started.')
    try {
      const res = await runStage({ post_id: pickedId, stages, overwrite: true })
      logRunOutput(stages, res.stdout ?? '', res.stderr ?? '')
      const [detailRes, listRes] = await Promise.all([getPost(pickedId), listPosts(queryKey)])
      setDetail(detailRes)
      setList(listRes)
    } catch (e: any) {
      updateLog(`Summarise error: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const handleCloseDetail = () => {
    setPickedId(null)
    setDetail(null)
  }

  const handleDateChange = (date: Date) => {
    setSelectedDate(date)
    setPage(1)
  }

  return (
    <div className="wrap">
      {oracleBanner && <div className="oracle-banner">{oracleBanner}</div>}
      {councilProgress && (
        <div className="analysis-overlay">
          <div className="analysis-overlay-content">
            <div className="analysis-overlay-title">Analysing post…</div>
            {councilStageLabel && (
              <div className="analysis-overlay-stage">Processing {councilStageLabel}</div>
            )}
            <div className="analysis-overlay-progress">
              <div style={{ width: `${councilPercent}%` }} />
            </div>
            <div className="analysis-overlay-counter">
              {councilDone}/{councilTotal} council members complete
            </div>
            <div className="analysis-overlay-log">
              <div className="analysis-overlay-log-title">Live Logs</div>
              <pre className="analysis-overlay-log-body">
                {log.trim() ? log : 'Waiting for council output…'}
              </pre>
            </div>
          </div>
        </div>
      )}
      <header className="topbar">
        <div className="header-pane">
          <h1>Wisdom of Sheep v1.0 - (C)2025 CH Electronics</h1>
          <div className="filters">
            <input
              placeholder="Search (title/text)"
              value={q}
              onChange={(e) => {
                setQ(e.target.value)
                setPage(1)
              }}
            />
            <input
              placeholder="platform (e.g. reddit)"
              value={platform}
              onChange={(e) => {
                setPlatform(e.target.value)
                setPage(1)
              }}
            />
            <input
              placeholder="source (e.g. stocks)"
              value={source}
              onChange={(e) => {
                setSource(e.target.value)
                setPage(1)
              }}
            />
            <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>
        </div>
      </header>

      <TraderBar selectedDate={selectedDate} onDateChange={handleDateChange} />

      <div className="toolbar">
        <div className="toolbar-row toolbar-primary">
          <button
            onClick={jobActive ? onStopRefreshSummaries : onRefreshSummaries}
            disabled={(!jobActive && working) || analyseNewActive}
          >
            {jobActive ? 'Stop Refresh' : 'Summarise and Backfill'}
          </button>
          <button onClick={onSummariseNewArticles} disabled={jobActive || working || analyseNewActive}>
            Summarise New Articles
          </button>

          <div className="progress" title={`${batchDone}/${batchTotal}`}>
            <div style={{ width: `${progressPct}%` }} />
          </div>
          {batchTotal > 0 && <div className="muted small">{batchDone}/{batchTotal}</div>}
          {batchMsg && (
            <div className="muted small" style={{ marginLeft: 8 }}>
              {batchMsg}
            </div>
          )}
          {logSnippet && (
            <div className="muted small toolbar-log-snippet" title={log}>
              {logSnippet}
            </div>
          )}
        </div>

        <div className="toolbar-row toolbar-council">
          <label className={`toolbar-toggle${councilJobActive ? ' active' : ''}`}>
            <input
              type="checkbox"
              checked={councilJobActive}
              onChange={(event) => {
                if (event.target.checked) {
                  if (!councilJobActive) onStartCouncilProcessing()
                } else if (councilJobActive) {
                  onStopCouncilProcessing()
                }
              }}
              disabled={analyseNewActive || working || councilJobBusy}
            />
            <span className="toolbar-toggle-track" />
            <span className="toolbar-toggle-label">Council Analysis</span>
          </label>
          <div className="toolbar-progress-group">
            <div className="council-progress-pair">
              <div className="council-progress-block">
                <div
                  className="progress council"
                  title={councilJobTotal ? `${councilJobDone}/${councilJobTotal}` : undefined}
                >
                  <div style={{ width: `${councilJobPercent}%` }} />
                </div>
                {councilJobTotal > 0 && (
                  <div className="muted small">{councilJobRemaining} remaining</div>
                )}
                {councilDisplayLog && (
                  <div className="muted small toolbar-log-snippet" title={councilDisplayLog}>
                    {councilDisplayLog}
                  </div>
                )}
              </div>
              <div
                className="eta-progress-block"
                data-testid="council-eta"
                title={
                  etaRemainingSeconds != null
                    ? `~${Math.max(Math.round(etaRemainingSeconds), 0)}s remaining`
                    : councilJobActive
                      ? 'Estimating time remaining'
                      : 'Council analysis idle'
                }
              >
                <div className="eta-progress-label">
                  <span className="muted small">Time remaining</span>
                  <span className={`eta-progress-value${councilJobActive ? '' : ' inactive'}`}>
                    {etaDisplayText}
                  </span>
                </div>
                <div className={`progress eta${councilJobActive ? '' : ' inactive'}`}>
                  <div style={{ width: `${etaPercent}%` }} />
                </div>
              </div>
            </div>
          </div>
          <button
            type="button"
            className="toolbar-danger"
            onClick={onEraseAllCouncilAnalysis}
            disabled={working || councilJobActive || councilJobBusy || analyseNewActive}
            title="Remove council analysis outputs for all posts"
          >
            Erase all Council Analysis
          </button>
        </div>

        <div className="toolbar-row toolbar-analyse-new">
          <div className="toolbar-analyse-new-main">
            <label className={`toolbar-toggle${analyseNewActive ? ' active' : ''}`}>
              <input
                type="checkbox"
                checked={analyseNewActive}
                onChange={(event) => handleAnalyseNewToggle(event.target.checked)}
                disabled={
                  working || (!analyseNewActive && (jobActive || councilJobActive || councilJobBusy))
                }
              />
              <span className="toolbar-toggle-track" />
              <span className="toolbar-toggle-label">Analyse New Articles</span>
            </label>
            <div className="oracle-controls">
              <label className="oracle-online-toggle">
                <input
                  type="checkbox"
                  checked={oracleOnline}
                  onChange={(event) => {
                    const next = event.target.checked
                    setOracleOnline(next)
                    if (!next) {
                      setOracleStatus('offline')
                      setOracleBanner(null)
                    }
                  }}
                  disabled={oracleControlsDisabled}
                />
                <span>Online source</span>
              </label>
              <input
                type="url"
                className="oracle-base-url"
                placeholder="https://oracle.example.com"
                value={oracleBaseUrl}
                onChange={(event) => setOracleBaseUrl(event.target.value)}
                disabled={oracleControlsDisabled}
              />
              <div className={`oracle-status ${oracleStatusVariant}`} title={oracleStatusText}>
                <span className="oracle-status-dot" aria-hidden="true" />
                <span className="oracle-status-text">{oracleStatusText}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="toolbar-row toolbar-slider">
          <label
            htmlFor="interest-filter"
            className={interestSliderDisabled ? 'disabled' : undefined}
            title={
              interestSliderDisabled
                ? 'Interest threshold is locked while analysis runs.'
                : undefined
            }
          >
            Interest Score ≥ <span>{interestFilter}%</span>
          </label>
          <input
            id="interest-filter"
            type="range"
            min={0}
            max={100}
            step={5}
            value={interestFilter}
            disabled={interestSliderDisabled}
            aria-disabled={interestSliderDisabled}
            onChange={(e) => {
              const next = Number(e.target.value)
              handleInterestThresholdChange(Number.isNaN(next) ? 0 : next)
            }}
          />
        </div>
      </div>

      <div className="content">
        <aside className="pane posts">
          {loadingList && <div className="muted">Loading…</div>}
          {list && (
            <PostList
              items={list.items}
              total={list.total}
              page={list.page}
              pageSize={list.page_size}
              onPage={setPage}
              onPick={setPickedId}
              selectedId={pickedId}
            />
          )}
        </aside>
        <section className="pane detail">
          {!detail && <div className="muted">Pick a post…</div>}
          {detail && (
            <PostDetailView
              detail={detail}
              onSummarise={onSummariseArticle}
              onAnalyse={onAnalyseSelected}
              onClear={onClearAnalysis}
              onClose={handleCloseDetail}
              controlsDisabled={panelDisabled}
              selectedTicker={researchSelection?.ticker ?? selectedTicker}
              tickers={articleTickers}
              onSelectTicker={setSelectedTicker}
              articleId={detail.post?.post_id ?? null}
              centerTimestamp={detail.post?.posted_at ?? detail.post?.scraped_at ?? null}
              technical={researchSelection?.data.technical ?? null}
            />
          )}
        </section>
      </div>

      <div className="log-panels">
        <section className={`card log primary-log${logExpanded ? ' log-expanded' : ''}`}>
          <div className="card-header log-header">
            <div className="log-header-left">
              <button
                type="button"
                className="log-toggle"
                onClick={() => setLogExpanded((prev) => !prev)}
              >
                <span className="log-toggle-icon">{logExpanded ? '▾' : '▸'}</span>
                <span>Logs</span>
              </button>
              {working && <div className="muted log-working">(working…)</div>}
            </div>
            <div className="log-header-actions">
              {copyStatus === 'copied' && <span className="log-copy-status">Copied!</span>}
              {copyStatus === 'failed' && <span className="log-copy-status error">Copy failed</span>}
              <button
                type="button"
                className="log-copy"
                onClick={async () => {
                  if (!navigator.clipboard) {
                    setCopyStatus('failed')
                    return
                  }
                  try {
                    await navigator.clipboard.writeText(log)
                    setCopyStatus('copied')
                  } catch (err) {
                    console.error('Copy failed', err)
                    setCopyStatus('failed')
                  }
                }}
                disabled={!log.trim()}
              >
                Copy
              </button>
            </div>
          </div>
          <pre className={`logbox${logExpanded ? ' expanded' : ''}`}>{log}</pre>
        </section>
        <section className="card log council-log">
          <div className="card-header log-header">
            <div className="log-header-left">
              <span className="log-heading">Council Analysis Log</span>
            </div>
            <div className="log-header-actions">
              {councilCopyStatus === 'copied' && <span className="log-copy-status">Copied!</span>}
              {councilCopyStatus === 'failed' && <span className="log-copy-status error">Copy failed</span>}
              <button
                type="button"
                className="log-copy"
                onClick={async () => {
                  if (!navigator.clipboard) {
                    setCouncilCopyStatus('failed')
                    return
                  }
                  try {
                    await navigator.clipboard.writeText(councilLog)
                    setCouncilCopyStatus('copied')
                  } catch (err) {
                    console.error('Council log copy failed', err)
                    setCouncilCopyStatus('failed')
                  }
                }}
                disabled={!councilLog.trim()}
              >
                Copy
              </button>
            </div>
          </div>
          <pre className="logbox council-logbox">
            {councilLog.trim()
              ? councilLog
              : 'Council analysis output will stream here when a job runs.'}
          </pre>
        </section>
      </div>

      <footer className="foot">API: {import.meta.env.VITE_API ?? 'http://127.0.0.1:8000'}</footer>
    </div>
  )
}

