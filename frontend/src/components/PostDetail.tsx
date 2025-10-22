import { Fragment, useEffect, useMemo, useState } from 'react'

import type { PostDetail, ResearchTechnical } from '../types'
import { spamClassForPct } from '../utils/spam'
import { buildInterestDisplay } from '../utils/interest'
import StageViewer, { parseChairmanStageData, parseSummariserData } from './StageViewer'
import PriceWindowChart from './PriceWindowChart'

const STAGES = [
  'chairman',
  'entity',
  'summariser',
  'conversation_hub',
  'claims',
  'context',
  'for',
  'against',
  'direction',
  'researcher',
  'technical_research',
  'sentiment_research',
] as const

const STAGE_LABELS: Record<string, string> = {
  chairman: 'Chairman Verdict',
  entity: 'Entity',
  summariser: 'Summariser',
  conversation_hub: 'Conversation Hub',
  claims: 'Claims',
  context: 'Context',
  for: 'For',
  against: 'Against',
  direction: 'Direction',
  researcher: 'Researcher',
  technical_research: 'Technical Research',
  sentiment_research: 'Sentiment Research',
}

const DEFAULT_SPAM_META = {
  hasData: false,
  pct: 0,
  display: '—',
  trackClass: spamClassForPct(null),
  reason: 'not assessed',
  reasons: [] as string[],
}

const COLLAPSIBLE_STAGES = new Set<
  (typeof STAGES)[number]
>([
  'entity',
  'summariser',
  'conversation_hub',
  'claims',
  'context',
  'for',
  'against',
  'direction',
  'researcher',
  'technical_research',
  'sentiment_research',
])

const formatStageLabel = (name: string) =>
  STAGE_LABELS[name] ||
  name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value)

const normalizeBulletValue = (value: unknown): string[] => {
  if (!value) return []
  if (Array.isArray(value)) {
    return value
      .map((item) => (item == null ? '' : String(item).trim()))
      .filter((item) => item.length > 0)
  }

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return []
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) {
        return normalizeBulletValue(parsed)
      }
    } catch (err) {
      // fall through to newline handling
    }
    return trimmed
      .split(/\r?\n+/)
      .map((line) => line.replace(/^[•\-*]\s*/, '').trim())
      .filter((line) => line.length > 0)
  }

  if (isPlainObject(value)) {
    if ('bullets' in value) return normalizeBulletValue((value as Record<string, unknown>).bullets)
    if ('summary_bullets' in value)
      return normalizeBulletValue((value as Record<string, unknown>).summary_bullets)
  }

  return []
}

const arraysEqual = (a: string[], b: string[]) => {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

const normalizeTickerValue = (value?: string | null) => {
  if (typeof value !== 'string') return ''
  return value.trim().toUpperCase()
}

const DIRECTION_BADGE_LABELS: Record<string, string> = {
  up: 'Bullish',
  down: 'Bearish',
  neutral: 'Neutral',
}

const DIRECTION_STRENGTH_BADGES = ['None', 'Low', 'Medium', 'High']

const formatTitleCase = (value: string) =>
  value
    .split(/[_\s]+/)
    .filter((part) => part.length > 0)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')

const describeDirection = (direction?: string | null, strength?: number | null) => {
  if (!direction) return ''
  const normalized = direction.toLowerCase()
  const base = DIRECTION_BADGE_LABELS[normalized] ?? formatTitleCase(direction)
  if (typeof strength === 'number' && Number.isFinite(strength)) {
    const idx = Math.max(0, Math.min(DIRECTION_STRENGTH_BADGES.length - 1, Math.round(strength)))
    const suffix = DIRECTION_STRENGTH_BADGES[idx]
    if (suffix && suffix !== 'None') {
      return `${base} · ${suffix}`
    }
  }
  return base
}

const describeTimeframe = (timeframe?: string | null) => {
  if (!timeframe) return ''
  return formatTitleCase(timeframe)
}

const describeScale = (value?: number | null) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return ''
  const labels = ['Low', 'Moderate', 'High', 'Severe']
  const idx = Math.max(0, Math.min(labels.length - 1, Math.round(value)))
  return labels[idx] ?? value.toString()
}

const describePercent = (value?: number | null) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return ''
  return `${Math.round(value)}%`
}

export default function PostDetailView({
  detail,
  onSummarise,
  onAnalyse,
  onClear,
  onClose,
  controlsDisabled,
  selectedTicker,
  tickers,
  onSelectTicker,
  articleId,
  centerTimestamp,
  technical,
}: {
  detail: PostDetail
  onSummarise: () => Promise<void>
  onAnalyse: () => Promise<void>
  onClear: () => Promise<void>
  onClose: () => void
  controlsDisabled: boolean
  selectedTicker: string | null
  tickers: string[]
  onSelectTicker: (ticker: string | null) => void
  articleId: string | null
  centerTimestamp: string | null
  technical: ResearchTechnical | null
}) {
  const summariserData = parseSummariserData(detail.stages?.summariser)
  const spamMeta = summariserData?.spam ?? DEFAULT_SPAM_META

  const summaryBullets = summariserData?.summaryBullets ?? []
  const catalysts = summariserData?.catalysts ?? []
  const risks = summariserData?.risks ?? []

  const chairmanData = parseChairmanStageData(detail.stages?.chairman)
  const chairmanPlainText = chairmanData?.plainEnglish?.trim() ?? ''
  const chairmanDirectionLabel = chairmanData ? describeDirection(chairmanData.direction, chairmanData.directionStrength) : ''
  const chairmanRiskLabel = chairmanData?.riskLevel ? formatTitleCase(chairmanData.riskLevel) : ''
  const chairmanTimeframeLabel = chairmanData ? describeTimeframe(chairmanData.timeframe) : ''
  const chairmanConvictionLabel = chairmanData ? describePercent(chairmanData.conviction) : ''
  const chairmanTradabilityLabel = chairmanData ? describePercent(chairmanData.tradability) : ''
  const chairmanUncertaintyLabel = chairmanData ? describeScale(chairmanData.uncertainty) : ''
  const chairmanStaleRiskLabel = chairmanData ? describeScale(chairmanData.staleRisk) : ''

  const chairmanTags: { key: string; label: string; className: string }[] = []
  if (chairmanDirectionLabel) {
    const dirClass = chairmanData?.direction ? `direction-${chairmanData.direction.toLowerCase()}` : ''
    chairmanTags.push({
      key: 'direction',
      label: chairmanDirectionLabel,
      className: ['chairman-tag', dirClass].filter(Boolean).join(' '),
    })
  }
  if (chairmanRiskLabel) {
    const riskClass = chairmanData?.riskLevel ? `risk-${chairmanData.riskLevel.toLowerCase()}` : ''
    chairmanTags.push({
      key: 'risk',
      label: `Risk: ${chairmanRiskLabel}`,
      className: ['chairman-tag', riskClass].filter(Boolean).join(' '),
    })
  }
  if (chairmanTimeframeLabel) {
    chairmanTags.push({ key: 'timeframe', label: chairmanTimeframeLabel, className: 'chairman-tag timeframe' })
  }

  const chairmanMetaItems: { label: string; value: string }[] = []
  if (chairmanConvictionLabel) chairmanMetaItems.push({ label: 'Conviction', value: chairmanConvictionLabel })
  if (chairmanTradabilityLabel) chairmanMetaItems.push({ label: 'Tradability', value: chairmanTradabilityLabel })
  if (chairmanUncertaintyLabel) chairmanMetaItems.push({ label: 'Uncertainty', value: chairmanUncertaintyLabel })
  if (chairmanStaleRiskLabel) chairmanMetaItems.push({ label: 'Stale Risk', value: chairmanStaleRiskLabel })

  const extraSummaryCandidates: unknown[] = []
  const extras = isPlainObject(detail.extras) ? (detail.extras as Record<string, unknown>) : null
  if (extras) {
    extraSummaryCandidates.push(extras['summary_bullets'])
    extraSummaryCandidates.push(extras['summaryBullets'])
    const extrasSummary = extras['summary']
    if (isPlainObject(extrasSummary)) {
      extraSummaryCandidates.push((extrasSummary as Record<string, unknown>)['bullets'])
      extraSummaryCandidates.push((extrasSummary as Record<string, unknown>)['summary_bullets'])
      extraSummaryCandidates.push((extrasSummary as Record<string, unknown>)['summaryBullets'])
    }
  }
  extraSummaryCandidates.push((detail.post as Record<string, unknown>)['summary_bullets'])
  extraSummaryCandidates.push((detail.post as Record<string, unknown>)['summaryBullets'])

  const normalizedResearchTicker = normalizeTickerValue(selectedTicker)

  const researchExtras = useMemo(() => {
    if (!extras) return null
    const research = (extras as Record<string, unknown>).research
    if (!isPlainObject(research)) return null
    return research as Record<string, unknown>
  }, [extras])

  const researchSelection = useMemo(() => {
    if (!researchExtras) return null
    const tickersValue = researchExtras['tickers']
    if (!isPlainObject(tickersValue)) return null
    const entries = Object.entries(tickersValue as Record<string, unknown>).filter(([, value]) =>
      isPlainObject(value),
    )
    if (entries.length === 0) return null

    const normalizedMap = new Map<string, Record<string, unknown>>()
    entries.forEach(([key, value]) => {
      normalizedMap.set(normalizeTickerValue(key), value as Record<string, unknown>)
    })

    const orderedField =
      (researchExtras['ordered_tickers'] as unknown) ??
      (researchExtras['ordered'] as unknown) ??
      []
    const orderedKeys = Array.isArray(orderedField)
      ? (orderedField as unknown[])
          .map((item) =>
            normalizeTickerValue(typeof item === 'string' ? item : String(item ?? '')),
          )
          .filter((key) => key.length > 0)
      : []

    const desiredTicker =
      (normalizedResearchTicker && normalizedMap.has(normalizedResearchTicker)
        ? normalizedResearchTicker
        : undefined) ||
      orderedKeys.find((key) => normalizedMap.has(key)) ||
      Array.from(normalizedMap.keys())[0]

    if (!desiredTicker) return null
    return { ticker: desiredTicker, data: normalizedMap.get(desiredTicker)! }
  }, [researchExtras, normalizedResearchTicker])

  const researchSummaryText =
    typeof researchSelection?.data.summary_text === 'string'
      ? (researchSelection?.data.summary_text as string).trim()
      : ''
  const researchUpdatedAt =
    typeof researchSelection?.data.updated_at === 'string'
      ? (researchSelection?.data.updated_at as string)
      : typeof researchExtras?.['updated_at'] === 'string'
        ? (researchExtras?.['updated_at'] as string)
        : ''

  let alternateSummaryBullets: string[] = []
  for (const candidate of extraSummaryCandidates) {
    const normalized = normalizeBulletValue(candidate)
    if (normalized.length > 0) {
      alternateSummaryBullets = normalized
      break
    }
  }

  const showAlternateSummary =
    alternateSummaryBullets.length > 0 && !arraysEqual(alternateSummaryBullets, summaryBullets)

  const metaValue = (value: unknown) => {
    if (value == null) return '—'
    const str = String(value).trim()
    return str.length > 0 ? str : '—'
  }

  const interestRecord = detail.interest ?? null
  const interestMeta = useMemo(() => buildInterestDisplay(interestRecord), [interestRecord])

  const interestSummaryText = useMemo(() => {
    if (!interestRecord) return ''
    const summary = typeof interestRecord.interest_why === 'string' ? interestRecord.interest_why.trim() : ''
    if (summary.length > 0) return summary
    return interestMeta.status === 'ok' ? interestMeta.reason : ''
  }, [interestRecord, interestMeta])

  const interestInfoRows = useMemo(() => {
    if (!interestRecord || interestMeta.status !== 'ok') return []
    const rows: { label: string; value: string }[] = []
    if (interestRecord.ticker && interestRecord.ticker.trim().length > 0) {
      rows.push({ label: 'Ticker', value: interestRecord.ticker })
    }
    rows.push({ label: 'Score', value: interestMeta.valueText })
    if (typeof interestRecord.council_recommended === 'boolean') {
      rows.push({
        label: 'Council recommended',
        value: interestRecord.council_recommended ? 'Yes' : 'No',
      })
    }
    if (interestRecord.council_priority && interestRecord.council_priority.trim().length > 0) {
      rows.push({ label: 'Priority', value: interestRecord.council_priority })
    }
    if (interestRecord.calculated_at && interestRecord.calculated_at.trim().length > 0) {
      rows.push({ label: 'Calculated', value: interestRecord.calculated_at })
    }
    return rows
  }, [interestRecord, interestMeta.status, interestMeta.valueText])

  const interestMetricsRows = useMemo(() => {
    if (!interestRecord || interestMeta.status !== 'ok') return []
    const rawMetrics = interestRecord.metrics
    if (!rawMetrics || typeof rawMetrics !== 'object') return []
    const metrics = rawMetrics as Record<string, unknown>
    const rows: { label: string; value: string }[] = []
    const formatPct = (value: unknown, digits = 1) =>
      typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}%` : null
    const formatRatio = (value: unknown, digits = 2) =>
      typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : null
    const add = (label: string, value: string | null | undefined) => {
      if (value && value.trim().length > 0) {
        rows.push({ label, value })
      }
    }
    add('1d return', formatPct(metrics.ret1d_pct))
    add('5d return', formatPct(metrics.ret5d_pct))
    add('Return volatility (60d)', formatPct(metrics.ret_std_pct))
    add('Volatility ratio', formatRatio(metrics.vol_ratio))
    if (typeof metrics.trend_strength === 'number' && Number.isFinite(metrics.trend_strength)) {
      add('Trend strength', String(metrics.trend_strength))
    }
    if (typeof metrics.slope_pct_per_day === 'number' && Number.isFinite(metrics.slope_pct_per_day)) {
      add('Trend slope', `${(metrics.slope_pct_per_day as number).toFixed(2)}%/day`)
    }
    const noveltyUnit =
      typeof metrics.novelty_unit === 'number' && Number.isFinite(metrics.novelty_unit)
        ? (metrics.novelty_unit as number)
        : null
    const noveltyNote =
      typeof metrics.novelty_note === 'string' && metrics.novelty_note.trim().length > 0
        ? metrics.novelty_note.trim()
        : ''
    if (noveltyUnit != null) {
      const percent = `${Math.round(noveltyUnit * 100)}%`
      add('Novelty', noveltyNote ? `${percent} · ${noveltyNote}` : percent)
    } else if (noveltyNote) {
      add('Novelty', noveltyNote)
    }
    const noveltyWindow =
      typeof metrics.novelty_window === 'string' && metrics.novelty_window.trim().length > 0
        ? metrics.novelty_window.trim()
        : ''
    if (noveltyWindow) {
      add('Novelty window', noveltyWindow)
    }
    if (typeof metrics.spam_pct === 'number' && Number.isFinite(metrics.spam_pct)) {
      add('Spam dampener', formatPct(metrics.spam_pct, 0))
    }
    if (typeof metrics.mover === 'boolean') {
      add('Mover heuristics', metrics.mover ? 'Yes' : 'No')
    }
    return rows
  }, [interestRecord, interestMeta.status])

  const articleText = detail.post.text ? detail.post.text.trim() : ''
  const articleDisplay = articleText.length > 0 ? articleText : '(no text)'

  const articleParagraphs = useMemo(() => {
    const normalized = articleDisplay.replace(/\r/g, '')
    const blocks = normalized.split(/\n{2,}/)

    return blocks.map((block, blockIdx) => {
      const lines = block.split('\n')
      return (
        <p key={blockIdx} className="article-paragraph">
          {lines.map((line, lineIdx) => (
            <Fragment key={lineIdx}>
              {line.length > 0 ? line : '\u00a0'}
              {lineIdx < lines.length - 1 && <br />}
            </Fragment>
          ))}
        </p>
      )
    })
  }, [articleDisplay])

  const researchParagraphs = useMemo(() => {
    if (!researchSummaryText) return []
    const normalized = researchSummaryText.replace(/\r/g, '')
    return normalized.split(/\n{2,}/).map((block, blockIdx) => {
      const lines = block.split('\n')
      return (
        <p key={`research-${blockIdx}`} className="article-paragraph">
          {lines.map((line, lineIdx) => (
            <Fragment key={lineIdx}>
              {line.length > 0 ? line : '\u00a0'}
              {lineIdx < lines.length - 1 && <br />}
            </Fragment>
          ))}
        </p>
      )
    })
  }, [researchSummaryText])

  const handleClear = () => {
    if (!controlsDisabled) void onClear()
  }

  const handleClose = () => {
    onClose()
  }

  const handleSummarise = () => {
    if (!controlsDisabled) void onSummarise()
  }

  const handleAnalyse = () => {
    if (!controlsDisabled) void onAnalyse()
  }

  const buttonBarClass = ['detail-button-bar', controlsDisabled ? 'disabled' : '']
    .filter(Boolean)
    .join(' ')

  const summarySections = useMemo(
    () =>
      [
        { key: 'summary', title: 'Summary', items: summaryBullets },
        { key: 'catalysts', title: 'Catalysts', items: catalysts },
        { key: 'risks', title: 'Risks', items: risks },
      ].filter((section) => section.items.length > 0),
    [summaryBullets, catalysts, risks],
  )

  const stagePayloads = detail.stages ?? {}
  const activeResearchTicker = researchSelection?.ticker ?? normalizedResearchTicker

  const createInitialCollapsedStages = () => {
    const initial: Record<string, boolean> = {}
    COLLAPSIBLE_STAGES.forEach((stage) => {
      initial[stage] = true
    })
    return initial
  }

  const [isArticleCollapsed, setIsArticleCollapsed] = useState(true)
  const [collapsedStages, setCollapsedStages] = useState<Record<string, boolean>>(
    () => createInitialCollapsedStages(),
  )

  useEffect(() => {
    setIsArticleCollapsed(true)
    setCollapsedStages(createInitialCollapsedStages())
  }, [detail.post?.post_id, detail.post?.url])

  const toggleArticleCollapsed = () => {
    setIsArticleCollapsed((prev) => !prev)
  }

  const toggleStageCollapsed = (stage: (typeof STAGES)[number]) => {
    if (!COLLAPSIBLE_STAGES.has(stage)) return
    setCollapsedStages((prev) => ({
      ...prev,
      [stage]: !(prev[stage] ?? false),
    }))
  }

  return (
    <div className="card">
      <div className="card-header detail-header">
        <div className="detail-header-left">
          <h2>{detail.post.title || detail.post.post_id}</h2>
          <div className="muted detail-header-sub">
            <span>{metaValue(detail.post.platform)}</span>
            <span className="detail-header-sep" aria-hidden="true">
              ·
            </span>
            <span>{metaValue(detail.post.source)}</span>
          </div>
          <div
            className={[
              'spam-meter',
              'interest-meter',
              'detail-interest-meter',
              interestMeta.status === 'error' ? 'interest-meter-error' : '',
            ]
              .filter(Boolean)
              .join(' ')}
          >
            <div className="spam-meter-header">
              <span className="spam-meter-label">Interest score</span>
              <span className="spam-meter-value">{interestMeta.valueText}</span>
            </div>
            <div className={['spam-meter-track', interestMeta.trackClass].join(' ')}>
              <div
                className={[
                  'spam-meter-handle',
                  interestMeta.hasScore ? '' : 'spam-meter-handle-empty',
                  interestMeta.status === 'error' ? 'interest-meter-handle-error' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                style={{ left: `${interestMeta.hasScore ? interestMeta.pct : 50}%` }}
              />
            </div>
            <div className="interest-meter-reason">{interestMeta.reason}</div>
          </div>
        </div>
        <div className="detail-header-right">
          <div className="spam-meter detail-header-spam">
            <div className="spam-meter-header">
              <span className="spam-meter-label">Spam likelihood</span>
              <span className="spam-meter-value">{spamMeta.display}</span>
            </div>
            <div className={spamMeta.trackClass}>
              <div
                className={['spam-meter-handle', spamMeta.hasData ? '' : 'spam-meter-handle-empty']
                  .filter(Boolean)
                  .join(' ')}
                style={{ left: `${spamMeta.hasData ? spamMeta.pct : 50}%` }}
              />
            </div>
            <div className="spam-meter-reason">{spamMeta.reason}</div>
          </div>
          <button
            type="button"
            className="close-detail"
            onClick={handleClose}
            aria-label="Close details"
          >
            ×
          </button>
        </div>
      </div>

      <div className={buttonBarClass}>
        <button type="button" className="detail-action-button" onClick={handleSummarise} disabled={controlsDisabled}>
          Summarise Article
        </button>
        <button type="button" className="detail-action-button" onClick={handleAnalyse} disabled={controlsDisabled}>
          Analyse Post
        </button>
        <button type="button" className="detail-action-button" onClick={handleClear} disabled={controlsDisabled}>
          Clear Analysis
        </button>
      </div>

      <div className="detail-meta-row">
        <div className="detail-meta-entry">
          <span className="detail-meta-label">Post ID</span>
          <span className="detail-meta-value">{detail.post.post_id}</span>
        </div>
        <div className="detail-meta-entry">
          <span className="detail-meta-label">Scraped</span>
          <span className="detail-meta-value">{metaValue(detail.post.scraped_at)}</span>
        </div>
      </div>

      <div className="detail-scroll">
      <div className="detail-grid">
        <div className="detail-article-panel">
          {chairmanPlainText && (
            <section className="chairman-highlight">
              <div className="chairman-highlight-header">
                <span className="chairman-highlight-label">Chairman Verdict</span>
                {chairmanTags.length > 0 && (
                  <div className="chairman-highlight-tags">
                    {chairmanTags.map((tag) => (
                      <span key={tag.key} className={tag.className}>
                        {tag.label}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <p className="chairman-highlight-text">{chairmanPlainText}</p>
              {chairmanMetaItems.length > 0 && (
                <ul className="chairman-highlight-meta">
                  {chairmanMetaItems.map((item) => (
                    <li key={item.label}>
                      <span className="meta-label">{item.label}</span>
                      <span className="meta-value">{item.value}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {summarySections.length > 0 && (
            <div className="article-summary">
              {summarySections.map((section) => (
                <section key={section.key} className="article-summary-section">
                    <h3>{section.title}</h3>
                    <ul className="article-summary-list">
                      {section.items.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </section>
                ))}
              </div>
            )}

            <PriceWindowChart
              className="article-price-window"
              tickers={tickers}
              selectedTicker={selectedTicker}
              onSelectTicker={onSelectTicker}
              articleId={articleId}
              centerTimestamp={centerTimestamp}
              technical={technical ?? null}
            />

            <section className="article-section">
              <header className="article-section-header">
                <button
                  type="button"
                  className="collapse-toggle article-collapse-toggle"
                  onClick={toggleArticleCollapsed}
                  aria-expanded={!isArticleCollapsed}
                  aria-controls="article-content"
                >
                  <span className="collapse-toggle-arrow" aria-hidden="true">
                    ▸
                  </span>
                  <div className="article-section-meta">
                    <h3>Article</h3>
                    {detail.post.url && <div className="mono small muted">{detail.post.url}</div>}
                  </div>
                </button>
              </header>
              {!isArticleCollapsed && (
                <div
                  className="article-text"
                  aria-label="Article content"
                  id="article-content"
                >
                  {articleParagraphs}
                </div>
              )}
            </section>

            {researchParagraphs.length > 0 && (
              <section className="article-section article-research">
                <header className="article-section-header">
                  <div className="article-section-meta">
                    <h3>Researcher Output</h3>
                    {researchUpdatedAt && <div className="mono small muted">Updated {researchUpdatedAt}</div>}
                  </div>
                </header>
                <div className="article-text research-text" aria-label="Research output">
                  {researchParagraphs}
                </div>
              </section>
            )}

            {showAlternateSummary && (
              <section className="article-summary-section article-secondary-summary">
                <h3>Summary</h3>
                <ul className="article-summary-list">
                  {alternateSummaryBullets.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </section>
            )}
          </div>

          <div className="detail-side-panel">
            <section className="detail-interest">
              <h3>Interest score</h3>
              {interestMeta.status === 'ok' ? (
                <>
                  <p className="detail-interest-summary">
                    {interestSummaryText.length > 0 ? interestSummaryText : 'Interest drivers unspecified.'}
                  </p>
                  {interestInfoRows.length > 0 && (
                    <table className="stage-table stage-kv">
                      <tbody>
                        {interestInfoRows.map((row) => (
                          <tr key={row.label}>
                            <th>{row.label}</th>
                            <td>{row.value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {interestMetricsRows.length > 0 && (
                    <table className="stage-table stage-kv detail-interest-metrics">
                      <tbody>
                        {interestMetricsRows.map((row) => (
                          <tr key={row.label}>
                            <th>{row.label}</th>
                            <td>{row.value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </>
              ) : interestMeta.status === 'error' ? (
                <div className="detail-interest-error">{interestMeta.reason}</div>
              ) : (
                <div className="muted small">Interest score pending.</div>
              )}
            </section>

            <hr className="detail-side-divider" aria-hidden="true" />

            <section className="stage-panels">
              {STAGES.map((stage, index) => {
                const isCollapsible = COLLAPSIBLE_STAGES.has(stage)
                const isCollapsed = isCollapsible ? collapsedStages[stage] ?? false : false
                const bodyId = `stage-panel-${stage}`

                return (
                  <Fragment key={stage}>
                    {index > 0 && <hr className="detail-side-divider" aria-hidden="true" />}
                    <article
                      className={['stage-panel', isCollapsed ? 'stage-panel-collapsed' : '']
                        .filter(Boolean)
                        .join(' ')}
                    >
                      <header className="stage-panel-header">
                        {isCollapsible ? (
                          <button
                            type="button"
                            className="collapse-toggle stage-panel-toggle"
                            onClick={() => toggleStageCollapsed(stage)}
                            aria-expanded={!isCollapsed}
                            aria-controls={bodyId}
                          >
                            <span className="collapse-toggle-arrow" aria-hidden="true">
                              ▸
                            </span>
                            <span className="stage-panel-title">{formatStageLabel(stage)}</span>
                          </button>
                        ) : (
                          <h3>{formatStageLabel(stage)}</h3>
                        )}
                      </header>
                      {!isCollapsible || !isCollapsed ? (
                        <>
                          <div className="stage-panel-divider" aria-hidden="true" />
                          <div className="stage-panel-body" id={bodyId}>
                            <StageViewer
                              name={stage}
                              payload={stagePayloads[stage]}
                              ticker={activeResearchTicker}
                            />
                          </div>
                        </>
                      ) : null}
                    </article>
                  </Fragment>
                )
              })}
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
