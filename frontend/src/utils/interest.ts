import type { InterestRecord } from '../types'

export type InterestDisplay = {
  status: 'ok' | 'error' | 'pending'
  hasScore: boolean
  pct: number
  displayValue: string
  valueText: string
  trackClass: string
  reason: string
  label?: string
  ticker?: string | null
  message?: string
}

export const clampInterestScore = (value: number) => Math.min(100, Math.max(0, value))

export function interestTrackClass(score: number | null, status: InterestDisplay['status']): string {
  if (status === 'error') return 'interest-meter-track-error'
  if (score == null || Number.isNaN(score)) return 'interest-meter-track-neutral'
  if (score >= 75) return 'interest-meter-track-high'
  if (score >= 45) return 'interest-meter-track-medium'
  return 'interest-meter-track-low'
}

export function buildInterestDisplay(record?: InterestRecord | null): InterestDisplay {
  const rawStatus = typeof record?.status === 'string' ? record.status.toLowerCase() : ''
  const status: InterestDisplay['status'] = rawStatus === 'ok' || rawStatus === 'error' ? (rawStatus as InterestDisplay['status']) : 'pending'

  if (status === 'ok') {
    const rawScore = typeof record?.interest_score === 'number' ? clampInterestScore(record.interest_score) : null
    const hasScore = rawScore != null
    const track = interestTrackClass(rawScore, 'ok')
    const displayValue = hasScore ? `${Math.round(rawScore)}%` : '—'
    const label = typeof record?.interest_label === 'string' ? record.interest_label.trim() : ''
    const reason = typeof record?.interest_why === 'string' && record.interest_why.trim().length > 0 ? record.interest_why.trim() : 'Interest drivers unspecified'
    const valueText = label ? `${displayValue} · ${label}` : displayValue
    return {
      status: 'ok',
      hasScore,
      pct: rawScore ?? 0,
      displayValue,
      valueText,
      trackClass: track,
      reason,
      label: label || undefined,
      ticker: record?.ticker ?? null,
    }
  }

  if (status === 'error') {
    const message = typeof record?.error_message === 'string' && record.error_message.trim().length > 0
      ? record.error_message.trim()
      : typeof record?.error_code === 'string' && record.error_code.trim().length > 0
        ? record.error_code.trim()
        : 'Interest score unavailable'
    return {
      status: 'error',
      hasScore: false,
      pct: 0,
      displayValue: '—',
      valueText: 'Unavailable',
      trackClass: 'interest-meter-track-error',
      reason: message,
      message,
      ticker: record?.ticker ?? null,
    }
  }

  return {
    status: 'pending',
    hasScore: false,
    pct: 0,
    displayValue: '—',
    valueText: 'Pending',
    trackClass: 'interest-meter-track-neutral',
    reason: 'Interest score pending',
    ticker: record?.ticker ?? null,
  }
}
