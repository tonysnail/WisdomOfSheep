export const clampSpamPct = (value: number) => Math.min(100, Math.max(0, value))

export const spamClassForPct = (pct: number | null) => {
  if (pct == null) return 'spam-meter-track'
  if (pct >= 70) return 'spam-meter-track spam-meter-track-high'
  if (pct >= 40) return 'spam-meter-track spam-meter-track-medium'
  return 'spam-meter-track spam-meter-track-low'
}
