import { useEffect, useMemo, useState } from 'react'

import { getStockWindow, type StockWindowResponse, type StockWindowRow } from '../api'
import type { ResearchTechnical } from '../types'

const CHART_WIDTH = 460
const CHART_PADDING = { top: 18, right: 26, bottom: 48, left: 64 }
const PRICE_AREA_HEIGHT = 190
const PANEL_GAP = 18
const RSI_AREA_HEIGHT = 64
const MACD_AREA_HEIGHT = 76
const TOTAL_INNER_HEIGHT = PRICE_AREA_HEIGHT + PANEL_GAP + RSI_AREA_HEIGHT + PANEL_GAP + MACD_AREA_HEIGHT
const CHART_HEIGHT = TOTAL_INNER_HEIGHT + CHART_PADDING.top + CHART_PADDING.bottom
const CHART_INNER_WIDTH = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right
const CENTER_RATIO = 2 / 3
const DAY_MS = 24 * 60 * 60 * 1000
const FUTURE_WINDOW_DAYS = 30
const PREVIEW_WINDOW_DAYS = 2

const MOVING_AVERAGE_CONFIG = [
  { period: 20, color: '#f3ac43' },
  { period: 50, color: '#48d6c4' },
  { period: 200, color: '#ff6b80' },
]

type ChartPoint = {
  timestamp: string
  close: number
}

type ChartRow = {
  timestamp: string
  open: number | null
  high: number | null
  low: number | null
  close: number | null
  volume: number | null
}

type LineShape = {
  line: string
  area: string
  baseline: string
  min: number
  max: number
  startX: number
  endX: number
}

type SupportResistanceLine = {
  level: number
  y: number
  x1: number
  x2: number
  isNearest: boolean
  timestamp?: string
}

export type PriceWindowChartProps = {
  tickers: string[]
  selectedTicker: string | null
  onSelectTicker: (ticker: string | null) => void
  articleId: string | null
  centerTimestamp: string | null
  technical?: ResearchTechnical | null
  className?: string
}

function computeSMA(values: number[], window: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null)
  if (window <= 1) {
    return values.map((value) => (Number.isFinite(value) ? value : null))
  }
  let sum = 0
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i]
    if (i >= window) {
      sum -= values[i - window]
    }
    if (i >= window - 1) {
      out[i] = sum / window
    }
  }
  return out
}

function computeRollingStd(values: number[], window: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null)
  if (window <= 1) {
    return out
  }
  let sum = 0
  let sumSq = 0
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i]
    sum += value
    sumSq += value * value
    if (i >= window) {
      const old = values[i - window]
      sum -= old
      sumSq -= old * old
    }
    if (i >= window - 1) {
      const mean = sum / window
      const variance = Math.max(sumSq / window - mean * mean, 0)
      out[i] = Math.sqrt(variance)
    }
  }
  return out
}

function computeRSI(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null)
  if (values.length === 0 || period <= 0) return out

  let avgGain = 0
  let avgLoss = 0
  for (let i = 1; i < values.length; i += 1) {
    const delta = values[i] - values[i - 1]
    const gain = Math.max(delta, 0)
    const loss = Math.max(-delta, 0)

    if (i <= period) {
      avgGain += gain
      avgLoss += loss
      if (i === period) {
        avgGain /= period
        avgLoss /= period
        const rs = avgLoss === 0 ? Number.POSITIVE_INFINITY : avgGain / avgLoss
        out[i] = 100 - 100 / (1 + rs)
      }
      continue
    }

    avgGain = (avgGain * (period - 1) + gain) / period
    avgLoss = (avgLoss * (period - 1) + loss) / period
    const rs = avgLoss === 0 ? Number.POSITIVE_INFINITY : avgGain / avgLoss
    out[i] = 100 - 100 / (1 + rs)
  }

  return out
}

function computeMFI(rows: ChartRow[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(rows.length).fill(null)
  if (period <= 0) return out

  const typicalPrices: number[] = []
  const rawMoneyFlow: number[] = []

  rows.forEach((row) => {
    if (row.high != null && row.low != null && row.close != null && row.volume != null) {
      const tp = (row.high + row.low + row.close) / 3
      typicalPrices.push(tp)
      rawMoneyFlow.push(tp * row.volume)
    } else {
      typicalPrices.push(Number.NaN)
      rawMoneyFlow.push(0)
    }
  })

  for (let i = period; i < rows.length; i += 1) {
    let positive = 0
    let negative = 0
    for (let j = i - period + 1; j <= i; j += 1) {
      const current = typicalPrices[j]
      const prev = typicalPrices[j - 1]
      if (!Number.isFinite(current) || !Number.isFinite(prev)) continue
      const flow = rawMoneyFlow[j]
      if (current > prev) positive += flow
      else if (current < prev) negative += flow
    }
    if (positive === 0 && negative === 0) {
      out[i] = 50
    } else if (negative === 0) {
      out[i] = 100
    } else {
      const moneyRatio = positive / negative
      out[i] = 100 - 100 / (1 + moneyRatio)
    }
  }

  return out
}

function computeMACD(values: number[]) {
  const ema = (period: number) => {
    const k = 2 / (period + 1)
    let prev = values[0]
    const series = values.map((value, idx) => {
      if (idx === 0) {
        prev = value
        return value
      }
      const next = value * k + prev * (1 - k)
      prev = next
      return next
    })
    return series
  }

  const fast = ema(12)
  const slow = ema(26)
  const line = fast.map((value, idx) => value - slow[idx])
  const signal = (() => {
    const k = 2 / (9 + 1)
    let prev = line[0]
    return line.map((value, idx) => {
      if (idx === 0) {
        prev = value
        return value
      }
      const next = value * k + prev * (1 - k)
      prev = next
      return next
    })
  })()
  const hist = line.map((value, idx) => value - signal[idx])
  return { line, signal, hist }
}

function computeOBV(rows: ChartRow[]) {
  const series: (number | null)[] = new Array(rows.length).fill(null)
  let prevClose: number | null = null
  let obv = 0
  rows.forEach((row, idx) => {
    if (row.close == null || row.volume == null || !Number.isFinite(row.close) || !Number.isFinite(row.volume)) {
      series[idx] = null
      return
    }
    if (prevClose == null) {
      prevClose = row.close
      series[idx] = obv
      return
    }
    if (row.close > prevClose) obv += row.volume
    else if (row.close < prevClose) obv -= row.volume
    series[idx] = obv
    prevClose = row.close
  })
  return series
}

function buildLineShape(points: ChartPoint[], xPositions: number[], height: number): LineShape | null {
  if (points.length === 0 || points.length !== xPositions.length) return null

  const values = points.map((point) => point.close)
  const max = Math.max(...values)
  const min = Math.min(...values)
  const range = max - min || 1

  const commands = points.map((point, index) => {
    const x = xPositions[index]
    const y = CHART_PADDING.top + height - ((point.close - min) / range) * height
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  })

  const line = commands.join(' ')
  const startX = xPositions[0] ?? CHART_PADDING.left
  const endX = xPositions[xPositions.length - 1] ?? startX
  const bottom = (CHART_PADDING.top + height).toFixed(1)
  const area = `${line} L${endX.toFixed(1)},${bottom} L${startX.toFixed(1)},${bottom} Z`
  const baseline = `M${startX.toFixed(1)},${bottom} L${endX.toFixed(1)},${bottom}`

  return { line, area, baseline, min, max, startX, endX }
}

const formatValue = (value: number | null, decimals = 1) =>
  value != null && Number.isFinite(value) ? value.toFixed(decimals) : '—'

export default function PriceWindowChart({
  tickers,
  selectedTicker,
  onSelectTicker,
  articleId,
  centerTimestamp,
  technical = null,
  className,
}: PriceWindowChartProps) {
  const [windowMeta, setWindowMeta] = useState<StockWindowResponse | null>(null)
  const [chartRows, setChartRows] = useState<ChartRow[]>([])
  const [chartPoints, setChartPoints] = useState<ChartPoint[]>([])
  const [seriesError, setSeriesError] = useState<string | null>(null)
  const [seriesLoading, setSeriesLoading] = useState(false)
  const [showFutureData, setShowFutureData] = useState(true)

  useEffect(() => {
    let cancelled = false

    if (!selectedTicker) {
      setWindowMeta(null)
      setChartRows([])
      setChartPoints([])
      setSeriesError(null)
      setSeriesLoading(false)
      return () => {
        cancelled = true
      }
    }

    setSeriesLoading(true)
    setSeriesError(null)

    getStockWindow({ ticker: selectedTicker, center: centerTimestamp ?? undefined })
      .then((res) => {
        if (cancelled) return
        setWindowMeta(res)
        const rows: StockWindowRow[] = Array.isArray(res.data) ? res.data : []
        const toNumber = (value: unknown): number | null => {
          if (typeof value === 'number' && Number.isFinite(value)) return value
          if (typeof value === 'string') {
            const num = Number.parseFloat(value)
            return Number.isFinite(num) ? num : null
          }
          return null
        }

        const parsedRows: ChartRow[] = rows
          .map((row) => {
            const timestamp = typeof row?.timestamp === 'string' ? row.timestamp : null
            if (!timestamp) return null
            return {
              timestamp,
              open: toNumber(row?.Open ?? null),
              high: toNumber(row?.High ?? null),
              low: toNumber(row?.Low ?? null),
              close: toNumber(row?.Close ?? null),
              volume: toNumber(row?.Volume ?? null),
            }
          })
          .filter((row): row is ChartRow => row !== null)

        parsedRows.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())

        const validRows = parsedRows
          .filter((row) => typeof row.close === 'number' && Number.isFinite(row.close))
          .map((row) => ({ ...row, close: row.close as number }))

        setChartRows(validRows)

        const points = validRows.map((row) => ({ timestamp: row.timestamp, close: row.close as number }))

        setChartPoints(points)

        if (points.length === 0) {
          setSeriesError(res.note ?? 'No price data returned for the selected window.')
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setWindowMeta(null)
        setChartRows([])
        setChartPoints([])
        setSeriesError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) {
          setSeriesLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [selectedTicker, centerTimestamp, articleId])

  const technicalByTool = useMemo(() => {
    const map = new Map<string, Record<string, any>>()
    if (!technical || !Array.isArray(technical.results)) return map
    technical.results.forEach((entry) => {
      if (!entry || typeof entry !== 'object') return
      const record = entry as Record<string, any>
      const tool = typeof record.tool === 'string' ? record.tool : ''
      if (!tool) return
      const status = typeof record.status === 'string' ? record.status : 'ok'
      if (status === 'error') return
      const result = record.result
      if (result && typeof result === 'object') {
        map.set(tool, result as Record<string, any>)
      }
    })
    return map
  }, [technical])

  const trendStrength = technicalByTool.get('trend_strength') ?? null
  const volatilityState = technicalByTool.get('volatility_state') ?? null
  const supportResistance = technicalByTool.get('support_resistance_check') ?? null
  const bollingerResult = technicalByTool.get('bollinger_breakout_scan') ?? null
  const obvResult = technicalByTool.get('obv_trend') ?? null
  const mfiResult = technicalByTool.get('mfi_flow') ?? null

  const effectiveCenterTimestamp = useMemo(
    () => centerTimestamp ?? windowMeta?.center_utc ?? null,
    [centerTimestamp, windowMeta?.center_utc],
  )

  const closes = useMemo(() => chartRows.map((row) => row.close as number), [chartRows])
  const chartPointTimes = useMemo(
    () => chartPoints.map((point) => new Date(point.timestamp).getTime()),
    [chartPoints],
  )

  const baseOverlayLimitIndex = useMemo(() => {
    if (chartPoints.length === 0) return -1
    if (!effectiveCenterTimestamp) return chartPoints.length - 1
    const cutoffMs = new Date(effectiveCenterTimestamp).getTime()
    if (Number.isNaN(cutoffMs)) return chartPoints.length - 1
    let idx = -1
    for (let i = 0; i < chartPoints.length; i += 1) {
      const ms = new Date(chartPoints[i].timestamp).getTime()
      if (!Number.isNaN(ms) && ms <= cutoffMs) {
        idx = i
      }
    }
    return idx < 0 ? chartPoints.length - 1 : idx
  }, [chartPoints, effectiveCenterTimestamp])

  const visibleRange = useMemo(() => {
    if (chartPoints.length === 0) return { start: 0, end: 0 }
    if (showFutureData) {
      if (!effectiveCenterTimestamp) return { start: 0, end: chartPoints.length }
      const centerMs = new Date(effectiveCenterTimestamp).getTime()
      if (!Number.isFinite(centerMs)) return { start: 0, end: chartPoints.length }
      const pastCutoff = centerMs - PREVIEW_WINDOW_DAYS * DAY_MS
      const futureCutoff = centerMs + FUTURE_WINDOW_DAYS * DAY_MS

      let startCandidate = chartPointTimes.findIndex((ms) => Number.isFinite(ms) && ms >= pastCutoff)
      if (startCandidate === -1) {
        startCandidate = chartPoints.length - 1
      }

      let startIndex = Math.max(0, startCandidate)
      if (startCandidate > 0) {
        const prevMs = chartPointTimes[startCandidate - 1]
        if (
          typeof prevMs === 'number' &&
          Number.isFinite(prevMs) &&
          centerMs - prevMs <= PREVIEW_WINDOW_DAYS * DAY_MS
        ) {
          startIndex = startCandidate - 1
        }
      }

      let endIndex = chartPointTimes.findIndex((ms) => Number.isFinite(ms) && ms > futureCutoff)
      if (endIndex === -1) {
        endIndex = chartPoints.length
      }

      if (endIndex <= startIndex) {
        endIndex = Math.min(chartPoints.length, startIndex + 1)
      }

      return {
        start: Math.max(0, startIndex),
        end: Math.min(chartPoints.length, Math.max(startIndex + 1, endIndex)),
      }
    }

    if (baseOverlayLimitIndex < 0) {
      return { start: 0, end: 0 }
    }

    return { start: 0, end: Math.min(baseOverlayLimitIndex + 1, chartPoints.length) }
  }, [chartPoints, chartPointTimes, showFutureData, effectiveCenterTimestamp, baseOverlayLimitIndex])

  const visibleStartIndex = visibleRange.start
  const visibleEndIndex = visibleRange.end

  const overlayLimitIndex = useMemo(() => {
    if (visibleEndIndex <= visibleStartIndex) return -1
    if (showFutureData) {
      return visibleEndIndex - visibleStartIndex - 1
    }
    const limit = Math.min(baseOverlayLimitIndex, visibleEndIndex - 1)
    if (limit < visibleStartIndex) {
      return visibleEndIndex - visibleStartIndex - 1
    }
    return limit - visibleStartIndex
  }, [showFutureData, baseOverlayLimitIndex, visibleStartIndex, visibleEndIndex])

  const overlayAbsoluteEndIndex = overlayLimitIndex >= 0 ? visibleStartIndex + overlayLimitIndex : -1

  const visibleChartPoints = useMemo(
    () => chartPoints.slice(visibleStartIndex, visibleEndIndex),
    [chartPoints, visibleEndIndex, visibleStartIndex],
  )

  const pointCount = visibleChartPoints.length

  const centerPosition = useMemo(() => {
    if (!effectiveCenterTimestamp || visibleChartPoints.length === 0) return null
    const centerMs = new Date(effectiveCenterTimestamp).getTime()
    if (!Number.isFinite(centerMs)) return null
    const validPoints = visibleChartPoints
      .map((point, index) => {
        const ms = new Date(point.timestamp).getTime()
        return Number.isFinite(ms) ? { index, ms } : null
      })
      .filter((entry): entry is { index: number; ms: number } => entry !== null)
    if (validPoints.length === 0) return null
    const first = validPoints[0]
    const last = validPoints[validPoints.length - 1]
    if (centerMs <= first.ms) return first.index
    if (centerMs >= last.ms) return last.index
    for (let i = 0; i < validPoints.length - 1; i += 1) {
      const current = validPoints[i]
      const next = validPoints[i + 1]
      if (centerMs < current.ms || centerMs > next.ms) continue
      const span = next.ms - current.ms
      const ratio = span > 0 ? (centerMs - current.ms) / span : 0
      const clamped = Math.min(1, Math.max(0, ratio))
      return current.index + clamped * (next.index - current.index)
    }
    return last.index
  }, [effectiveCenterTimestamp, visibleChartPoints])

  const { xPositions, xForIndex } = useMemo(() => {
    const count = visibleChartPoints.length
    if (count === 0) {
      const fallback = [CHART_PADDING.left + CHART_INNER_WIDTH / 2]
      return {
        xPositions: fallback,
        xForIndex: () => fallback[0],
      }
    }

    let center = centerPosition
    if (center == null || !Number.isFinite(center)) {
      center = count - 1
    }
    center = Math.min(Math.max(center, 0), Math.max(count - 1, 0))

    const leftWidth = CHART_INNER_WIDTH * CENTER_RATIO
    const rightWidth = CHART_INNER_WIDTH - leftWidth

    const xPositions: number[] = new Array(count)

    if (count === 1) {
      xPositions[0] = CHART_PADDING.left + CHART_INNER_WIDTH / 2
    } else if (center <= 0 || rightWidth <= 0) {
      for (let i = 0; i < count; i += 1) {
        const ratio = i / (count - 1)
        xPositions[i] = CHART_PADDING.left + ratio * CHART_INNER_WIDTH
      }
    } else if (center >= count - 1 || leftWidth <= 0) {
      for (let i = 0; i < count; i += 1) {
        const ratio = i / (count - 1)
        xPositions[i] = CHART_PADDING.left + ratio * CHART_INNER_WIDTH
      }
    } else {
      for (let i = 0; i < count; i += 1) {
        if (i <= center) {
          const denom = center === 0 ? 1 : center
          const portion = denom === 0 ? 0 : i / denom
          xPositions[i] = CHART_PADDING.left + portion * leftWidth
        } else {
          const denom = count - 1 - center
          const portion = denom === 0 ? 1 : (i - center) / denom
          xPositions[i] = CHART_PADDING.left + leftWidth + portion * rightWidth
        }
      }
    }

    const xForIndex = (index: number) => {
      if (xPositions.length === 0) return CHART_PADDING.left + CHART_INNER_WIDTH / 2
      if (index <= 0) return xPositions[0]
      if (index >= xPositions.length - 1) return xPositions[xPositions.length - 1]
      const lower = Math.floor(index)
      const upper = Math.ceil(index)
      if (lower === upper) return xPositions[lower]
      const ratio = (index - lower) / (upper - lower)
      return xPositions[lower] + ratio * (xPositions[upper] - xPositions[lower])
    }

    return { xPositions, xForIndex }
  }, [centerPosition, visibleChartPoints])

  const lineShape = useMemo(() => buildLineShape(visibleChartPoints, xPositions, PRICE_AREA_HEIGHT), [visibleChartPoints, xPositions])

  const priceAreaBottom = CHART_PADDING.top + PRICE_AREA_HEIGHT
  const rsiTop = priceAreaBottom + PANEL_GAP
  const rsiBottom = rsiTop + RSI_AREA_HEIGHT
  const macdTop = rsiBottom + PANEL_GAP
  const chartBottom = CHART_PADDING.top + TOTAL_INNER_HEIGHT

  const priceValueToY = (value: number) => {
    if (!lineShape) return priceAreaBottom
    const range = lineShape.max - lineShape.min || 1
    return CHART_PADDING.top + PRICE_AREA_HEIGHT - ((value - lineShape.min) / range) * PRICE_AREA_HEIGHT
  }

  const rsiPeriod = 14
  const mfiPeriod = Math.max(2, Math.round(Number(mfiResult?.period ?? 14)))
  const bollingerPeriod = Math.max(2, Math.round(Number(bollingerResult?.period ?? 20)))
  const bollingerStd = Number.isFinite(Number(bollingerResult?.num_std)) ? Number(bollingerResult?.num_std) : 2
  const trendLookback = Math.max(2, Math.round(Number(trendStrength?.lookback_days ?? 30)))

  const trendMeta = useMemo(() => {
    if (!trendStrength) return null
    const directionRaw = (trendStrength as Record<string, any>).direction
    const direction = typeof directionRaw === 'string' ? directionRaw : null
    const strengthRaw = (trendStrength as Record<string, any>).strength
    const strength =
      typeof strengthRaw === 'number'
        ? strengthRaw
        : typeof strengthRaw === 'string'
          ? Number.parseFloat(strengthRaw)
          : null
    const slopeRaw = (trendStrength as Record<string, any>).slope_pct_per_day
    const slope =
      typeof slopeRaw === 'number'
        ? slopeRaw
        : typeof slopeRaw === 'string'
          ? Number.parseFloat(slopeRaw)
          : null
    const r2Raw = (trendStrength as Record<string, any>).r2
    const r2 =
      typeof r2Raw === 'number'
        ? r2Raw
        : typeof r2Raw === 'string'
          ? Number.parseFloat(r2Raw)
          : null

    const formatSlope = (value: number | null) => {
      if (value == null || !Number.isFinite(value)) return null
      const sign = value >= 0 ? '+' : ''
      return `${sign}${value.toFixed(2)}%/day`
    }

    const formatR2 = (value: number | null) => {
      if (value == null || !Number.isFinite(value)) return null
      return `R² ${value.toFixed(2)}`
    }

    const color = direction === 'down' ? '#ff6b80' : direction === 'up' ? '#48d6c4' : '#f3ac43'
    const directionLabel = direction ? `${direction[0].toUpperCase()}${direction.slice(1)}` : null

    return {
      direction,
      directionLabel,
      strength,
      slope,
      slopeLabel: formatSlope(slope),
      r2,
      r2Label: formatR2(r2),
      color,
      lookback: trendLookback,
    }
  }, [trendStrength, trendLookback])

  const movingAveragePaths = useMemo(() => {
    if (!lineShape || overlayLimitIndex < 1) return [] as { period: number; color: string; path: string }[]
    return MOVING_AVERAGE_CONFIG.map((cfg) => {
      const series = computeSMA(closes, cfg.period)
      const commands: string[] = []
      for (let i = 0; i <= overlayLimitIndex; i += 1) {
        const absoluteIndex = visibleStartIndex + i
        if (absoluteIndex >= series.length) break
        const value = series[absoluteIndex]
        if (value == null) continue
        const x = xForIndex(i)
        const y = priceValueToY(value)
        commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
      }
      if (commands.length < 2) return null
      return { period: cfg.period, color: cfg.color, path: commands.join(' ') }
    }).filter((entry): entry is { period: number; color: string; path: string } => entry !== null)
  }, [closes, overlayLimitIndex, lineShape, visibleStartIndex, xForIndex])

  const bollingerBands = useMemo(() => {
    if (!lineShape || overlayLimitIndex < bollingerPeriod - 1) return null
    const sma = computeSMA(closes, bollingerPeriod)
    const std = computeRollingStd(closes, bollingerPeriod)
    const upper: { x: number; y: number }[] = []
    const lower: { x: number; y: number }[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex < bollingerPeriod - 1) continue
      if (absoluteIndex >= closes.length) break
      const mean = sma[absoluteIndex]
      const deviation = std[absoluteIndex]
      if (mean == null || deviation == null) continue
      const upperValue = mean + bollingerStd * deviation
      const lowerValue = mean - bollingerStd * deviation
      upper.push({ x: xForIndex(i), y: priceValueToY(upperValue) })
      lower.push({ x: xForIndex(i), y: priceValueToY(lowerValue) })
    }
    if (upper.length < 2 || lower.length < 2) return null
    const upperPath = upper
      .map((point, idx) => `${idx === 0 ? 'M' : 'L'}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(' ')
    const lowerPath = lower
      .map((point, idx) => `${idx === 0 ? 'M' : 'L'}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(' ')
    const areaPath = `${upperPath} ${lower
      .slice()
      .reverse()
      .map((point) => `L${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(' ')} Z`
    return { areaPath, upperPath, lowerPath }
  }, [closes, overlayLimitIndex, lineShape, bollingerPeriod, bollingerStd, visibleStartIndex, xForIndex])

  const trendLinePath = useMemo(() => {
    if (!lineShape || overlayLimitIndex < 1 || closes.length === 0) return null
    if (overlayAbsoluteEndIndex < 0) return null
    const endIndex = overlayAbsoluteEndIndex
    const startIndex = Math.max(0, endIndex - trendLookback + 1)
    if (endIndex - startIndex < 1) return null
    const slice = closes.slice(startIndex, endIndex + 1)
    if (slice.some((value) => !Number.isFinite(value))) return null
    const yValues = slice.map((value) => Math.log(value))
    const xValues = yValues.map((_, idx) => idx)
    const n = xValues.length
    const sumX = xValues.reduce((acc, value) => acc + value, 0)
    const sumY = yValues.reduce((acc, value) => acc + value, 0)
    const sumXY = xValues.reduce((acc, value, idx) => acc + value * yValues[idx], 0)
    const sumXX = xValues.reduce((acc, value) => acc + value * value, 0)
    const denom = n * sumXX - sumX * sumX
    if (denom === 0) return null
    const slope = (n * sumXY - sumX * sumY) / denom
    const intercept = (sumY - slope * sumX) / n
    const coords: { x: number; y: number }[] = []
    for (let i = 0; i < slice.length; i += 1) {
      const price = Math.exp(intercept + slope * i)
      const relativeIndex = startIndex + i - visibleStartIndex
      if (relativeIndex < 0 || relativeIndex >= visibleChartPoints.length) continue
      coords.push({ x: xForIndex(relativeIndex), y: priceValueToY(price) })
    }
    if (coords.length < 2) return null
    return coords
      .map((point, idx) => `${idx === 0 ? 'M' : 'L'}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(' ')
  }, [closes, lineShape, overlayAbsoluteEndIndex, overlayLimitIndex, trendLookback, visibleChartPoints.length, visibleStartIndex, xForIndex])

  const rsiSeries = useMemo(() => computeRSI(closes, rsiPeriod), [closes, rsiPeriod])
  const mfiSeries = useMemo(() => computeMFI(chartRows, mfiPeriod), [chartRows, mfiPeriod])
  const macdSeries = useMemo(() => computeMACD(closes), [closes])
  const obvSeries = useMemo(() => computeOBV(chartRows), [chartRows])

  const macdLine = macdSeries.line
  const macdSignal = macdSeries.signal
  const macdHist = macdSeries.hist

  const latestIndex = overlayAbsoluteEndIndex >= 0 ? Math.min(overlayAbsoluteEndIndex, closes.length - 1) : -1
  const latestClose = latestIndex >= 0 ? closes[latestIndex] ?? null : null
  const latestRsi = latestIndex >= 0 ? rsiSeries[latestIndex] ?? null : null
  const latestMfi = latestIndex >= 0 ? mfiSeries[latestIndex] ?? null : null
  const latestMacdLine = latestIndex >= 0 ? macdLine[latestIndex] ?? null : null
  const latestMacdSignal = latestIndex >= 0 ? macdSignal[latestIndex] ?? null : null

  const rsiPath = useMemo(() => {
    if (overlayLimitIndex < 1) return null
    const commands: string[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= rsiSeries.length) break
      const value = rsiSeries[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      const ratio = Math.min(1, Math.max(0, value / 100))
      const y = rsiTop + (1 - ratio) * RSI_AREA_HEIGHT
      const x = xForIndex(i)
      commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return commands.length < 2 ? null : commands.join(' ')
  }, [overlayLimitIndex, rsiSeries, visibleStartIndex, xForIndex])

  const mfiPath = useMemo(() => {
    if (overlayLimitIndex < mfiPeriod - 1) return null
    const commands: string[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= mfiSeries.length) break
      const value = mfiSeries[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      const ratio = Math.min(1, Math.max(0, value / 100))
      const y = rsiTop + (1 - ratio) * RSI_AREA_HEIGHT
      const x = xForIndex(i)
      commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return commands.length < 2 ? null : commands.join(' ')
  }, [mfiPeriod, mfiSeries, overlayLimitIndex, visibleStartIndex, xForIndex])

  const macdMetrics = useMemo(() => {
    if (overlayLimitIndex < 1) return null
    const values: number[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      const lineVal = macdLine[absoluteIndex]
      const sigVal = macdSignal[absoluteIndex]
      const histVal = macdHist[absoluteIndex]
      if (lineVal != null && Number.isFinite(lineVal)) values.push(lineVal)
      if (sigVal != null && Number.isFinite(sigVal)) values.push(sigVal)
      if (histVal != null && Number.isFinite(histVal)) values.push(histVal)
    }
    if (values.length === 0) return null
    const min = Math.min(...values)
    const max = Math.max(...values)
    let range = max - min
    if (range === 0) {
      range = Math.abs(max) > 0 ? Math.abs(max) * 2 : 1
    }
    return { min, max, range }
  }, [macdLine, macdSignal, macdHist, overlayLimitIndex, visibleStartIndex])

  const macdValueToY = (value: number) => {
    if (!macdMetrics) return macdTop + MACD_AREA_HEIGHT / 2
    const { min, range } = macdMetrics
    return macdTop + MACD_AREA_HEIGHT - ((value - min) / range) * MACD_AREA_HEIGHT
  }

  const macdLinePath = useMemo(() => {
    if (!macdMetrics || overlayLimitIndex < 1) return null
    const commands: string[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= macdLine.length) break
      const value = macdLine[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      const x = xForIndex(i)
      const y = macdValueToY(value)
      commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return commands.length < 2 ? null : commands.join(' ')
  }, [macdMetrics, macdLine, overlayLimitIndex, visibleStartIndex, xForIndex])

  const macdSignalPath = useMemo(() => {
    if (!macdMetrics || overlayLimitIndex < 1) return null
    const commands: string[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= macdSignal.length) break
      const value = macdSignal[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      const x = xForIndex(i)
      const y = macdValueToY(value)
      commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return commands.length < 2 ? null : commands.join(' ')
  }, [macdMetrics, macdSignal, overlayLimitIndex, visibleStartIndex, xForIndex])

  const macdHistogram = useMemo(() => {
    if (!macdMetrics || overlayLimitIndex < 1) return [] as { x: number; y: number; width: number; height: number; positive: boolean }[]
    const bars: { x: number; y: number; width: number; height: number; positive: boolean }[] = []
    const barWidth = Math.max(2, CHART_INNER_WIDTH / Math.max(pointCount, 40))
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= macdHist.length) break
      const value = macdHist[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      const xCenter = xForIndex(i)
      const x = xCenter - barWidth / 2
      const y = macdValueToY(Math.max(0, value))
      const zero = macdValueToY(0)
      const height = Math.abs(zero - macdValueToY(value))
      bars.push({ x, y: value >= 0 ? y : zero, width: barWidth, height, positive: value >= 0 })
    }
    return bars
  }, [macdMetrics, macdHist, overlayLimitIndex, pointCount, visibleStartIndex, xForIndex])

  const obvPath = useMemo(() => {
    if (overlayLimitIndex < 1) return null
    const values: number[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= obvSeries.length) break
      const value = obvSeries[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      values.push(value)
    }
    if (values.length < 2) return null
    const maxAbs = Math.max(...values.map((value) => Math.abs(value))) || 1
    const commands: string[] = []
    for (let i = 0; i <= overlayLimitIndex; i += 1) {
      const absoluteIndex = visibleStartIndex + i
      if (absoluteIndex >= obvSeries.length) break
      const value = obvSeries[absoluteIndex]
      if (value == null || Number.isNaN(value)) continue
      const normalized = value / maxAbs
      const y = macdTop + MACD_AREA_HEIGHT / 2 - normalized * (MACD_AREA_HEIGHT * 0.4)
      const x = xForIndex(i)
      commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return commands.length < 2 ? null : commands.join(' ')
  }, [obvSeries, overlayLimitIndex, visibleStartIndex, xForIndex])

  const timeDomain = useMemo(() => {
    const times = visibleChartPoints
      .map((point) => new Date(point.timestamp).getTime())
      .filter((ms) => Number.isFinite(ms))

    if (times.length === 0) {
      return null
    }

    const min = Math.min(...times)
    const max = Math.max(...times)
    return { min, max }
  }, [visibleChartPoints])

  const xAxisTicks = useMemo(() => {
    if (!timeDomain || !lineShape) return []

    const { min, max } = timeDomain
    if (!Number.isFinite(min) || !Number.isFinite(max)) return []

    const range = Math.max(max - min, 1)
    const tickCount = Math.min(6, Math.max(2, Math.round(CHART_INNER_WIDTH / 90)))

    const interval = windowMeta?.interval?.toLowerCase() ?? ''
    const showDateOnly = (() => {
      if (interval.includes('day') || interval.includes('d') || interval.includes('wk') || interval.includes('mo')) {
        return true
      }
      const dayMs = 24 * 60 * 60 * 1000
      return range >= 36 * 60 * 60 * 1000 || range >= dayMs
    })()

    const formatter = new Intl.DateTimeFormat('en-US', showDateOnly
      ? { month: 'short', day: '2-digit' }
      : { hour: '2-digit', minute: '2-digit', hour12: false })

    return new Array(tickCount).fill(0).map((_, idx) => {
      const ratio = tickCount === 1 ? 0 : idx / (tickCount - 1)
      const timestamp = min + ratio * range
      const x = lineShape.startX + ratio * (lineShape.endX - lineShape.startX)
      return {
        x,
        label: formatter.format(new Date(timestamp)),
      }
    })
  }, [timeDomain, windowMeta?.interval, lineShape])

  const yAxisTicks = useMemo(() => {
    if (!lineShape) return []

    const tickCount = 4
    const range = lineShape.max - lineShape.min
    const displayRange = range === 0 ? Math.max(Math.abs(lineShape.max), 1) : range

    let decimals = 2
    if (displayRange < 1) {
      decimals = 4
    } else if (displayRange < 5) {
      decimals = 3
    } else if (displayRange >= 50) {
      decimals = 1
    }

    const formatter = new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })

    return new Array(tickCount + 1).fill(0).map((_, idx) => {
      const ratio = idx / tickCount
      const value = range === 0 ? lineShape.max : lineShape.max - ratio * range
      const y = CHART_PADDING.top + ratio * PRICE_AREA_HEIGHT
      return {
        y,
        label: formatter.format(value),
      }
    })
  }, [lineShape])

  const indicatorX = useMemo(() => {
    if (centerPosition == null) return null
    return xForIndex(centerPosition)
  }, [centerPosition, xForIndex])

  const overlayRightX = useMemo(() => {
    if (showFutureData) return lineShape?.endX ?? CHART_PADDING.left + CHART_INNER_WIDTH
    if (indicatorX !== null) return indicatorX
    if (overlayLimitIndex >= 0) {
      return xForIndex(Math.min(overlayLimitIndex, Math.max(visibleChartPoints.length - 1, 0)))
    }
    return lineShape?.endX ?? CHART_PADDING.left + CHART_INNER_WIDTH
  }, [showFutureData, lineShape?.endX, indicatorX, overlayLimitIndex, visibleChartPoints.length, xForIndex])

  const supportResistanceLines = useMemo(() => {
    if (!supportResistance || overlayLimitIndex < 0 || !lineShape) {
      return { supports: [] as SupportResistanceLine[], resistances: [] as SupportResistanceLine[] }
    }

    const nearestSupportValue = (() => {
      const value = (supportResistance as Record<string, any>).nearest_support
      if (typeof value === 'number') return value
      if (typeof value === 'string') {
        const num = Number.parseFloat(value)
        return Number.isFinite(num) ? num : null
      }
      return null
    })()

    const nearestResistanceValue = (() => {
      const value = (supportResistance as Record<string, any>).nearest_resistance
      if (typeof value === 'number') return value
      if (typeof value === 'string') {
        const num = Number.parseFloat(value)
        return Number.isFinite(num) ? num : null
      }
      return null
    })()

    const levels = (supportResistance as Record<string, any>).levels
    const supportEntries =
      levels && typeof levels === 'object' && Array.isArray((levels as Record<string, any>).support_levels)
        ? ((levels as Record<string, any>).support_levels as Record<string, any>[])
        : []
    const resistanceEntries =
      levels && typeof levels === 'object' && Array.isArray((levels as Record<string, any>).resistance_levels)
        ? ((levels as Record<string, any>).resistance_levels as Record<string, any>[])
        : []

    const buildLines = (entries: Record<string, any>[], nearest: number | null): SupportResistanceLine[] => {
      const lines: SupportResistanceLine[] = []
      entries.slice(-6).forEach((entry) => {
        const rawLevel = entry?.level
        const level =
          typeof rawLevel === 'number'
            ? rawLevel
            : typeof rawLevel === 'string'
              ? Number.parseFloat(rawLevel)
              : null
        if (level == null || !Number.isFinite(level)) return
        const y = priceValueToY(level)
        const ts = typeof entry?.timestamp === 'string' ? entry.timestamp : null
        const findIndexForDate = (date: string | null) => {
          if (!date) return null
          const iso = date.slice(0, 10)
          let idx = chartPoints.findIndex((pt) => pt.timestamp.slice(0, 10) === iso)
          if (idx < 0) {
            idx = chartPoints.findIndex((pt) => pt.timestamp.slice(0, 10) > iso)
            if (idx > 0) idx -= 1
          }
          if (idx < 0) return null
          if (visibleChartPoints.length === 0) return null
          const relative = idx - visibleStartIndex
          if (relative <= 0) return 0
          if (relative >= visibleChartPoints.length) {
            return Math.min(overlayLimitIndex, visibleChartPoints.length - 1)
          }
          if (relative > overlayLimitIndex) return overlayLimitIndex
          return relative
        }
        const idx = findIndexForDate(ts)
        const x1 = idx != null ? xForIndex(idx) : CHART_PADDING.left
        const x2 = Math.max(x1, overlayRightX)
        const isNearest = nearest != null && Math.abs(nearest - level) <= Math.max(0.15 * Math.abs(level), 0.02 * Math.abs(nearest))
        lines.push({ level, y, x1, x2, isNearest, timestamp: ts ?? undefined })
      })
      return lines
    }

    return {
      supports: buildLines(supportEntries, nearestSupportValue),
      resistances: buildLines(resistanceEntries, nearestResistanceValue),
    }
  }, [supportResistance, overlayLimitIndex, lineShape, overlayRightX, chartPoints, priceValueToY, visibleChartPoints.length, visibleStartIndex, xForIndex])

  const volatilityOverlay = useMemo(() => {
    if (!volatilityState) return null
    const state = typeof (volatilityState as Record<string, any>).state === 'string'
      ? ((volatilityState as Record<string, any>).state as string)
      : null
    const ratioRaw = (volatilityState as Record<string, any>).ratio
    const ratioValue =
      typeof ratioRaw === 'number'
        ? ratioRaw
        : typeof ratioRaw === 'string'
          ? Number.parseFloat(ratioRaw)
          : null
    const realizedRaw = (volatilityState as Record<string, any>).realized_vol_annual_pct
    const baselineRaw = (volatilityState as Record<string, any>).baseline_vol_annual_pct
    const realized =
      typeof realizedRaw === 'number'
        ? realizedRaw
        : typeof realizedRaw === 'string'
          ? Number.parseFloat(realizedRaw)
          : null
    const baseline =
      typeof baselineRaw === 'number'
        ? baselineRaw
        : typeof baselineRaw === 'string'
          ? Number.parseFloat(baselineRaw)
          : null
    let fill: string | null = 'rgba(94, 129, 172, 0.14)'
    if (state === 'elevated') fill = null
    else if (state === 'compressed') fill = 'rgba(76, 201, 240, 0.16)'
    return {
      fill,
      state,
      ratioText: ratioValue != null && Number.isFinite(ratioValue) ? `${ratioValue.toFixed(2)}×` : null,
      realized: realized != null && Number.isFinite(realized) ? realized.toFixed(1) : null,
      baseline: baseline != null && Number.isFinite(baseline) ? baseline.toFixed(1) : null,
    }
  }, [volatilityState])

  const obvColor = useMemo(() => {
    const trend = typeof obvResult?.trend === 'string' ? (obvResult?.trend as string) : null
    if (trend === 'down') return '#ff6b80'
    if (trend === 'flat') return '#f3ac43'
    return '#48d6c4'
  }, [obvResult])
  const obvTrendLabel = typeof obvResult?.trend === 'string' ? (obvResult?.trend as string) : '—'

  const trendLineColor = trendMeta?.color ?? '#f0f6ff'
  const trendStrengthLine1 = trendMeta
    ? [
        trendMeta.directionLabel ? `Trend ${trendMeta.directionLabel}` : null,
        trendMeta.strength != null && Number.isFinite(trendMeta.strength)
          ? `Strength ${Math.round(trendMeta.strength)}`
          : null,
      ]
        .filter(Boolean)
        .join(' • ')
    : null
  const trendStrengthLine2 = trendMeta ? [trendMeta.slopeLabel, trendMeta.r2Label].filter(Boolean).join(' • ') : null

  const volatilityText = volatilityOverlay
    ? [
        'Volatility',
        volatilityOverlay.state ? String(volatilityOverlay.state) : null,
        volatilityOverlay.ratioText,
      ]
        .filter(Boolean)
        .join(' • ')
    : null
  const volatilityDetail = volatilityOverlay
    ? [
        volatilityOverlay.realized ? `RV ${volatilityOverlay.realized}%` : null,
        volatilityOverlay.baseline ? `Base ${volatilityOverlay.baseline}%` : null,
      ]
        .filter(Boolean)
        .join(' • ')
    : null

  const legendItems = useMemo(
    () =>
      [
        ...MOVING_AVERAGE_CONFIG.map((cfg) => ({
          key: `sma-${cfg.period}`,
          label: `SMA ${cfg.period}`,
          color: cfg.color,
          dash: '0',
        })),
        ...(bollingerBands
          ? [
              {
                key: 'bollinger',
                label: `Bollinger ${bollingerPeriod}`,
                color: 'rgba(148,196,255,0.85)',
                dash: '4 2',
              },
            ]
          : []),
        ...(trendLinePath
          ? [
              {
                key: 'trend',
                label: `Trend ${trendMeta?.lookback ?? trendLookback}d`,
                color: trendLineColor,
                dash: '6 3',
              },
            ]
          : []),
      ],
    [bollingerBands, bollingerPeriod, trendLinePath, trendLookback, trendLineColor, trendMeta?.lookback],
  )

  const closeLabel = formatValue(latestClose, 2)
  const rsiLabel = formatValue(latestRsi, 1)
  const mfiLabel = formatValue(latestMfi, 1)
  const macdLabel = formatValue(latestMacdLine, 3)
  const macdSignalLabel = formatValue(latestMacdSignal, 3)

  const centerLabel = useMemo(() => {
    if (!effectiveCenterTimestamp) return null
    const centerDate = new Date(effectiveCenterTimestamp)
    if (Number.isNaN(centerDate.getTime())) return null
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(centerDate)
  }, [effectiveCenterTimestamp])

  const infoItems = [
    closeLabel && `Close ${closeLabel}`,
    trendStrengthLine1,
    trendStrengthLine2,
    volatilityText && (volatilityDetail ? `${volatilityText} • ${volatilityDetail}` : volatilityText),
    centerLabel && `Article Date ${centerLabel}`,
  ].filter(Boolean) as string[]

  const secondaryInfoItems = [
    `RSI 14 ${rsiLabel}`,
    `MFI ${mfiLabel}`,
    `MACD ${macdLabel}`,
    `Signal ${macdSignalLabel}`,
    `OBV ${obvTrendLabel}`,
  ]

  const nearestSupport = supportResistanceLines.supports.find((line) => line.isNearest)
  const nearestResistance = supportResistanceLines.resistances.find((line) => line.isNearest)

  const supportResistanceItems: string[] = []
  if (nearestSupport) supportResistanceItems.push(`Support ${nearestSupport.level.toFixed(2)}`)
  if (nearestResistance) supportResistanceItems.push(`Resistance ${nearestResistance.level.toFixed(2)}`)

  return (
    <section className={['price-window', className].filter(Boolean).join(' ')}>
      <header className="price-window-header">
        <div className="price-window-title">
          <h3>Price Window</h3>
          {centerLabel && <span className="muted small">Centered on {centerLabel}</span>}
        </div>
        <div className="price-window-controls">
          <label className="trader-bar-toggle">
            <input
              type="checkbox"
              checked={showFutureData}
              onChange={(event) => setShowFutureData(event.target.checked)}
            />
            <span className="trader-bar-toggle-track" aria-hidden="true" />
            <span className="trader-bar-toggle-label">Show future data</span>
          </label>
          <select
            value={selectedTicker ?? ''}
            onChange={(event) => onSelectTicker(event.target.value || null)}
            disabled={tickers.length === 0}
          >
            {tickers.length === 0 && <option value="">No tickers</option>}
            {tickers.length > 0 &&
              tickers.map((ticker) => (
                <option key={ticker} value={ticker}>
                  {ticker}
                </option>
              ))}
          </select>
        </div>
      </header>

      <div className="price-window-topline" aria-hidden={infoItems.length === 0 && legendItems.length === 0}>
        {infoItems.length > 0 && (
          <div className="price-window-info">
            {infoItems.map((item, idx) => (
              <span key={`info-${idx}`}>{item}</span>
            ))}
          </div>
        )}
        {legendItems.length > 0 && (
          <div className="price-window-legend">
            {legendItems.map((item) => (
              <span key={item.key} className="price-window-legend-item">
                <span
                  className="price-window-legend-swatch"
                  style={{ background: item.color, borderStyle: item.dash === '0' ? 'solid' : 'dashed' }}
                />
                {item.label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="price-window-secondary">
        {secondaryInfoItems.map((item, idx) => (
          <span key={`secondary-${idx}`}>{item}</span>
        ))}
        {supportResistanceItems.map((item, idx) => (
          <span key={`sr-${idx}`}>{item}</span>
        ))}
      </div>

      {tickers.length === 0 && <div className="muted">No tickers identified for this article.</div>}
      {tickers.length > 0 && (
        <div className="trader-bar-chart-content">
          {seriesLoading && <div className="muted">Loading price data…</div>}
          {!seriesLoading && seriesError && <div className="muted">{seriesError}</div>}
          {!seriesLoading && !seriesError && lineShape && (
            <svg
              viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
              role="img"
              aria-label={`Price window for ${selectedTicker ?? ''}`}
            >
              <defs>
                <linearGradient id="trader-bar-line" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#5bd" />
                  <stop offset="100%" stopColor="#4fb" />
                </linearGradient>
                <linearGradient id="trader-bar-fill" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="rgba(91, 221, 255, 0.35)" />
                  <stop offset="100%" stopColor="rgba(91, 221, 255, 0)" />
                </linearGradient>
              </defs>
              {volatilityOverlay?.fill && (
                <rect
                  x={CHART_PADDING.left}
                  y={CHART_PADDING.top}
                  width={Math.max(0, overlayRightX - CHART_PADDING.left)}
                  height={PRICE_AREA_HEIGHT}
                  fill={volatilityOverlay.fill}
                />
              )}
              {bollingerBands && (
                <>
                  <path d={bollingerBands.areaPath} fill="rgba(148, 196, 255, 0.12)" />
                  <path
                    d={bollingerBands.upperPath}
                    fill="none"
                    stroke="rgba(148, 196, 255, 0.9)"
                    strokeWidth="1"
                    strokeDasharray="4 2"
                  />
                  <path
                    d={bollingerBands.lowerPath}
                    fill="none"
                    stroke="rgba(148, 196, 255, 0.9)"
                    strokeWidth="1"
                    strokeDasharray="4 2"
                  />
                </>
              )}
              <path d={lineShape.baseline} stroke="rgba(33, 40, 53, 0.65)" strokeWidth="1" />
              {yAxisTicks.map((tick, idx) => (
                <g key={`y-${tick.y}`}>
                  {idx > 0 && idx < yAxisTicks.length - 1 && (
                    <line
                      x1={CHART_PADDING.left}
                      x2={lineShape.endX}
                      y1={tick.y}
                      y2={tick.y}
                      stroke="rgba(46, 59, 82, 0.35)"
                      strokeDasharray="3 3"
                    />
                  )}
                  <line
                    x1={CHART_PADDING.left - 6}
                    x2={CHART_PADDING.left}
                    y1={tick.y}
                    y2={tick.y}
                    stroke="rgba(60, 72, 92, 0.7)"
                    strokeWidth="1"
                  />
                  <text
                    x={CHART_PADDING.left - 10}
                    y={tick.y + 4}
                    textAnchor="end"
                    fontSize="11"
                    fill="#9fb2c8"
                  >
                    {tick.label}
                  </text>
                </g>
              ))}
              {xAxisTicks.map((tick) => (
                <g key={`x-${tick.x}`}>
                  <line
                    x1={tick.x}
                    x2={tick.x}
                    y1={priceAreaBottom}
                    y2={priceAreaBottom + 6}
                    stroke="rgba(60, 72, 92, 0.7)"
                    strokeWidth="1"
                  />
                  <text
                    x={tick.x}
                    y={priceAreaBottom + 18}
                    textAnchor="middle"
                    fontSize="11"
                    fill="#9fb2c8"
                  >
                    {tick.label}
                  </text>
                </g>
              ))}
              <path d={lineShape.area} fill="url(#trader-bar-fill)" opacity="0.55" />
              {movingAveragePaths.map((ma) => (
                <path
                  key={ma.period}
                  d={ma.path}
                  fill="none"
                  stroke={ma.color}
                  strokeWidth="1.2"
                  opacity="0.9"
                />
              ))}
              {trendLinePath && (
                <path
                  d={trendLinePath}
                  fill="none"
                  stroke={trendLineColor}
                  strokeWidth="1.6"
                  strokeDasharray="6 3"
                  opacity="0.85"
                />
              )}
              {supportResistanceLines.supports.map((line, idx) => (
                <line
                  key={`support-${line.level}-${idx}`}
                  x1={line.x1}
                  x2={line.x2}
                  y1={line.y}
                  y2={line.y}
                  stroke="rgba(72, 214, 196, 0.85)"
                  strokeWidth={line.isNearest ? 1.8 : 1}
                  strokeDasharray={line.isNearest ? '0' : '4 3'}
                />
              ))}
              {supportResistanceLines.resistances.map((line, idx) => (
                <line
                  key={`resistance-${line.level}-${idx}`}
                  x1={line.x1}
                  x2={line.x2}
                  y1={line.y}
                  y2={line.y}
                  stroke="rgba(255, 107, 128, 0.9)"
                  strokeWidth={line.isNearest ? 1.8 : 1}
                  strokeDasharray={line.isNearest ? '0' : '4 3'}
                />
              ))}
              <path
                d={lineShape.line}
                fill="none"
                stroke="url(#trader-bar-line)"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <line
                x1={CHART_PADDING.left}
                x2={lineShape.endX}
                y1={rsiTop}
                y2={rsiTop}
                stroke="rgba(33, 40, 53, 0.85)"
                strokeWidth="1"
              />
              <line
                x1={CHART_PADDING.left}
                x2={lineShape.endX}
                y1={macdTop}
                y2={macdTop}
                stroke="rgba(33, 40, 53, 0.85)"
                strokeWidth="1"
              />
              {[70, 50, 30].map((level) => {
                const y = rsiTop + (1 - level / 100) * RSI_AREA_HEIGHT
                const stroke = level === 50 ? 'rgba(99, 112, 135, 0.6)' : 'rgba(255, 174, 66, 0.45)'
                const dash = level === 50 ? '2 4' : '4 4'
                return (
                  <line
                    key={`rsi-level-${level}`}
                    x1={CHART_PADDING.left}
                    x2={lineShape.endX}
                    y1={y}
                    y2={y}
                    stroke={stroke}
                    strokeDasharray={dash}
                  />
                )
              })}
              {[80, 20].map((level) => {
                const y = rsiTop + (1 - level / 100) * RSI_AREA_HEIGHT
                return (
                  <line
                    key={`mfi-level-${level}`}
                    x1={CHART_PADDING.left}
                    x2={lineShape.endX}
                    y1={y}
                    y2={y}
                    stroke="rgba(72, 214, 196, 0.35)"
                    strokeDasharray="2 6"
                  />
                )
              })}
              {rsiPath && <path d={rsiPath} fill="none" stroke="#c084fc" strokeWidth="1.8" strokeLinecap="round" />}
              {mfiPath && (
                <path
                  d={mfiPath}
                  fill="none"
                  stroke="#ffae42"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  opacity="0.9"
                />
              )}
              {macdHistogram.map((bar, idx) => (
                <rect
                  key={`hist-${idx}`}
                  x={bar.x}
                  y={bar.y}
                  width={bar.width}
                  height={bar.height}
                  fill={bar.positive ? '#48d6c4' : '#ff6b80'}
                  opacity="0.75"
                />
              ))}
              <line
                x1={CHART_PADDING.left}
                x2={lineShape.endX}
                y1={macdValueToY(0)}
                y2={macdValueToY(0)}
                stroke="rgba(99, 112, 135, 0.6)"
                strokeDasharray="4 4"
              />
              {macdLinePath && <path d={macdLinePath} fill="none" stroke="#00f5d4" strokeWidth="1.6" strokeLinecap="round" />}
              {macdSignalPath && <path d={macdSignalPath} fill="none" stroke="#ffbd59" strokeWidth="1.4" strokeLinecap="round" />}
              {obvPath && <path d={obvPath} fill="none" stroke={obvColor} strokeWidth="1.6" strokeLinecap="round" opacity="0.9" />}
              {indicatorX !== null && (
                <line
                  x1={indicatorX}
                  x2={indicatorX}
                  y1={CHART_PADDING.top}
                  y2={chartBottom}
                  stroke="#ffae42"
                  strokeWidth={2.5}
                  strokeLinecap="round"
                  opacity={0.95}
                />
              )}
            </svg>
          )}
          {!seriesLoading && !seriesError && !lineShape && <div className="muted">No price data available.</div>}
        </div>
      )}
      {windowMeta?.note && <div className="muted small">{windowMeta.note}</div>}
      {windowMeta?.interval && <div className="muted small">Interval: {windowMeta.interval}</div>}
    </section>
  )
}
