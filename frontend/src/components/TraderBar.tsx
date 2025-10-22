import { useEffect, useMemo, useState } from 'react'

import { getPostsCalendar } from '../api'
import { formatDateKey } from '../utils/dates'

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function buildCalendar(anchor: Date) {
  const startOfMonth = new Date(anchor.getFullYear(), anchor.getMonth(), 1)
  const startDay = startOfMonth.getDay()
  const gridStart = new Date(startOfMonth)
  gridStart.setDate(startOfMonth.getDate() - startDay)

  const days: { date: Date; inMonth: boolean }[] = []

  for (let i = 0; i < 42; i += 1) {
    const date = new Date(gridStart)
    date.setDate(gridStart.getDate() + i)
    days.push({
      date,
      inMonth: date.getMonth() === anchor.getMonth(),
    })
  }

  return days
}

function isSameDay(a: Date | null, b: Date) {
  if (!a) return false
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
}

type TraderBarProps = {
  selectedDate: Date | null
  onDateChange: (date: Date) => void
}

type CalendarAvailability = Map<string, { count: number; analysed: number }>

export default function TraderBar({ selectedDate, onDateChange }: TraderBarProps) {
  const [anchor, setAnchor] = useState(() => {
    const base = selectedDate ?? new Date()
    return new Date(base.getFullYear(), base.getMonth(), 1)
  })

  const [monthAvailability, setMonthAvailability] = useState<CalendarAvailability>(new Map())

  useEffect(() => {
    if (!selectedDate) return
    setAnchor((prev) => {
      if (prev.getFullYear() === selectedDate.getFullYear() && prev.getMonth() === selectedDate.getMonth()) {
        return prev
      }
      return new Date(selectedDate.getFullYear(), selectedDate.getMonth(), 1)
    })
  }, [selectedDate])

  const calendar = useMemo(() => buildCalendar(anchor), [anchor])

  useEffect(() => {
    let cancelled = false

    setMonthAvailability(new Map())

    const year = anchor.getFullYear()
    const month = anchor.getMonth() + 1

    getPostsCalendar(year, month)
      .then((days) => {
        if (cancelled) return
        const next: CalendarAvailability = new Map()
        days.forEach((day) => {
          if (!day.date || day.count <= 0) return
          const analysed = day.analysed_count && day.analysed_count > 0 ? day.analysed_count : 0
          next.set(day.date, { count: day.count, analysed })
        })
        setMonthAvailability(next)
      })
      .catch(() => {
        if (!cancelled) {
          setMonthAvailability(new Map())
        }
      })

    return () => {
      cancelled = true
    }
  }, [anchor])

  const monthFormatter = useMemo(
    () => new Intl.DateTimeFormat('en-US', { month: 'long', year: 'numeric' }),
    [],
  )

  const goPrev = () => {
    setAnchor((prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1))
  }

  const goNext = () => {
    setAnchor((prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1))
  }

  const handleSelect = (entry: { date: Date }) => {
    const normalized = new Date(entry.date.getFullYear(), entry.date.getMonth(), entry.date.getDate())
    setAnchor(new Date(normalized.getFullYear(), normalized.getMonth(), 1))
    onDateChange(normalized)
  }

  const goToday = () => {
    const today = new Date()
    const normalized = new Date(today.getFullYear(), today.getMonth(), today.getDate())
    setAnchor(new Date(today.getFullYear(), today.getMonth(), 1))
    onDateChange(normalized)
  }

  return (
    <section className="trader-bar trader-bar-calendar-only">
      <div className="trader-bar-panel trader-bar-calendar">
        <header className="trader-bar-calendar-header">
          <button type="button" onClick={goPrev} aria-label="Previous month">
            ‹
          </button>
          <div className="trader-bar-calendar-title">{monthFormatter.format(anchor)}</div>
          <button type="button" onClick={goNext} aria-label="Next month">
            ›
          </button>
        </header>
        <div className="trader-bar-weekdays">
          {WEEKDAYS.map((day) => (
            <div key={day}>{day}</div>
          ))}
        </div>
        <div className="trader-bar-grid">
          {calendar.map((entry, index) => {
            const dateKey = formatDateKey(entry.date)
            const dayInfo = monthAvailability.get(dateKey)
            const articleCount = dayInfo?.count ?? 0
            const analysedCount = dayInfo?.analysed ?? 0
            const dateLabel = entry.date.toLocaleDateString('en-US', {
              month: 'long',
              day: 'numeric',
              year: 'numeric',
            })
            const countLabel =
              articleCount <= 0 ? 'No articles' : `${articleCount} ${articleCount === 1 ? 'article' : 'articles'}`
            const analysisLabel = analysedCount > 0 ? 'Fully analysed articles present' : 'No fully analysed articles'
            const accessibleLabel = `${dateLabel} · ${countLabel} · ${analysisLabel}`

            return (
              <button
                key={`${entry.date.getTime()}-${index}`}
                type="button"
                onClick={() => handleSelect(entry)}
                className={
                  [
                    'trader-bar-day',
                    entry.inMonth ? 'current' : 'muted',
                    articleCount > 0 ? 'has-posts' : '',
                    analysedCount > 0 ? 'has-analysis' : '',
                    isSameDay(selectedDate, entry.date) ? 'selected' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')
                }
                aria-label={accessibleLabel}
                title={accessibleLabel}
                data-count={articleCount > 0 ? articleCount : undefined}
                data-analysed={analysedCount > 0 ? analysedCount : undefined}
              >
                {entry.date.getDate()}
              </button>
            )
          })}
        </div>
        <div className="trader-bar-calendar-actions">
          <button type="button" onClick={goToday}>
            Today
          </button>
        </div>
      </div>
    </section>
  )
}
