import type { PostListItem } from '../types'
import { clampSpamPct, spamClassForPct } from '../utils/spam'
import { buildInterestDisplay } from '../utils/interest'
import Badge from './Badge'

type PostListProps = {
  items: PostListItem[]
  total: number
  page: number
  pageSize: number
  onPage: (p: number) => void
  onPick: (id: string) => void
  selectedId?: string | null
}

export default function PostList({
  items,
  total,
  page,
  pageSize,
  onPage,
  onPick,
  selectedId,
}: PostListProps) {
  return (
    <div className="list">
      <div className="list-head">
        <div className="list-head-meta">
          <h2>Posts</h2>
          <div className="muted small">{items.length} shown / {total} total</div>
        </div>
        <div className="pager">
          <button disabled={page <= 1} onClick={() => onPage(page - 1)}>
            Prev
          </button>
          <span>
            Page {page} / {Math.ceil(total / pageSize)}
          </span>
          <button disabled={page >= Math.ceil(total / pageSize)} onClick={() => onPage(page + 1)}>
            Next
          </button>
        </div>
      </div>
      <div className="rows">
        {items.map((p) => {
          const interestMeta = buildInterestDisplay(p.interest ?? null)
          const interestTrackClass = ['spam-meter-track', interestMeta.trackClass].filter(Boolean).join(' ')
          const interestHandleClasses = ['spam-meter-handle', interestMeta.hasScore ? '' : 'spam-meter-handle-empty']
          if (interestMeta.status === 'error') {
            interestHandleClasses.push('interest-meter-handle-error')
          }
          const interestMeterClasses = ['spam-meter', 'interest-meter']
          if (interestMeta.status === 'error') {
            interestMeterClasses.push('interest-meter-error')
          }

          const rawSpamValue = p.spam_likelihood_pct
          let hasSpamData = false
          let spamPct = 0
          if (typeof rawSpamValue === 'number' && Number.isFinite(rawSpamValue)) {
            hasSpamData = true
            spamPct = clampSpamPct(rawSpamValue)
          } else if (typeof rawSpamValue === 'string') {
            const parsed = Number.parseFloat(rawSpamValue)
            if (Number.isFinite(parsed)) {
              hasSpamData = true
              spamPct = clampSpamPct(parsed)
            }
          }

          const spamDisplay = hasSpamData ? Math.round(spamPct) : null
          const spamTrackClass = spamClassForPct(hasSpamData ? spamPct : null)
          const rawChairmanText =
            typeof p.chairman_plain_english === 'string' ? p.chairman_plain_english.trim() : ''
          const chairmanText = rawChairmanText.replace(/\s+/g, ' ').trim()
          const hasChairmanVerdict = chairmanText.length > 0
          const chairmanDirection = typeof p.chairman_direction === 'string' ? p.chairman_direction.trim() : ''
          const chairmanDirectionClass = chairmanDirection ? `direction-${chairmanDirection.toLowerCase()}` : ''
          const rowClassName = [
            'row',
            p.has_analysis ? 'analysed' : '',
            hasChairmanVerdict ? 'has-chairman' : '',
            selectedId && p.post_id === selectedId ? 'active' : '',
          ]
            .filter(Boolean)
            .join(' ')
          const assetBadges = Array.isArray(p.assets_mentioned)
            ? p.assets_mentioned.map((asset, idx) => {
                const str = (value: string | null | undefined) => (value == null ? '' : `${value}`)
                const ticker = str(asset?.ticker).toUpperCase()
                const name = str(asset?.name_or_description)
                const exchange = str(asset?.exchange_or_market)
                const display = ticker || name || '?'
                const nameLower = name.trim().toLowerCase()
                const missingName =
                  nameLower === '' ||
                  nameLower === 'null' ||
                  nameLower === 'none' ||
                  nameLower === 'n/a' ||
                  nameLower === 'na'
                const titleParts = [ticker, !missingName ? name : '', exchange].filter((part) => part && part.trim().length > 0)
                const title = titleParts.length > 0 ? titleParts.join(' · ') : undefined

                return (
                  <Badge key={`${ticker || name || idx}`} title={title}>
                    <span className={missingName ? 'ticker-missing' : undefined}>{display}</span>
                  </Badge>
                )
              })
            : []

          return (
            <button key={p.post_id} className={rowClassName} onClick={() => onPick(p.post_id)}>
              {hasChairmanVerdict && (
                <div className={['row-chairman-banner', chairmanDirectionClass].filter(Boolean).join(' ')}>
                  <span className="row-chairman-label">Chairman Verdict</span>
                  <span className="row-chairman-text">{chairmanText}</span>
                </div>
              )}
              <div className="row-inner">
                <div className="row-icons">
                  {p.has_summary && <span className="badge s">S</span>}
                  {p.has_analysis && <span className="badge a">A</span>}
                </div>
                <div className="row-content">
                  <div className="row-main">
                    <div className="row-title">{p.title || '(no title)'}</div>
                    <div className="row-sub">
                      {p.platform} · {p.source} · {p.posted_at || p.scraped_at || ''}
                    </div>
                  </div>
                  <div className="row-meta">
                    <div className={interestMeterClasses.join(' ')}>
                      <div className="spam-meter-header">
                        <span className="spam-meter-label">Interest score</span>
                        <span className="spam-meter-value">{interestMeta.valueText}</span>
                      </div>
                      <div className={interestTrackClass}>
                        <div
                          className={interestHandleClasses.filter(Boolean).join(' ')}
                          style={{ left: `${interestMeta.hasScore ? interestMeta.pct : 50}%` }}
                        />
                      </div>
                      <div className="interest-meter-reason" title={interestMeta.reason}>
                        {interestMeta.reason}
                      </div>
                    </div>
                    {!hasChairmanVerdict && (
                      <div className="spam-meter">
                        <div className="spam-meter-header">
                          <span className="spam-meter-label">Spam likelihood</span>
                          <span className="spam-meter-value">{spamDisplay != null ? `${spamDisplay}%` : '—'}</span>
                        </div>
                        <div className={spamTrackClass}>
                          <div
                            className={['spam-meter-handle', hasSpamData ? '' : 'spam-meter-handle-empty']
                              .filter(Boolean)
                              .join(' ')}
                            style={{ left: `${hasSpamData ? spamPct : 50}%` }}
                          />
                        </div>
                      </div>
                    )}
                    <div className="badges">
                      {p.markets?.slice(0, 8).map((t) => (
                        <span key={t} className="badge">
                          {t}
                        </span>
                      ))}
                    </div>
                    {assetBadges.length > 0 && <div className="badges asset-badges">{assetBadges}</div>}
                  </div>
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
