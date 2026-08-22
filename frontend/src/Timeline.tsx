import { useState } from 'react'
import { imageUrl, sendFeedback, type Box, type Match, type PhotoRecord, type QueryResponse } from './api'

const DAY = 86_400_000

const dateFmt = new Intl.DateTimeFormat(undefined, {
  day: 'numeric', month: 'long', year: 'numeric',
})
const timeFmt = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })
const rel = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

function ago(iso: string) {
  const d = (Date.now() - new Date(iso).getTime()) / DAY
  const mag = Math.abs(d)
  if (mag < 1) return rel.format(-Math.round(d * 24), 'hour')
  if (mag < 45) return rel.format(-Math.round(d), 'day')
  if (mag < 365) return rel.format(-Math.round(d / 30.44), 'month')
  return rel.format(-Math.round(d / 365.25), 'year')
}

/** "1 yr 4 mo" — the length of an absence, spelled out on the spine. */
function span(days: number) {
  const y = Math.floor(days / 365.25)
  const m = Math.floor((days - y * 365.25) / 30.44)
  if (y) return m ? `${y} yr ${m} mo` : `${y} yr`
  if (m) return `${m} mo`
  return `${Math.round(days)} days`
}

const coord = (v: number, pos: string, neg: string) =>
  `${Math.abs(v).toFixed(4)}° ${v >= 0 ? pos : neg}`

/** Zoom a source photo to the region the user boxed. */
function cropStyle(photo: PhotoRecord, b: Box, size: number) {
  const s = size / Math.max((b.x2 - b.x1) * photo.width, (b.y2 - b.y1) * photo.height, 1)
  return {
    position: 'absolute' as const,
    width: photo.width * s,
    height: photo.height * s,
    left: size / 2 - ((b.x1 + b.x2) / 2) * photo.width * s,
    top: size / 2 - ((b.y1 + b.y2) / 2) * photo.height * s,
    maxWidth: 'none',
  }
}

export default function Timeline({
  result,
  source,
  box,
  onBack,
  onRefine,
  busy,
}: {
  result: QueryResponse
  source: PhotoRecord
  box: Box
  onBack: () => void
  onRefine: () => void
  busy: boolean
}) {
  const [verdicts, setVerdicts] = useState<Record<string, boolean>>({})
  const matches = result.matches
  const taught = Object.keys(verdicts).length

  function judge(m: Match, confirmed: boolean) {
    setVerdicts((v) => ({ ...v, [m.photo_id]: confirmed }))
    // Send the box too: a confirm stores exactly the region the user endorsed
    // rather than making the backend re-derive it.
    sendFeedback(result.query_id, m.photo_id, confirmed, m.bbox).catch(() => {})
  }

  const last = result.last_seen

  return (
    <div className="results">
      {last ? (
        <div className="lastseen">
          <div className="crop">
            <img src={imageUrl(source.id)} alt="" style={cropStyle(source, box, 116)} />
          </div>
          <div className="body">
            <div className="eyebrow">Last seen</div>
            <p className="when">{last.taken_at ? dateFmt.format(new Date(last.taken_at)) : 'Date unknown'}</p>
            <div className="ago">
              {last.taken_at
                ? `${ago(last.taken_at)}, around ${timeFmt.format(new Date(last.taken_at))}`
                : 'This photo carries no timestamp.'}
            </div>
            {last.gps ? (
              <>
                <div className="where">
                  {coord(last.gps.lat, 'N', 'S')}, {coord(last.gps.lon, 'E', 'W')}
                </div>
                <div className="whereno">
                  Coordinates as written in the photo. Same doesn't look up a place name —
                  that would mean sending your location to a map service.
                </div>
              </>
            ) : (
              <div className="whereno">No location in this photo's EXIF.</div>
            )}
          </div>
        </div>
      ) : (
        <div className="empty">
          <div className="glyph" />
          <h3>No other photos contain this object</h3>
          <p>
            Same compared your selection against every indexed photo and none held up under
            geometric verification. That's a real answer, not a failure — this object appears
            once in the library.
          </p>
          <p className="faint mono" style={{ marginTop: 14 }}>
            {matches.length === 0
              ? 'No candidates survived the shortlist.'
              : `${matches.length} candidates matched loosely but failed verification.`}
          </p>
          <button className="ghost" style={{ marginTop: 22 }} onClick={onBack}>
            Choose a different object
          </button>
        </div>
      )}

      {matches.length > 0 && (
        <>
          <div className="tlhead">
            <h2>Across the library</h2>
            <span className="mono faint">
              {matches.length} sighting{matches.length === 1 ? '' : 's'}, oldest first
            </span>
          </div>

          <div className="timeline">
            {matches.map((m, i) => {
              const prev = i > 0 ? matches[i - 1] : null
              const t = m.taken_at ? new Date(m.taken_at) : null
              const pt = prev?.taken_at ? new Date(prev.taken_at) : null
              const days = t && pt ? (t.getTime() - pt.getTime()) / DAY : 0
              const newYear = t && (!pt || pt.getFullYear() !== t.getFullYear())
              const verdict = verdicts[m.photo_id]

              return (
                <div key={m.photo_id}>
                  {/* The empty stretch IS the information: it is drawn as long as
                      the object was out of sight. */}
                  {days > 14 && (
                    <div className="gap" style={{ height: Math.min(130, (Math.log2(days + 1) - 3) * 16) }}>
                      {days > 45 && <span className="lbl">{span(days)}</span>}
                    </div>
                  )}
                  {newYear && <div className="yr">{t!.getFullYear()}</div>}

                  <div
                    className={`row ${verdict === true ? 'confirmed' : ''} ${verdict === false ? 'rejected' : ''}`}
                    style={{ animationDelay: `${Math.min(i, 12) * 45}ms` }}
                  >
                    <span className="node" />
                    <div className="shot">
                      <img src={imageUrl(m.photo_id, 480)} alt="" loading="lazy" />
                      <span
                        className="mark"
                        style={{
                          left: `${m.bbox.x1 * 100}%`,
                          top: `${m.bbox.y1 * 100}%`,
                          width: `${(m.bbox.x2 - m.bbox.x1) * 100}%`,
                          height: `${(m.bbox.y2 - m.bbox.y1) * 100}%`,
                        }}
                      />
                    </div>

                    <div className="meta">
                      <div className="date">
                        {t ? dateFmt.format(t) : 'No timestamp'}
                      </div>
                      <div className="sub">
                        {t ? `${timeFmt.format(t)} · ${ago(m.taken_at!)}` : 'Not dated in EXIF'}
                        {m.gps && ` · ${coord(m.gps.lat, 'N', 'S')}, ${coord(m.gps.lon, 'E', 'W')}`}
                      </div>
                      <div className="conf">
                        {m.inlier_count} inliers
                        {!m.verified && <span className="faint">· unverified</span>}
                      </div>
                    </div>

                    <div className={`verdict ${verdict !== undefined ? 'set' : ''}`}>
                      <button
                        className={verdict === true ? 'on' : ''}
                        aria-label="Yes, same object"
                        title="Yes, same object"
                        onClick={() => judge(m, true)}
                      >
                        <Check />
                      </button>
                      <button
                        className={verdict === false ? 'on no' : ''}
                        aria-label="No, different object"
                        title="No, different object"
                        onClick={() => judge(m, false)}
                      >
                        <Cross />
                      </button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {taught > 0 && (
            <div className="refine">
              <button className="ghost" onClick={onRefine} disabled={busy}>
                {busy ? 'Searching' : `Search again using my ${taught} answer${taught === 1 ? '' : 's'}`}
              </button>
              <span className="faint">
                Confirmed sightings become extra views of the object; rejected ones are
                blacklisted. The next search uses both.
              </span>
            </div>
          )}
        </>
      )}

      <p className="footnote">
        Each sighting is scored by geometric verification: local features matched between
        your selection and the candidate, then filtered by RANSAC. The inlier count is how
        many of those matches agreed on one consistent transform — it's the number Same
        actually trusts. Two identical manufactured items (two of the same mug) will match
        each other, because they genuinely look the same; confirming and rejecting sightings
        is how you teach Same which one is yours.
      </p>
    </div>
  )
}

const Check = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
    <path d="M2.5 7.4l3 3 6-6.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)
const Cross = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
    <path d="M3.4 3.4l7.2 7.2M10.6 3.4l-7.2 7.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
)
