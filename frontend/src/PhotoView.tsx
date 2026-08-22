import { useEffect, useRef, useState } from 'react'
import { imageUrl, type Box, type PhotoRecord } from './api'

const MIN_AREA = 0.0012 // a stray click is not a selection

export default function PhotoView({
  photo,
  onSearch,
  busy,
}: {
  photo: PhotoRecord
  onSearch: (box: Box) => void
  busy: boolean
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [box, setBox] = useState<Box | null>(null)
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setBox(null)
      if (e.key === 'Enter' && box && !busy) onSearch(box)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [box, busy, onSearch])

  function at(e: React.PointerEvent) {
    const r = imgRef.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    }
  }

  function down(e: React.PointerEvent) {
    const p = at(e)
    e.currentTarget.setPointerCapture(e.pointerId)
    setDrag(p)
    setBox({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
  }

  function move(e: React.PointerEvent) {
    const p = at(e)
    setCursor(p)
    if (!drag) return
    setBox({
      x1: Math.min(drag.x, p.x),
      y1: Math.min(drag.y, p.y),
      x2: Math.max(drag.x, p.x),
      y2: Math.max(drag.y, p.y),
    })
  }

  function up() {
    setDrag(null)
    setBox((b) =>
      b && (b.x2 - b.x1) * (b.y2 - b.y1) >= MIN_AREA ? b : null,
    )
  }

  const pct = (n: number) => `${n * 100}%`
  const px = box
    ? `${Math.round((box.x2 - box.x1) * photo.width)} × ${Math.round(
        (box.y2 - box.y1) * photo.height,
      )}`
    : ''

  return (
    <div className="photo">
      <div className="stage-area">
        <div
          className="canvas"
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          onPointerLeave={() => setCursor(null)}
        >
          <img ref={imgRef} src={imageUrl(photo.id)} alt="" draggable={false} />

          {cursor && !box && (
            <>
              <div className="guide h" style={{ top: pct(cursor.y) }} />
              <div className="guide v" style={{ left: pct(cursor.x) }} />
            </>
          )}

          {box && (
            <>
              <div className="veil" style={{ left: 0, right: 0, top: 0, height: pct(box.y1) }} />
              <div className="veil" style={{ left: 0, right: 0, top: pct(box.y2), bottom: 0 }} />
              <div
                className="veil"
                style={{ left: 0, width: pct(box.x1), top: pct(box.y1), height: pct(box.y2 - box.y1) }}
              />
              <div
                className="veil"
                style={{ left: pct(box.x2), right: 0, top: pct(box.y1), height: pct(box.y2 - box.y1) }}
              />
              <div
                className="sel"
                style={{
                  left: pct(box.x1),
                  top: pct(box.y1),
                  width: pct(box.x2 - box.x1),
                  height: pct(box.y2 - box.y1),
                }}
              >
                <span /><span /><span /><span />
              </div>
              <div className="readout" style={{ left: pct(box.x1), top: pct(box.y1) }}>
                {px} px
              </div>
            </>
          )}
        </div>
      </div>

      <div className="dock">
        <span className="instruct">
          {box ? (
            <>
              Selected <b>{px}</b> pixels. Drag again to reselect.
            </>
          ) : (
            <>
              <b>Drag a box</b> around one object — a watch, a bear, a wallet.
            </>
          )}
        </span>
        <span className="spacer" />
        {box && (
          <button className="ghost" onClick={() => setBox(null)}>
            Clear
          </button>
        )}
        <button className="go" disabled={!box || busy} onClick={() => box && onSearch(box)}>
          {busy ? 'Searching' : 'Find this object'}
        </button>
      </div>
    </div>
  )
}
