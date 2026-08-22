import { useEffect, useRef, useState } from 'react'
import { imageUrl, type Box, type PhotoRecord } from './api'

const MIN_AREA = 0.0012 // a stray click is not a selection
const GRAB = 12 // px from an edge that counts as grabbing it

type Edge = { l: boolean; r: boolean; t: boolean; b: boolean }
type Grab = 'inside' | Edge | null

type Drag =
  | { k: 'new'; ax: number; ay: number }
  | { k: 'move'; px: number; py: number; from: Box }
  | { k: 'size'; e: Edge; from: Box }

const clamp = (v: number) => Math.min(1, Math.max(0, v))
const order = (b: Box): Box => ({
  x1: Math.min(b.x1, b.x2),
  y1: Math.min(b.y1, b.y2),
  x2: Math.max(b.x1, b.x2),
  y2: Math.max(b.y1, b.y2),
})

/** What the pointer is over: an edge to pull, the inside to slide, or open canvas. */
function grabAt(b: Box, x: number, y: number, w: number, h: number): Grab {
  const gx = GRAB / w
  const gy = GRAB / h
  const e = {
    l: Math.abs(x - b.x1) < gx,
    r: Math.abs(x - b.x2) < gx,
    t: Math.abs(y - b.y1) < gy,
    b: Math.abs(y - b.y2) < gy,
  }
  const near = x > b.x1 - gx && x < b.x2 + gx && y > b.y1 - gy && y < b.y2 + gy
  if (near && (e.l || e.r || e.t || e.b)) return e
  if (x > b.x1 && x < b.x2 && y > b.y1 && y < b.y2) return 'inside'
  return null
}

function cursorFor(g: Grab): string {
  if (!g) return 'crosshair'
  if (g === 'inside') return 'grab'
  if ((g.l && g.t) || (g.r && g.b)) return 'nwse-resize'
  if ((g.r && g.t) || (g.l && g.b)) return 'nesw-resize'
  return g.l || g.r ? 'ew-resize' : 'ns-resize'
}

export default function PhotoView({
  photo,
  onSearch,
  onSettle,
  busy,
}: {
  photo: PhotoRecord
  onSearch: (box: Box) => void
  /** Fires when a selection stops changing — App warms the query on it. */
  onSettle?: (box: Box | null) => void
  busy: boolean
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [box, setBox] = useState<Box | null>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  const [hoverGrab, setHoverGrab] = useState<Grab>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)

  // Every settled box is announced once. App turns that into a warm query, so
  // the click on "Find this object" usually has nothing left to wait for.
  useEffect(() => {
    if (drag) return
    const id = setTimeout(() => onSettle?.(box), 260)
    return () => clearTimeout(id)
  }, [box, drag, onSettle])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') return setBox(null)
      if (e.key === 'Enter' && box && !busy) return onSearch(box)
      if (!box || !e.key.startsWith('Arrow')) return
      const r = imgRef.current?.getBoundingClientRect()
      if (!r) return
      e.preventDefault()
      // Arrows slide the box; Shift+arrows pull its bottom-right corner.
      const step = e.altKey ? 10 : 1
      const dx = ((e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0) * step) / r.width
      const dy = ((e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0) * step) / r.height
      setBox((b) => {
        if (!b) return b
        if (e.shiftKey) {
          return order({ ...b, x2: clamp(b.x2 + dx), y2: clamp(b.y2 + dy) })
        }
        const w = b.x2 - b.x1
        const h = b.y2 - b.y1
        const x1 = Math.min(1 - w, Math.max(0, b.x1 + dx))
        const y1 = Math.min(1 - h, Math.max(0, b.y1 + dy))
        return { x1, y1, x2: x1 + w, y2: y1 + h }
      })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [box, busy, onSearch])

  function at(e: React.PointerEvent) {
    const r = imgRef.current!.getBoundingClientRect()
    return {
      x: clamp((e.clientX - r.left) / r.width),
      y: clamp((e.clientY - r.top) / r.height),
      w: r.width,
      h: r.height,
    }
  }

  function down(e: React.PointerEvent) {
    const p = at(e)
    e.currentTarget.setPointerCapture(e.pointerId)
    const g = box ? grabAt(box, p.x, p.y, p.w, p.h) : null
    if (g === 'inside') setDrag({ k: 'move', px: p.x, py: p.y, from: box! })
    else if (g) setDrag({ k: 'size', e: g, from: box! })
    else {
      setDrag({ k: 'new', ax: p.x, ay: p.y })
      setBox({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
    }
  }

  function move(e: React.PointerEvent) {
    const p = at(e)
    setCursor({ x: p.x, y: p.y })
    if (!drag) return setHoverGrab(box ? grabAt(box, p.x, p.y, p.w, p.h) : null)

    if (drag.k === 'new') {
      setBox(order({ x1: drag.ax, y1: drag.ay, x2: p.x, y2: p.y }))
    } else if (drag.k === 'move') {
      const f = drag.from
      const w = f.x2 - f.x1
      const h = f.y2 - f.y1
      const x1 = Math.min(1 - w, Math.max(0, f.x1 + (p.x - drag.px)))
      const y1 = Math.min(1 - h, Math.max(0, f.y1 + (p.y - drag.py)))
      setBox({ x1, y1, x2: x1 + w, y2: y1 + h })
    } else {
      const f = drag.from
      setBox(
        order({
          x1: drag.e.l ? p.x : f.x1,
          y1: drag.e.t ? p.y : f.y1,
          x2: drag.e.r ? p.x : f.x2,
          y2: drag.e.b ? p.y : f.y2,
        }),
      )
    }
  }

  function up() {
    setDrag(null)
    setBox((b) => (b && (b.x2 - b.x1) * (b.y2 - b.y1) >= MIN_AREA ? b : null))
  }

  const pct = (n: number) => `${n * 100}%`
  const px = box
    ? `${Math.round((box.x2 - box.x1) * photo.width)} × ${Math.round(
        (box.y2 - box.y1) * photo.height,
      )}`
    : ''
  const grabbing = drag?.k === 'move'
  const sizing = drag?.k === 'size'
  // Keep the readout clear of the top edge of the frame.
  const readoutBelow = box ? box.y1 < 0.05 : false

  return (
    <div className="photo">
      <div className="stage-area">
        <div
          className={`canvas${box ? ' has-sel' : ''}${drag ? ' dragging' : ''}`}
          style={{ cursor: grabbing ? 'grabbing' : cursorFor(drag ? (sizing ? drag.e : 'inside') : hoverGrab) }}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          onPointerLeave={() => {
            setCursor(null)
            setHoverGrab(null)
          }}
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
                className={`sel${drag ? ' live' : ''}`}
                style={{
                  left: pct(box.x1),
                  top: pct(box.y1),
                  width: pct(box.x2 - box.x1),
                  height: pct(box.y2 - box.y1),
                }}
              >
                <span className="c tl" /><span className="c tr" />
                <span className="c bl" /><span className="c br" />
                {/* thirds, only while the box is being shaped — they help you
                    centre an object, and would be clutter once it is placed. */}
                {drag && (
                  <>
                    <b style={{ left: '33.333%' }} /><b style={{ left: '66.666%' }} />
                    <i style={{ top: '33.333%' }} /><i style={{ top: '66.666%' }} />
                  </>
                )}
              </div>
              <div
                className={`readout${readoutBelow ? ' below' : ''}`}
                style={{ left: pct(box.x1), top: pct(readoutBelow ? box.y2 : box.y1) }}
              >
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
              <b>{px}</b> selected. Drag the edges to adjust, or arrow keys to nudge.
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
          {busy && <span className="spin tiny" />}
          {busy ? 'Searching' : 'Find this object'}
        </button>
      </div>
    </div>
  )
}
