import { useEffect, useState } from 'react'
import { getPhotos, imageUrl, type PhotoRecord } from './api'

export default function Library({ onOpen }: { onOpen: (p: PhotoRecord) => void }) {
  const [photos, setPhotos] = useState<PhotoRecord[] | null>(null)
  const [total, setTotal] = useState(0)
  const [fault, setFault] = useState<string | null>(null)

  useEffect(() => {
    getPhotos(0, 2000)
      .then((p) => {
        setPhotos(p.photos)
        setTotal(p.total)
      })
      .catch((e) => setFault(String((e as Error).message)))
  }, [])

  if (fault)
    return (
      <div className="empty">
        <div className="glyph" />
        <h3>Can't read the library</h3>
        <p>{fault}</p>
      </div>
    )

  if (!photos)
    return (
      <div className="empty">
        <div className="spin" style={{ margin: '0 auto 18px' }} />
        <p>Reading the index.</p>
      </div>
    )

  if (photos.length === 0)
    return (
      <div className="empty">
        <div className="glyph" />
        <h3>Nothing indexed yet</h3>
        <p>Point Same at a folder of photos and it will read them once, here on this Mac.</p>
      </div>
    )

  return (
    <>
      <div className="libhead">
        <h2>Library</h2>
        <span className="mono faint">{total.toLocaleString()} indexed</span>
      </div>
      <div className="grid">
        {photos.map((p) => (
          <button key={p.id} className="tile" onClick={() => onOpen(p)} title={p.path}>
            <img
              src={imageUrl(p.id, 320)}
              alt=""
              loading="lazy"
              decoding="async"
            />
            {p.source_video_path && (
              <span className="vid">{fmt(p.frame_time_sec ?? 0)}</span>
            )}
          </button>
        ))}
      </div>
    </>
  )
}

function fmt(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
