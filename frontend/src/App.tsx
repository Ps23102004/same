import { useCallback, useEffect, useRef, useState } from 'react'
import Setup from './Setup'
import Library from './Library'
import PhotoView from './PhotoView'
import TimelineView from './Timeline'
import { getStatus, query, type Box, type PhotoRecord, type QueryResponse, type Status } from './api'

type View =
  | { k: 'boot' }
  | { k: 'setup' }
  | { k: 'library' }
  | { k: 'photo'; photo: PhotoRecord }
  | { k: 'results'; photo: PhotoRecord; box: Box; result: QueryResponse }

export default function App() {
  const [view, setView] = useState<View>({ k: 'boot' })
  const [status, setStatus] = useState<Status | null>(null)
  const [busy, setBusy] = useState(false)
  const [fault, setFault] = useState<string | null>(null)

  // A settled selection is searched immediately, in the background. By the time
  // the user reaches "Find this object" the answer is usually already here, so
  // the button resolves in one frame instead of one second. The work is the
  // real query — nothing is faked, it is only started earlier.
  const warm = useRef(new Map<string, Promise<QueryResponse>>())
  const keyOf = (id: string, b: Box) =>
    `${id}|${[b.x1, b.y1, b.x2, b.y2].map((n) => n.toFixed(4)).join(',')}`

  const refresh = useCallback(async () => {
    try {
      const s = await getStatus()
      setStatus(s)
      setView({ k: s.indexed && s.photo_count > 0 ? 'library' : 'setup' })
    } catch (e) {
      setFault(String((e as Error).message))
      setView({ k: 'setup' })
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  function prewarm(photo: PhotoRecord, box: Box | null) {
    if (!box) return
    const k = keyOf(photo.id, box)
    if (warm.current.has(k)) return
    if (warm.current.size > 6) warm.current.clear()
    const pending = query(photo.id, box)
    // Observe the rejection here so a failed warm-up is never an unhandled
    // promise; search() still sees the failure through its own await.
    pending.catch(() => warm.current.delete(k))
    warm.current.set(k, pending)
  }

  async function search(photo: PhotoRecord, box: Box, objectId?: string) {
    setBusy(true)
    setFault(null)
    try {
      // Feedback changes the answer, so a refine never reuses a warm result.
      const pending = objectId ? undefined : warm.current.get(keyOf(photo.id, box))
      const result = await (pending ?? query(photo.id, box, objectId))
      setView({ k: 'results', photo, box, result })
    } catch (e) {
      setFault(String((e as Error).message))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="bar">
        {(view.k === 'photo' || (view.k === 'setup' && status?.indexed)) && (
          <button className="back" onClick={() => setView({ k: 'library' })}>
            <Chevron /> Library
          </button>
        )}
        {view.k === 'results' && (
          <button className="back" onClick={() => setView({ k: 'photo', photo: view.photo })}>
            <Chevron /> Photo
          </button>
        )}
        <span className="wordmark">Same</span>
        <span className="spacer" />
        {status?.folder_path && view.k !== 'setup' && (
          <span className="crumb" title={status.folder_path}>
            {status.folder_path}
          </span>
        )}
        {view.k === 'library' && (
          <button className="back" onClick={() => setView({ k: 'setup' })}>
            Change folder
          </button>
        )}
        <span className="privacy" tabIndex={0}>
          <span className="led" />
          On this Mac
          <span className="note">
            <b>Nothing uploads.</b> Photos are read from disk, embedded and matched by a
            model running locally, and the index is written next to them. This page loads no
            fonts, scripts, or images from anywhere but 127.0.0.1 — the only network call
            Same ever makes is the one-time model download during setup.
          </span>
        </span>
      </header>

      <main className="main">
        {view.k === 'boot' && (
          <div className="empty">
            <div className="spin" style={{ margin: '0 auto' }} />
          </div>
        )}

        {view.k === 'setup' && <Setup onDone={refresh} />}

        {view.k === 'library' && (
          <Library onOpen={(photo) => setView({ k: 'photo', photo })} />
        )}

        {view.k === 'photo' && (
          <>
            <PhotoView
              key={view.photo.id}
              photo={view.photo}
              busy={busy}
              onSettle={(b) => prewarm(view.photo, b)}
              onSearch={(b) => search(view.photo, b)}
            />
            {fault && (
              <div className="err" style={{ margin: '0 18px 18px' }}>
                <div className="what">Search didn't run</div>
                <div className="fix">{fault}</div>
              </div>
            )}
          </>
        )}

        {view.k === 'results' && (
          <TimelineView
            result={view.result}
            source={view.photo}
            box={view.box}
            busy={busy}
            onBack={() => setView({ k: 'photo', photo: view.photo })}
            onRefine={() => search(view.photo, view.box, view.result.query_id)}
          />
        )}
      </main>
    </div>
  )
}

const Chevron = () => (
  <svg width="7" height="12" viewBox="0 0 7 12" fill="none" aria-hidden="true">
    <path d="M6 1L1 6l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)
