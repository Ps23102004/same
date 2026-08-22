import { useEffect, useRef, useState } from 'react'
import { startIndex, getProgress, type IndexProgress, type IndexStage } from './api'

const STAGES: { k: IndexStage; label: string }[] = [
  { k: 'scanning', label: 'Scanning' },
  { k: 'extracting_frames', label: 'Video frames' },
  { k: 'embedding', label: 'Embedding' },
  { k: 'building_index', label: 'Building index' },
  { k: 'done', label: 'Done' },
]

export default function Setup({ onDone }: { onDone: () => void }) {
  const [path, setPath] = useState('')
  const [prog, setProg] = useState<IndexProgress | null>(null)
  const [fault, setFault] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  async function poll(jobId: string) {
    try {
      const p = await getProgress(jobId)
      setProg(p)
      if (p.stage === 'done') return onDone()
      if (p.stage === 'error') return setFault(p.error || 'Indexing stopped.')
      timer.current = window.setTimeout(() => poll(jobId), 400)
    } catch (e) {
      setFault(String((e as Error).message))
    }
  }

  async function begin() {
    setFault(null)
    try {
      const p = await startIndex(path.trim())
      setProg(p)
      poll(p.job_id)
    } catch (e) {
      setFault(String((e as Error).message))
    }
  }

  const running = !!prog && prog.stage !== 'error'
  const at = prog ? STAGES.findIndex((s) => s.k === prog.stage) : -1
  const files = prog?.total_files ?? 0
  const pct = files ? Math.min(100, ((prog?.processed_files ?? 0) / files) * 100) : 0

  return (
    <div className="setup">
      <h1>
        Every photo it has
        <br />
        ever <em>appeared</em> in.
      </h1>
      <p className="lede">
        Point Same at a folder. It reads each photo and video once, on this machine, and
        builds an index you can search by pointing at an object.
      </p>

      <div className="field">
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && path.trim() && !running && begin()}
          placeholder="/Users/you/Pictures/Archive"
          spellCheck={false}
          autoFocus
          aria-label="Folder to index"
          disabled={running}
        />
        <button className="go" onClick={begin} disabled={!path.trim() || running}>
          {running ? 'Indexing' : 'Index folder'}
        </button>
      </div>
      <p className="hint">
        An absolute path. A web page can't be handed a real folder, so this is a text
        field — the local server does the reading.
      </p>

      {prog && (
        <div className="prog">
          <div className="stages">
            {STAGES.map((s, i) => (
              <span key={s.k} className={`stage ${i === at ? 'now' : i < at ? 'done' : ''}`}>
                {s.label}
              </span>
            ))}
          </div>
          <div className="track">
            <i style={{ width: `${pct}%` }} />
          </div>
          <div className="counts">
            <span>
              <b>{prog.processed_files.toLocaleString()}</b> of {files.toLocaleString()} files
              {prog.total_frames > 0 && (
                <>
                  {' · '}
                  <b>{prog.processed_frames.toLocaleString()}</b> of{' '}
                  {prog.total_frames.toLocaleString()} frames
                </>
              )}
            </span>
            <span className="faint">{prog.message}</span>
          </div>
        </div>
      )}

      {fault && (
        <div className="err">
          <div className="what">Indexing didn't start</div>
          <div className="fix">
            {fault}
            <br />
            Start the backend with <code>uv run uvicorn backend.app.main:app --port 8000</code>,
            then try again.
          </div>
        </div>
      )}
    </div>
  )
}
