/**
 * The local API. Every call goes to 127.0.0.1 via the Vite proxy — there is no
 * other origin in this app, no CDN, no font host, no analytics. That is
 * deliberate: the product's claim is that nothing leaves the machine, and a
 * frontend that phones home for a typeface would make that claim a lie.
 *
 * Types mirror backend/app/schemas.py. Routes the UI expects:
 *
 *   GET  /api/status                   -> Status
 *   POST /api/index      IndexRequest  -> IndexProgress   (starts a job)
 *   GET  /api/index/{job_id}           -> IndexProgress   (poll)
 *   GET  /api/photos?offset=&limit=    -> PhotoPage
 *   GET  /api/photos/{id}              -> PhotoRecord
 *   GET  /api/photos/{id}/image[?w=N]  -> image bytes
 *   POST /api/query      QueryRequest  -> QueryResponse
 *   POST /api/feedback   FeedbackRequest -> {ok: true}
 */

export type Box = { x1: number; y1: number; x2: number; y2: number }
export type Gps = { lat: number; lon: number }

export type IndexStage =
  | 'scanning' | 'extracting_frames' | 'embedding' | 'building_index' | 'done' | 'error'

export type IndexProgress = {
  job_id: string
  stage: IndexStage
  total_files: number
  processed_files: number
  total_frames: number
  processed_frames: number
  message: string
  error: string | null
}

export type PhotoRecord = {
  id: string
  path: string
  source_video_path: string | null
  frame_time_sec: number | null
  taken_at: string | null
  gps: Gps | null
  width: number
  height: number
}

export type Match = {
  photo_id: string
  score: number
  inlier_count: number
  verified: boolean
  bbox: Box
  taken_at: string | null
  gps: Gps | null
}

export type QueryResponse = {
  query_id: string
  matches: Match[]
  last_seen: Match | null
}

export type Status = { indexed: boolean; photo_count: number; folder_path: string | null }
export type PhotoPage = { total: number; photos: PhotoRecord[] }

class ApiError extends Error {}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new ApiError('Local server not reachable at 127.0.0.1:8000.')
  }
  if (!res.ok) throw new ApiError(`${res.status} ${await res.text().catch(() => res.statusText)}`)
  return res.json() as Promise<T>
}

const post = <T,>(url: string, body: unknown) =>
  json<T>(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })

export const getStatus = () => json<Status>('/api/status')

export const startIndex = (folder_path: string) =>
  post<IndexProgress>('/api/index', { folder_path, video_frame_interval_sec: 2.0 })

export const getProgress = (jobId: string) => json<IndexProgress>(`/api/index/${jobId}`)

export const getPhotos = (offset = 0, limit = 500) =>
  json<PhotoPage>(`/api/photos?offset=${offset}&limit=${limit}`)

export const getPhoto = (id: string) => json<PhotoRecord>(`/api/photos/${encodeURIComponent(id)}`)

export const imageUrl = (id: string, w?: number) =>
  `/api/photos/${encodeURIComponent(id)}/image${w ? `?w=${w}` : ''}`

/** `object_id` is a previous QueryResponse.query_id — pass it to search with
 *  everything the user has confirmed or rejected for that object so far. */
export const query = (source_photo_id: string, box: Box, object_id?: string, top_k = 200) =>
  post<QueryResponse>('/api/query', { source_photo_id, box, top_k, object_id })

export const sendFeedback = (query_id: string, photo_id: string, confirmed: boolean, box?: Box) =>
  post<{ ok: boolean; exemplars: number }>('/api/feedback', {
    query_id,
    photo_id,
    confirmed,
    box,
  })
