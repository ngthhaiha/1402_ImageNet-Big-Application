import { type ChangeEvent, type DragEvent, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import { FileVideo, FolderOpen, Info, Trash2, UploadCloud } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { probeVideoDuration, uploadVideos } from '../api/api'
import { ProgressBar } from '../components/ProgressBar'
import { StatusBadge } from '../components/StatusBadge'
import { Toast } from '../components/Toast'
import type { ApiResponse } from '../types/types'

const ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov'] as const
const MAX_FILES_PER_BATCH = 3
const MAX_BATCH_SIZE_BYTES = 300 * 1024 * 1024
const MAX_VIDEO_DURATION_SECONDS = 300
const PROCESSING_MINUTES_PER_VIDEO = 15

type UploadItemStatus = 'checking' | 'ready' | 'invalid' | 'uploading' | 'uploaded' | 'failed'
type ToastVariant = 'success' | 'error' | 'info'

interface UploadFileItem {
  id: string
  file: File
  duration: number | null
  status: UploadItemStatus
  progress: number
  error: string | null
}

interface ToastState {
  title: string
  message?: string
  variant: ToastVariant
}

function getFileExtension(filename: string): string {
  const extensionStart = filename.lastIndexOf('.')
  return extensionStart >= 0 ? filename.slice(extensionStart).toLowerCase() : ''
}

function getFileBaseName(filename: string): string {
  const extensionStart = filename.lastIndexOf('.')
  return extensionStart >= 0 ? filename.slice(0, extensionStart) : filename
}

function isSupportedFormat(file: File): boolean {
  return ALLOWED_EXTENSIONS.includes(getFileExtension(file.name) as (typeof ALLOWED_EXTENSIONS)[number])
}

function formatFileSize(bytes: number): string {
  const megabytes = bytes / (1024 * 1024)
  if (megabytes >= 1024) {
    return `${(megabytes / 1024).toFixed(2)} GB`
  }

  return `${megabytes.toFixed(1)} MB`
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return '--'
  }

  const rounded = Math.round(seconds)
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const remainingSeconds = rounded % 60
  return [hours, minutes, remainingSeconds].map((part) => String(part).padStart(2, '0')).join(':')
}

function createFileId(file: File, index: number): string {
  return `${file.name}-${file.size}-${file.lastModified}-${Date.now()}-${index}`
}

function readBrowserVideoDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const video = document.createElement('video')
    const objectUrl = URL.createObjectURL(file)

    const cleanup = () => {
      URL.revokeObjectURL(objectUrl)
      video.removeAttribute('src')
      video.load()
    }

    video.preload = 'metadata'
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : null
      cleanup()
      resolve(duration)
    }
    video.onerror = () => {
      cleanup()
      resolve(null)
    }
    video.src = objectUrl
  })
}

async function readVideoDuration(file: File): Promise<number> {
  const browserDuration = await readBrowserVideoDuration(file)
  if (browserDuration !== null) {
    return browserDuration
  }

  const response = await probeVideoDuration(file)
  if (!response.success || response.data === null) {
    throw new Error(response.message)
  }

  return response.data.duration
}

function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError<ApiResponse<unknown>>(error)) {
    return error.response?.data?.message ?? error.message
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'Upload failed'
}

export function Upload() {
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [items, setItems] = useState<UploadFileItem[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [toast, setToast] = useState<ToastState | null>(null)

  const totalSize = useMemo(
    () => items.reduce((sum, item) => sum + item.file.size, 0),
    [items],
  )
  const readyItems = useMemo(
    () => items.filter((item) => item.status === 'ready'),
    [items],
  )
  const estimatedMinutes = Math.max(readyItems.length, items.length) * PROCESSING_MINUTES_PER_VIDEO
  const canUpload = readyItems.length > 0 && !isUploading

  function showToast(nextToast: ToastState) {
    setToast(nextToast)
  }

  function handleBrowseClick() {
    if (!isUploading) {
      fileInputRef.current?.click()
    }
  }

  function validateBatchBeforeAdd(files: File[]): boolean {
    if (files.length === 0) {
      return false
    }

    if (items.length + files.length > MAX_FILES_PER_BATCH) {
      showToast({
        title: 'Maximum 3 videos per batch',
        message: 'Remove a file before adding another video.',
        variant: 'error',
      })
      return false
    }

    const incomingSize = files.reduce((sum, file) => sum + file.size, 0)
    if (totalSize + incomingSize > MAX_BATCH_SIZE_BYTES) {
      showToast({
        title: 'Total batch size exceeds 300 MB',
        message: 'Remove files or choose smaller videos.',
        variant: 'error',
      })
      return false
    }

    return true
  }

  function addFiles(fileList: FileList | File[]) {
    const nextFiles = Array.from(fileList)
    if (isUploading || !validateBatchBeforeAdd(nextFiles)) {
      return
    }

    const nextItems = nextFiles.map<UploadFileItem>((file, index) => {
      const isFormatValid = isSupportedFormat(file)
      return {
        id: createFileId(file, index),
        file,
        duration: null,
        status: isFormatValid ? 'checking' : 'invalid',
        progress: 0,
        error: isFormatValid ? null : 'Unsupported format',
      }
    })

    setItems((currentItems) => [...currentItems, ...nextItems])

    nextItems
      .filter((item) => item.status === 'checking')
      .forEach((item) => {
        void readVideoDuration(item.file)
          .then((duration) => {
            setItems((currentItems) =>
              currentItems.map((currentItem) => {
                if (currentItem.id !== item.id) {
                  return currentItem
                }

                if (duration > MAX_VIDEO_DURATION_SECONDS) {
                  return {
                    ...currentItem,
                    duration,
                    status: 'invalid',
                    error: 'Exceeds 5 min limit',
                  }
                }

                return {
                  ...currentItem,
                  duration,
                  status: 'ready',
                  error: null,
                }
              }),
            )
          })
          .catch((error: unknown) => {
            const message = getErrorMessage(error)
            setItems((currentItems) =>
              currentItems.map((currentItem) =>
                currentItem.id === item.id
                  ? {
                      ...currentItem,
                      duration: null,
                      status: 'invalid',
                      error: message,
                    }
                  : currentItem,
              ),
            )
          })
      })
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      addFiles(event.target.files)
    }
    event.target.value = ''
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    if (!isUploading) {
      setIsDragging(true)
    }
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    if (event.currentTarget === event.target) {
      setIsDragging(false)
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    addFiles(event.dataTransfer.files)
  }

  function removeItem(itemId: string) {
    if (!isUploading) {
      setItems((currentItems) => currentItems.filter((item) => item.id !== itemId))
    }
  }

  function clearQueue() {
    if (!isUploading) {
      setItems([])
      setToast(null)
    }
  }

  async function handleUpload() {
    if (!canUpload) {
      return
    }

    const uploadItems = readyItems
    const uploadItemIds = new Set(uploadItems.map((item) => item.id))
    const totalUploadSize = uploadItems.reduce((sum, item) => sum + item.file.size, 0)

    setIsUploading(true)
    setToast(null)
    setItems((currentItems) =>
      currentItems.map((item) =>
        uploadItemIds.has(item.id)
          ? { ...item, status: 'uploading', progress: 0, error: null }
          : item,
      ),
    )

    try {
      const response = await uploadVideos(
        uploadItems.map((item) => item.file),
        uploadItems.map((item) => ({
          name: getFileBaseName(item.file.name),
          duration: item.duration ?? undefined,
        })),
        (event) => {
          const total = event.total ?? totalUploadSize
          const progress = total > 0 ? Math.round((event.loaded / total) * 100) : 0
          setItems((currentItems) =>
            currentItems.map((currentItem) =>
              uploadItemIds.has(currentItem.id)
                ? { ...currentItem, progress: Math.min(100, progress) }
                : currentItem,
            ),
          )
        },
      )

      if (!response.success || response.data === null) {
        throw new Error(response.message)
      }

      setItems((currentItems) =>
        currentItems.map((currentItem) =>
          uploadItemIds.has(currentItem.id)
            ? { ...currentItem, status: 'uploaded', progress: 100, error: null }
            : currentItem,
        ),
      )
      navigate(`/queue?batch_id=${encodeURIComponent(response.data.batch.id)}`)
    } catch (error) {
      const message = getErrorMessage(error)
      setIsUploading(false)
      setItems((currentItems) =>
        currentItems.map((currentItem) =>
          uploadItemIds.has(currentItem.id)
            ? { ...currentItem, status: 'failed', progress: 0, error: message }
            : currentItem,
        ),
      )
      showToast({
        title: 'Upload failed',
        message,
        variant: 'error',
      })
    }
  }

  function renderStatus(item: UploadFileItem) {
    if (item.status === 'uploading' || item.status === 'uploaded' || item.status === 'failed') {
      return (
        <ProgressBar
          value={item.status === 'failed' ? 0 : item.progress}
          label={item.status === 'uploaded' ? 'Uploaded' : item.status === 'failed' ? 'Failed' : 'Uploading'}
        />
      )
    }

    if (item.status === 'checking') {
      return (
        <span className="inline-flex items-center rounded-full bg-blue-100 px-2.5 py-1 text-xs font-semibold text-blue-800 ring-1 ring-inset ring-blue-200">
          Checking
        </span>
      )
    }

    return (
      <span title={item.error ?? undefined}>
        <StatusBadge status={item.status === 'ready' ? 'Ready' : 'Invalid Format'} />
        {item.error ? <span className="mt-1 block text-xs text-red-800">{item.error}</span> : null}
      </span>
    )
  }

  return (
    <section className="min-h-screen bg-[#FAF8FF] px-8 py-8 text-[#131B2E]">
      <Toast
        open={toast !== null}
        title={toast?.title ?? ''}
        message={toast?.message}
        variant={toast?.variant}
        onClose={() => setToast(null)}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <div>
          <p className="text-sm font-medium text-[#434655]">Cases / Upload Video</p>
          <h2 className="mt-2 text-3xl font-semibold text-[#131B2E]">Upload Surveillance Videos</h2>
          <p className="mt-2 text-base text-[#434655]">
            Add video files to the processing queue for automated anomaly detection.
          </p>
        </div>

        <div
          className={[
            'flex min-h-[320px] flex-col items-center justify-center rounded-xl border-2 border-dashed bg-white px-8 py-12 text-center transition',
            isDragging ? 'border-[#004AC6] bg-[#F2F3FF]' : 'border-[#C3C6D7]',
          ].join(' ')}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="mb-6 inline-flex h-16 w-16 items-center justify-center rounded-full bg-[#DBE1FF] text-[#004AC6]">
            <UploadCloud className="h-8 w-8" aria-hidden="true" />
          </div>
          <h3 className="text-xl font-semibold text-[#131B2E]">Drop videos here</h3>
          <p className="mt-2 max-w-2xl text-sm text-[#434655]">
            Support MP4, AVI, MOV. Maximum 3 videos per batch, total batch size up to 300 MB,
            maximum 5 minutes per video.
          </p>
          <button
            type="button"
            className="mt-8 inline-flex items-center justify-center gap-2 rounded-lg bg-[#004AC6] px-8 py-4 text-base font-semibold text-white shadow-md transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={handleBrowseClick}
            disabled={isUploading}
          >
            <FolderOpen className="h-5 w-5" aria-hidden="true" />
            Browse Files
          </button>
          <input
            ref={fileInputRef}
            className="hidden"
            type="file"
            accept=".mp4,.avi,.mov"
            multiple
            onChange={handleFileChange}
          />
        </div>

        {items.length > 0 ? (
          <div className="overflow-hidden rounded-xl border border-[#C3C6D7] bg-white shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#C3C6D7] bg-[#F2F3FF] px-6 py-4">
              <h3 className="text-base font-semibold text-[#131B2E]">Queue ({items.length} selected)</h3>
              <p className="text-xs font-medium uppercase tracking-wide text-[#434655]">
                Total: {formatFileSize(totalSize)}
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] border-collapse">
                <thead>
                  <tr className="border-b border-[#C3C6D7] text-left text-xs font-medium uppercase tracking-wide text-[#434655]">
                    <th className="px-6 py-4">File Name</th>
                    <th className="px-6 py-4">Size</th>
                    <th className="px-6 py-4">Duration</th>
                    <th className="px-6 py-4">Status</th>
                    <th className="px-6 py-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr key={item.id} className="border-b border-[#C3C6D7] last:border-b-0">
                      <td className="px-6 py-4">
                        <div className="flex min-w-0 items-center gap-3">
                          <FileVideo className="h-5 w-5 shrink-0 text-[#434655]" aria-hidden="true" />
                          <span className="min-w-0 truncate text-sm text-[#131B2E]">{item.file.name}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-[#434655]">{formatFileSize(item.file.size)}</td>
                      <td className="px-6 py-4 text-sm text-[#434655]">{formatDuration(item.duration)}</td>
                      <td className="px-6 py-4">{renderStatus(item)}</td>
                      <td className="px-6 py-4 text-right">
                        <button
                          type="button"
                          className="inline-flex h-9 w-9 items-center justify-center rounded-full text-[#434655] transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
                          aria-label={`Remove ${item.file.name}`}
                          onClick={() => removeItem(item.id)}
                          disabled={isUploading}
                        >
                          <Trash2 className="h-5 w-5" aria-hidden="true" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex max-w-2xl items-center gap-4 rounded-lg border border-[#B7C8E1] bg-[#D3E4FE] p-4 text-sm text-[#0B1C30]">
            <Info className="h-5 w-5 shrink-0 text-[#004AC6]" aria-hidden="true" />
            <p>
              Videos will be processed sequentially in the order uploaded. Estimated processing time:{' '}
              {estimatedMinutes} minutes.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-lg border border-[#C3C6D7] bg-white px-6 py-4 text-base font-semibold text-[#434655] transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={clearQueue}
              disabled={isUploading || items.length === 0}
            >
              Cancel
            </button>
            <button
              type="button"
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#004AC6] px-8 py-4 text-base font-semibold text-white shadow-md transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleUpload}
              disabled={!canUpload}
            >
              Upload & Analyze
              <UploadCloud className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
