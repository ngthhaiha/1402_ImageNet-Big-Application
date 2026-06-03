# DATABASE_SCHEMA.md
# SQLite Schema — Human Anomaly Detection System (v2)

> Schema này đã được chốt. Worker và FastAPI đều dùng chung file SQLite này.
> Không thêm bảng hoặc cột nếu chưa có trong tài liệu này.

---

## Bảng: `videos`

```sql
CREATE TABLE videos (
    id            TEXT PRIMARY KEY,     -- Format: YYYYMMDD_HHMMSS_xxxx
    batch_id      TEXT,                 -- FK → batches.id
    filename      TEXT NOT NULL,        -- Tên file gốc
    name          TEXT NOT NULL,        -- Tên do user nhập (mặc định = filename)
    description   TEXT,
    location      TEXT,
    file_path     TEXT NOT NULL,        -- uploads/{id}.{ext}
    file_size     INTEGER,              -- Bytes
    duration      REAL,                 -- Giây (float), update sau khi xử lý xong
    status        TEXT NOT NULL DEFAULT 'WAITING',
    -- Allowed: WAITING | PROCESSING | PENDING_CONFIRM | COMPLETED | FAILED
    progress_step TEXT NOT NULL DEFAULT 'WAITING',
    -- Allowed: WAITING | PHASE1_START | PHASE1_DONE | PHASE2_DONE | PENDING_CONFIRM | FAILED
    -- Frontend map: WAITING=0%, PHASE1_START=10%, PHASE1_DONE=50%, PHASE2_DONE=90%, PENDING_CONFIRM=100%
    error_message TEXT,                 -- Lưu khi status = FAILED
    created_at    TEXT NOT NULL,        -- ISO 8601
    updated_at    TEXT NOT NULL         -- ISO 8601, cập nhật mỗi khi status/progress_step đổi
);
```

**Status flow**:
```
WAITING → PROCESSING → PENDING_CONFIRM → COMPLETED
                     → FAILED
```

**progress_step → % map (frontend)**:
```
WAITING        →  0%  (gray)
PHASE1_START   → 10%  (blue)
PHASE1_DONE    → 50%  (blue)
PHASE2_DONE    → 90%  (blue)
PENDING_CONFIRM→ 100% (green)
FAILED         →  —   (red)
```

---

## Bảng: `batches`

Nhóm các video được upload cùng lúc.

```sql
CREATE TABLE batches (
    id           TEXT PRIMARY KEY,   -- Format: BCH-{YYYYMMDD_HHMMSS}
    name         TEXT,               -- Tên batch (tự gen từ filename đầu tiên)
    total_videos INTEGER NOT NULL,
    created_at   TEXT NOT NULL
);
```

---

## Bảng: `processing_jobs`

```sql
CREATE TABLE processing_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT NOT NULL UNIQUE REFERENCES videos(id),
    status      TEXT NOT NULL DEFAULT 'PENDING',
    -- Allowed: PENDING | RUNNING | COMPLETED | FAILED
    -- COMPLETED = worker xong Phase 2 (video ở PENDING_CONFIRM)
    -- Không có status PENDING_CONFIRM ở đây — đó là video status
    started_at  TEXT,       -- ISO 8601, khi worker bắt đầu
    finished_at TEXT,       -- ISO 8601, khi worker kết thúc (dù COMPLETED hay FAILED)
    created_at  TEXT NOT NULL
);
```

**Worker query**: `SELECT * FROM processing_jobs WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT 1`

---

## Bảng: `anomaly_segments`

```sql
CREATE TABLE anomaly_segments (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id             TEXT NOT NULL REFERENCES videos(id),
    segment_index        INTEGER NOT NULL,   -- 0-based
    start_time           REAL NOT NULL,      -- Giây
    end_time             REAL NOT NULL,      -- Giây
    anomaly_score        REAL NOT NULL,      -- 0.0–1.0 (Phase 1)
    predicted_class      TEXT NOT NULL,      -- Label từ Phase 2 (từ danh sách 15)
    confidence_score     REAL NOT NULL,      -- 0.0–1.0 (Phase 2)
    -- Feedback fields (null cho đến khi user Submit Feedback)
    is_correct           INTEGER,            -- NULL | 1 (Correct) | 0 (Incorrect)
    verified_label       TEXT,               -- NULL cho đến khi feedback
    other_description    TEXT,               -- Chỉ có giá trị khi verified_label = 'Other'
    investigator_comment TEXT,               -- Textarea từ Feedback panel (optional)
    feedback_submitted_at TEXT,              -- ISO 8601, null cho đến khi submit
    review_status        TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    -- Allowed: PENDING_REVIEW | LABEL_CORRECT | CORRECTED | LOGGED
    created_at           TEXT NOT NULL
);
```

**review_status logic** (ưu tiên: LOGGED > CORRECTED > LABEL_CORRECT):

| review_status | Điều kiện |
|---|---|
| `PENDING_REVIEW` | `feedback_submitted_at` IS NULL |
| `LABEL_CORRECT` | is_correct=1 AND verified_label = predicted_class AND comment rỗng |
| `CORRECTED` | verified_label ≠ predicted_class AND comment rỗng |
| `LOGGED` | Đã feedback AND investigator_comment có nội dung (không rỗng) |

**Allowed values cho `predicted_class` và `verified_label`**:
```
Abuse | Arrest | Arson | Assault | Burglary | Explosion | Fighting |
RoadAccidents | Robbery | Shooting | Shoplifting | Stealing | Vandalism |
Normal | Other
```

---

## Index

```sql
CREATE INDEX idx_segments_video_id ON anomaly_segments(video_id);
CREATE INDEX idx_jobs_video_id ON processing_jobs(video_id);
CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_batch_id ON videos(batch_id);
CREATE INDEX idx_videos_created_at ON videos(created_at);
```

---

## Quan hệ

```
batches (1) ──── (N) videos
videos  (1) ──── (1) processing_jobs
videos  (1) ──── (N) anomaly_segments
```

---

## Ví dụ dữ liệu mẫu

### videos
```
id: "20250115_103045_0001"
batch_id: "BCH-20250115_103045"
filename: "parking_lot_cam1.mp4"
status: "PENDING_CONFIRM"        ← đã xử lý xong, chờ user review
progress_step: "PENDING_CONFIRM" ← 100% green
duration: 180.0
```

### anomaly_segments (chưa feedback)
```
id: 1, video_id: "20250115_103045_0001"
segment_index: 0, start_time: 45.2, end_time: 58.7
anomaly_score: 0.87, predicted_class: "Fighting", confidence_score: 0.94
is_correct: null, verified_label: null, other_description: null
investigator_comment: null, feedback_submitted_at: null
review_status: "PENDING_REVIEW"
```

### anomaly_segments (đã feedback — LOGGED)
```
id: 2, video_id: "20250115_103045_0001"
segment_index: 1, start_time: 78.2, end_time: 91.4
anomaly_score: 0.73, predicted_class: "Robbery", confidence_score: 0.81
is_correct: 0, verified_label: "Assault"
investigator_comment: "Victim pushed against wall, not robbery"
feedback_submitted_at: "2025-01-15T10:45:00"
review_status: "LOGGED"
```
