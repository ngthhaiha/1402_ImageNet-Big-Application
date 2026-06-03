# REQUIREMENTS.md
# Human Anomaly Detection and Classification Web System

## Project Summary

Web app demo cho phép người dùng upload video giám sát, hệ thống tự động chạy AI pipeline để phát hiện và phân loại hành vi bất thường. Người dùng xem kết quả, xác nhận/sửa nhãn, và xem thống kê tổng hợp.

**Dùng cho**: Demo đề tài nghiên cứu khoa học — không phải production.  
**Không có login / phân quyền**: Tất cả người dùng có quyền như nhau.

---

## Tech Stack (đã chốt, không thay đổi)

| Layer | Choice |
|---|---|
| Frontend | React + Vite + TypeScript |
| Backend | FastAPI (Python) |
| Database | SQLite (file local) |
| Storage | Thư mục `uploads/` local |
| AI Pipeline | Worker riêng (`worker.py`) import trực tiếp PyTorch pipeline |
| Async | FastAPI BackgroundTasks |
| Charts | Recharts |
| ORM | SQLAlchemy (sync, simple) |

---

## In Scope

- Upload batch video (tối đa 3 video / lần) và validate format
- Tự động tạo Processing Job sau upload
- GPU concurrency = 1: xử lý lần lượt, video sau chờ (WAITING) cho đến khi video trước xong
- Worker chạy AI pipeline (Phase 1 + Phase 2) ghi kết quả vào SQLite
- Hiển thị tiến trình xử lý realtime bằng polling (progress step)
- Queue Analyze page: xem toàn bộ queue + trạng thái từng video
- Video player + timeline anomaly + bảng segments + panel Investigation Summary
- Jump to segment theo timestamp
- Validate kết quả AI: Feedback & Validation form → Submit → hiển thị Feedback Detail
- Edit Feedback với Save/Cancel
- Investigator comments khi feedback
- Xuất báo cáo (Export Report) cho từng video (JSON)
- Profile page: stats thật từ DB + recent activity log + notification toggles (localStorage)
- Dashboard thống kê
- Lưu feedback dataset

## Out of Scope (không implement, không thêm)

- Login / authentication / phân quyền
- Thanh toán
- Email notification
- Export video clip (chỉ export report JSON, không export video)
- Multi-user / multi-tenant
- Xử lý realtime từ camera stream
- Model training trong app
- REST API public / API key
- Docker / deployment config
- Bất kỳ tính năng nào không được liệt kê trong FRD

---

## Danh sách Anomaly Labels (toàn hệ thống)

Dùng cho cả `predicted_class` (Phase 2) và `verified_label` (feedback).  
**Tổng 15 options — bắt buộc dùng đúng danh sách này, không thêm bớt:**

| # | Label | Loại |
|---|---|---|
| 1 | Abuse | Anomaly |
| 2 | Arrest | Anomaly |
| 3 | Arson | Anomaly |
| 4 | Assault | Anomaly |
| 5 | Burglary | Anomaly |
| 6 | Explosion | Anomaly |
| 7 | Fighting | Anomaly |
| 8 | RoadAccidents | Anomaly |
| 9 | Robbery | Anomaly |
| 10 | Shooting | Anomaly |
| 11 | Shoplifting | Anomaly |
| 12 | Stealing | Anomaly |
| 13 | Vandalism | Anomaly |
| 14 | Normal | Normal |
| 15 | Other | Special — yêu cầu nhập text mô tả |

---

## Video Status Flow (toàn hệ thống)

```
WAITING → PROCESSING → PENDING_CONFIRM → COMPLETED
                     → FAILED
```

| Status | Ý nghĩa | Trigger |
|---|---|---|
| `WAITING` | Video đã upload vào DB, đang chờ đến lượt xử lý theo thứ tự FIFO | Ngay sau khi upload |
| `PROCESSING` | Worker đang chạy AI pipeline (GPU đang chạy) | Worker poll job |
| `PENDING_CONFIRM` | Pipeline xong, có kết quả, đang chờ user submit Feedback | Worker hoàn thành Phase 2 |
| `COMPLETED` | User đã submit Feedback, case đã được xác nhận | User nhấn Submit Feedback |
| `FAILED` | Pipeline lỗi | Exception trong worker |

**Lưu ý quan trọng**:
- `COMPLETED` được trigger bởi user action (Submit Feedback), không phải worker
- Video ở trạng thái `PENDING_CONFIRM` vẫn có thể xem và tương tác hoàn toàn
- Worker chỉ lấy video tiếp theo sau khi video hiện tại đạt `PENDING_CONFIRM` hoặc `FAILED`

**Progress step** (lưu trong cột `progress_step` của bảng `videos`, frontend map sang %):

| progress_step | % hiển thị | Màu bar |
|---|---|---|
| `WAITING` | 0% | Gray |
| `PHASE1_START` | 10% | Blue |
| `PHASE1_DONE` | 50% | Blue |
| `PHASE2_DONE` | 90% | Blue |
| `PENDING_CONFIRM` | 100% | Green |
| `FAILED` | — | Red |

Worker update `progress_step` sau mỗi bước. Frontend polling đọc `progress_step` và render progress bar tương ứng — không cần biết bên trong pipeline chạy đến đâu.

---

## Functional Requirements

### FR01 — Upload Video (Batch)

**Trigger**: Người dùng nhấn nút "Upload & Analyze".

**Input**:
- Danh sách video files (bắt buộc, 1–3 file)
- Video name: lấy từ filename (optional, có thể sửa)
- Description (optional)
- Location (optional)

**Batch Rules** (hard limits, validate cả frontend lẫn backend):

| Rule | Giá trị |
|---|---|
| Số video tối đa / batch | 3 video |
| Dung lượng tối đa / batch | 300 MB (tổng) |
| Thời lượng tối đa / video | 5 phút (300 giây) — hard limit |
| Thời lượng khuyến nghị | 30 giây – 5 phút |
| Supported formats | `.mp4`, `.avi`, `.mov` |

**Upload Page UI**:
- Drag-and-drop zone + nút Browse Files
- Queue table: Filename, Size, Duration, Status badge (Ready / Invalid Format), nút xóa
- File format sai → badge "Invalid Format" (đỏ), không thể upload file đó
- Video > 5 phút → badge "Invalid Format" + tooltip "Exceeds 5 min limit"
- Tổng dung lượng batch hiển thị góc phải
- Info bar: "Videos will be processed sequentially."
- Nút Cancel + nút **Upload & Analyze** (disabled nếu không có file Ready)
- Khi đang upload: mỗi file hiển thị progress bar riêng (HTTP upload progress)
- Upload xong tất cả → tự động redirect `/queue`

**Process**:
1. Validate batch rules phía frontend → block nếu vi phạm
2. Upload từng file lên backend (song song HTTP)
3. Backend validate lại từng file
4. Lưu file vào `uploads/{video_id}.{ext}`
5. Generate Video ID: `{YYYYMMDD_HHMMSS}_{xxxx}`
6. Tạo Batch record
7. Tạo Video record với status = `WAITING`, progress_step = `WAITING`
8. Tạo ProcessingJob record với status = `PENDING`
9. BackgroundTask khởi động worker loop nếu chưa chạy

**Ví dụ batch 3 video**:
```
video_1: PROCESSING   ← worker đang chạy
video_2: WAITING      ← chờ
video_3: WAITING      ← chờ
→ video_1 PENDING_CONFIRM → worker lấy video_2
→ video_2 PENDING_CONFIRM → worker lấy video_3
```

**Acceptance Criteria**:
- AC1: Upload 1–3 file `.mp4 / .avi / .mov` thành công
- AC2: File format không hỗ trợ → badge "Invalid Format", không upload
- AC3: Batch > 3 video → block + thông báo
- AC4: Tổng batch > 300 MB → block + thông báo
- AC5: Video > 5 phút → badge "Invalid Format"
- AC6: Mỗi file hiển thị progress bar HTTP riêng khi uploading
- AC7: Video ID gen đúng format
- AC8: File lưu vào `uploads/`
- AC9: ProcessingJob được tạo cho từng video
- AC10: Chỉ 1 video PROCESSING tại một thời điểm, còn lại WAITING

---

### FR02 — AI Processing Pipeline (Worker)

**Trigger**: Worker loop tự động poll job queue.

**Concurrency rule**: Worker chỉ chạy 1 job tại một thời điểm. Sau khi job hiện tại đạt `PENDING_CONFIRM` hoặc `FAILED`, worker mới lấy job tiếp theo.

**Worker flow** (`worker.py`):
```
1. Poll: SELECT job WHERE status='PENDING' ORDER BY created_at ASC LIMIT 1
2. Update job.status = RUNNING
   Update video.status = PROCESSING, video.progress_step = PHASE1_START
3. Chạy Phase 1: Temporal Anomaly Detection
   → trả về: list[{start_time, end_time, anomaly_score}]
   Update video.progress_step = PHASE1_DONE
4. Chạy Phase 2: Classification từng segment bất thường
   → trả về: {predicted_class, confidence_score} cho mỗi segment
   Update video.progress_step = PHASE2_DONE
5. Ghi tất cả segments vào bảng anomaly_segments
6. Update video.status = PENDING_CONFIRM
   Update video.progress_step = PENDING_CONFIRM
   Update job.status = COMPLETED, job.finished_at = now()
7. [Nếu exception ở bất kỳ bước nào]
   Update video.status = FAILED, video.error_message = str(exception)
   Update job.status = FAILED, job.finished_at = now()
8. Lặp lại từ bước 1
```

**Output ghi vào DB** (mỗi segment):
- `video_id`, `segment_index` (0-based)
- `start_time`, `end_time` (float, seconds)
- `anomaly_score` (float 0–1, từ Phase 1)
- `predicted_class` (string, từ danh sách 15 labels)
- `confidence_score` (float 0–1, từ Phase 2)
- `verified_label` = null, `is_correct` = null
- `other_description` = null, `investigator_comment` = null
- `feedback_submitted_at` = null
- `review_status` = `PENDING_REVIEW`

**Acceptance Criteria**:
- AC11: Worker tự động chạy, không cần user trigger
- AC12: Concurrency = 1, không bao giờ có 2 video PROCESSING cùng lúc
- AC13: `progress_step` cập nhật từng bước, frontend polling hiển thị progress bar đúng %
- AC14: Xử lý thành công → status = `PENDING_CONFIRM`, segments lưu đầy đủ
- AC15: Exception → status = `FAILED`, error_message lưu
- AC16: Sau PENDING_CONFIRM hoặc FAILED → worker tự động lấy job tiếp theo

**Stub pattern** (dùng khi chưa integrate pipeline thật):
```python
def run_phase1(video_path: str) -> list[dict]:
    # REPLACE with actual PyTorch pipeline import
    # Must return: [{"start_time": float, "end_time": float, "anomaly_score": float}]
    return [
        {"start_time": 10.5, "end_time": 25.0, "anomaly_score": 0.87},
        {"start_time": 78.2, "end_time": 91.4, "anomaly_score": 0.73},
    ]

def run_phase2(video_path: str, segment: dict) -> dict:
    # REPLACE with actual PyTorch pipeline import
    # Must return: {"predicted_class": str, "confidence_score": float}
    return {"predicted_class": "Fighting", "confidence_score": 0.91}
```

---

### FR03 — Queue Analyze Page

**Route**: `/queue`

**Header — Active Batch Panel**:
- Batch ID, tên batch, thời gian uploaded
- Completion % = số video `PENDING_CONFIRM` + `COMPLETED` / tổng × 100
- Progress bar tổng, text "X of Y Videos Processed"

**4 Summary Cards**:
| Card | Đếm video có status |
|---|---|
| DONE | `PENDING_CONFIRM` + `COMPLETED` |
| ACTIVE | `PROCESSING` |
| QUEUED | `WAITING` |
| ERRORS | `FAILED` |

**Queue Details Table**:
| Cột | Nội dung |
|---|---|
| Video Name | Thumbnail + tên file |
| Status | Badge theo status |
| Progress | Progress bar theo `progress_step` (map sang %) |
| Duration | HH:MM:SS |
| Submitted | HH:MM:SS |
| Action | Nút theo status |

**Status Badge colors**:
- `WAITING` → gray
- `PROCESSING` → yellow/blue (đang chạy)
- `PENDING_CONFIRM` → orange (chờ user xác nhận)
- `COMPLETED` → green
- `FAILED` → red

**Action buttons**:
- `WAITING` → "—" (disabled)
- `PROCESSING` → nút **Monitor** → `/videos/:id`
- `PENDING_CONFIRM` → nút **Review** → `/videos/:id` (highlighted, CTA chính)
- `COMPLETED` → nút **View Detail** → `/videos/:id`
- `FAILED` → nút **Retry**

**Pagination**: 10 rows / page  
**Auto-refresh**: polling mỗi 5 giây (dừng khi tất cả COMPLETED hoặc FAILED)

**Acceptance Criteria**:
- AC17: Completion % tính đúng (tính cả PENDING_CONFIRM)
- AC18: 4 summary cards đúng số liệu
- AC19: Status badge đúng màu
- AC20: Progress bar đúng theo `progress_step`
- AC21: Retry → tạo lại ProcessingJob mới, reset video về WAITING
- AC22: Auto-refresh hoạt động

---

### FR04 — Video Detail Page

**Route**: `/videos/:id`

**Header**:
- Breadcrumb: "Analysis Queue / Segment Review"
- Tiêu đề: "Video Investigation"
- Nút **Archive Case** (outline) + Nút **Export Report** (primary, → FR08)

**Layout**: 2 cột (65% trái / 35% phải).

---

**CỘT TRÁI**:

**4a. Video Player**
- HTML5 video player, source `/uploads/{video_id}`
- Controls: play/pause/seek/volume/fullscreen
- Overlay badge (góc trên trái, chỉ hiện khi currentTime trong range segment):
  `● ANOMALY DETECTED: {CLASS} ({CONFIDENCE}%)`

**4b. Processing Progress** (chỉ hiện khi status = PROCESSING)
- Progress bar màu xanh biển với %
- Label bước hiện tại (vd: "Running Phase 1 Detection...")
- Polling mỗi 3 giây để cập nhật `progress_step` → %

| progress_step | Label hiển thị | % |
|---|---|---|
| PHASE1_START | Running Phase 1: Anomaly Detection... | 10% |
| PHASE1_DONE | Phase 1 Complete. Running Phase 2: Classification... | 50% |
| PHASE2_DONE | Phase 2 Complete. Saving results... | 90% |
| PENDING_CONFIRM | ✓ Analysis Complete | 100% (green) |

**4c. Analysis Timeline** (hiện khi status = `PENDING_CONFIRM` hoặc `COMPLETED`)
- Thanh ngang, scale tuyến tính theo duration
- Nền xám = toàn video; block đỏ = anomaly segments
- Legend: Normal (xám) / Detected Anomaly (đỏ)
- Timestamp markers đều nhau
- Click block đỏ → video seek + cập nhật panel phải

**4d. Detected Segments Table** (hiện khi `PENDING_CONFIRM` hoặc `COMPLETED`)

| Cột | Nội dung |
|---|---|
| Time Range | `MM:SS - MM:SS` |
| Predicted Activity | Label badge (màu theo class) |
| Confidence | `XX%` |
| Review Status | Badge: Pending Review / Label Correct / Corrected / Logged |

- Click row → cập nhật panel phải (Investigation Summary + Feedback panel)
- Row đang chọn được highlight
- Mặc định: chọn row đầu tiên khi trang load

---

**CỘT PHẢI**:

**4d. Investigation Summary Panel**
```
Investigation Summary          [Status badge]
─────────────────────────────────────────────
Video Name     {video.name}
Segment ID     #SEG-{index+1}
Predicted      ● {predicted_class}
Confidence     [████████░░] {confidence}%
Timestamp      {MM:SS} - {MM:SS}
Status         [Pending Review]
```

**4e. Feedback & Validation Panel** — 2 trạng thái:

---

**State A — Form** (khi segment chưa có feedback hoặc đang Edit):

```
Feedback & Validation
─────────────────────────────────────────────
Is the detected anomaly segment correct?
  [✓ Correct]   [✗ Incorrect]

Is the predicted activity correct?
  [✓ Label Correct]   [✎ Edit Label]
  (nếu Edit Label: dropdown 14 options, không được hiển thị option = predict activity)
  (nếu Other: textarea "Describe the activity" — bắt buộc)
(1. Bắt buộc chọn cả 2 câu hỏi.
2. Khi Q1 = Incorrect:
   - tự chọn Q2 = Edit Label
   - set verified_label = Normal
   - disable nút Label Correct
3. Khi Q1 = Correct:
   - cho chọn Label Correct hoặc Edit Label bình thường
4. Nếu Edit Label:
   - hiện dropdown label
5. Nếu chọn Other:
   - bắt buộc nhập mô tả
trong case Q1 Incorrect tự chọn Edit Label, verified label mặc định Normal, khóa Label Correct.
label mặc định là Normal và khi nhấn vào dropdown chỉ hiện Normal và Other thôi)

INVESTIGATOR COMMENTS
[Describe findings, involved parties...    ]
(optional)

[          Submit Feedback          ]  ← disabled cho đến khi chọn ≥1 option
```

**Submit Feedback** trigger:
- Lưu feedback vào DB
- Update `video.status = COMPLETED` (nếu là lần đầu submit của video này và TẤT CẢ segments đã có feedback)
  - Nếu còn segment chưa feedback → video giữ `PENDING_CONFIRM`
- Panel chuyển sang **State B — Feedback Detail**

---

**State B — Feedback Detail** (sau khi submit, hoặc khi load segment đã có feedback):

```
Feedback Detail
─────────────────────────────────────────────
Segment Detect    ● Correct  /  ✗ Incorrect
Verified Label    {verified_label}
                  (nếu Other: "{other_description}")
Investigator      {investigator_comment hoặc "—"}
Comments
Submitted At      {DD/MM/YYYY HH:mm}

[          ✎ Edit Feedback          ]
```

**Edit Feedback** flow:
- Click "Edit Feedback" → panel quay về **State A — Form**, pre-fill giá trị cũ
- Nút Submit đổi thành **Save Changes** + nút **Cancel**
- Cancel → panel quay về **State B** với dữ liệu cũ (không thay đổi)
- Save → lưu đè, quay về **State B** với dữ liệu mới

---

**4f. Video Status khi tất cả segments đã feedback**:
- Backend kiểm tra: nếu tất cả segments của video có `feedback_submitted_at != null`
  → update `video.status = COMPLETED`
- Nếu còn segment chưa feedback → giữ `PENDING_CONFIRM`

**Acceptance Criteria**:
- AC23: Video player hoạt động, overlay badge đúng segment
- AC24: Progress bar cập nhật đúng theo `progress_step` khi đang PROCESSING
- AC25: Timeline và Segments Table chỉ hiện khi PENDING_CONFIRM hoặc COMPLETED
- AC26: Click row → cập nhật Investigation Summary + Feedback panel
- AC27: Submit Feedback lưu đúng DB
- AC28: Nếu Edit Label = Other → bắt buộc nhập description, không submit nếu rỗng
- AC29: Sau Submit → panel chuyển sang Feedback Detail ngay (không reload page)
- AC30: Edit Feedback → Cancel → giữ nguyên feedback cũ
- AC31: Edit Feedback → Save → ghi đè, hiển thị dữ liệu mới
- AC32: video.status = COMPLETED chỉ khi TẤT CẢ segments đã feedback

---

### FR05 — AI Result Validation (Feedback)

**Trigger**: User click Submit Feedback (hoặc Save Changes khi Edit).

**Data lưu mỗi lần submit**:

| Field | Nguồn |
|---|---|
| `is_correct` | Câu hỏi 1: Correct=true / Incorrect=false |
| `verified_label` | Câu hỏi 2: Label Correct → predicted_class; Edit Label → label chọn |
| `other_description` | Text nhập khi verified_label = "Other" |
| `investigator_comment` | Nội dung textarea (nullable) |
| `feedback_submitted_at` | Timestamp lúc submit |
| `review_status` | Tính theo bảng mapping dưới |

**Review Status mapping**:

| review_status | Điều kiện |
|---|---|
| `PENDING_REVIEW` | Chưa có feedback (`feedback_submitted_at` = null) |
| `LABEL_CORRECT` | is_correct=true AND verified_label = predicted_class |
| `CORRECTED` | verified_label ≠ predicted_class |
| `LOGGED` | Đã feedback AND investigator_comment có nội dung |

> Ưu tiên: LOGGED > CORRECTED > LABEL_CORRECT (nếu có comment thì luôn là LOGGED)

**Acceptance Criteria**:
- AC33: Label Correct → lưu đúng, review_status = LABEL_CORRECT
- AC34: Edit Label + Other → bắt buộc nhập description
- AC35: review_status cập nhật đúng logic ưu tiên
- AC36: Feedback overwrite được (Save Changes ghi đè toàn bộ fields)
- AC37: `feedback_submitted_at` lưu timestamp lúc submit (hoặc lúc Save Changes)

---

### FR06 — Feedback Dataset

**Trigger**: Tự động mỗi khi feedback submit.

**Logic**: Mọi segment có `feedback_submitted_at != null` đều thuộc dataset.  
**Không cần UI riêng** — export thủ công từ SQLite.

**Acceptance Criteria**:
- AC38: Tất cả feedback fields lưu chính xác vào `anomaly_segments`

---

### FR07 — Dashboard

**Route**: `/dashboard`

**Tham khảo**: Figma dashboard.txt + ảnh screenshot.

---

**Welcome Banner** (full width):
- Text "Welcome back"
- Sub-text: "Monitor abnormal activities detected from surveillance videos..."
- Nút **Upload New Video** → navigate `/`

---

**4 Summary Cards** (grid ngang, equal width):

| Card | Label | Value từ DB | Sub-text | Badge | Icon color |
|---|---|---|---|---|---|
| 1 | Total Videos Analyzed | COUNT videos | "Total uploaded videos processed by AI" | +12% (hard-code) | `#004AC6` |
| 2 | Abnormal Events Detected | COUNT segments anomaly | "Total abnormal segments detected" | +3% (hard-code) | `#BA1A1A` |
| 3 | Pending Reviews | COUNT segments WHERE feedback_submitted_at IS NULL AND video.status = PENDING_CONFIRM | "Segments waiting for user validation" | "24 New" (hard-code) | `#943700` |
| 4 | Reviewed Cases | COUNT segments WHERE feedback_submitted_at IS NOT NULL | "Validated anomaly events" | "98% Acc." (hard-code) | `#505F76` |

Card 2 value: màu `#BA1A1A` (đỏ). Card 1,3,4 value: màu `#131B2E`.

---

**Row 2: 2 cột** (LEFT ~60% / RIGHT ~40%)

**LEFT — Anomaly Distribution** (Donut chart):
- Title: "Anomaly Distribution" + icon `...` (menu)
- Recharts PieChart dạng donut (innerRadius lớn)
- Center text: tổng số segments + "TOTAL"
- Legend bên phải chart: mỗi class một dòng → dot màu + label + "XX% (N)"
- Data: COUNT segments GROUP BY predicted_class (chỉ class có segments)
- Màu mỗi class: Fighting=`#BA1A1A`, Robbery=`#004AC6`, RoadAccidents=`#4CAF50`, Shooting=`#F59E0B`, Burglary=`#9C27B0`, Other=`#4DD0E1`

**RIGHT — Recent Activity** (từ `activity_log`):
- Title: "Recent Activity"
- List 5 activity gần nhất (dot màu + title + timestamp relative)
- Dot colors: UPLOAD=`#004AC6`, REVIEW_COMPLETE=`#4CAF50`, FLAG=`#BA1A1A`
- Không click (chỉ display)

---

**Row 3: 2 cột** (LEFT ~60% / RIGHT ~40%)

**LEFT — Recent Alerts table**:
- Title: "Recent Alerts" + link "View All" → (UI only, không navigate)
- Columns: Time | Activity Type | Confidence | Severity | Status
- 10 segments gần nhất có video.status = PENDING_CONFIRM hoặc COMPLETED
- **Severity column** (tính từ anomaly_score):
  - score ≥ 0.85 → badge HIGH (bg `rgba(186,26,26,0.10)`, text `#BA1A1A`)
  - score 0.65–0.84 → badge MEDIUM (bg `rgba(245,158,11,0.10)`, text `#D97706`)
  - score < 0.65 → badge LOW (bg `rgba(80,95,118,0.10)`, text `#505F76`)
- **Status column** (map từ review_status):
  - PENDING_REVIEW → dot `#943700` + text "Unreviewed"
  - PROCESSING (video đang xử lý) → text "Processing" (gray)
  - LABEL_CORRECT → dot xanh lá + text "Validated"
  - CORRECTED → dot `#004AC6` + text "Corrected"
  - LOGGED → dot `#004AC6` + text "Logged"
  - is_correct=false → text "False Positive" (gray)
- Không cần pagination cho demo (10 rows cố định)
- Row click → navigate `/videos/:video_id`

**RIGHT — Top Detections** (bar chart ngang):
- Title: "Top Detections"
- Recharts BarChart nằm ngang (layout="vertical")
- X axis: số lượng, Y axis: class name
- Bar fill: `#004AC6`
- Track (background bar): `#E2E7FF`
- Số lượng hiện ở cuối bar bên phải
- Data: COUNT segments GROUP BY predicted_class, sort DESC, top 6

---

**Row 4: Recent Investigations table** (full width):
- Title: "Recent Investigations" + sub-text "Detailed log of AI-assisted surveillance analysis"
- Buttons góc phải: **Filter** (outline) + **Export Data** (outline với icon download)
- Columns: Video Name | Detected Activity | Confidence | Review Status | Created Time
- **Video Name column**: thumbnail 48×48 (placeholder) + tên video + resolution/duration
- **Detected Activity**: text (lấy predicted_class của segment đầu tiên của video)
- **Confidence column**: mini bar (64px wide, height 6px) + % text màu theo severity
  - HIGH ≥ 85%: bar `#BA1A1A`, text `#BA1A1A`
  - MEDIUM 65–84%: bar `#F59E0B`, text `#D97706`
  - LOW < 65%: bar `#004AC6`, text `#004AC6`
- **Review Status badge**:
  - HIGH ALERT: bg `rgba(186,26,26,0.10)`, text `#BA1A1A`, border-radius 9999
  - IN REVIEW: bg `rgba(80,95,118,0.10)`, text `#505F76`
  - VALIDATED: bg `rgba(0,74,198,0.10)`, text `#004AC6` (estimate từ ảnh)
- **Created Time**: "Today, HH:MM" hoặc "DD/MM"
- Nút **View Investigation** per row → navigate `/videos/:video_id`
- "Load more investigations" button ở cuối (UI only, load thêm 5 videos)
- Data: 5 videos gần nhất có status PENDING_CONFIRM hoặc COMPLETED, kèm segment đầu tiên

---

**Filter** (áp dụng cho Recent Alerts + Top Detections + Recent Investigations):
- Dropdown: Anomaly Class (All + 15 options)
- Date range picker: theo ngày tạo video
- Nút Apply / Reset
- Filter button trong Recent Investigations header trigger panel filter này

---

**Export Data**: UI only → toast "Coming soon"
**View All** (Recent Alerts): UI only → toast "Coming soon"

---

**API Endpoints cho Dashboard**:

| Endpoint | Response |
|---|---|
| `GET /api/dashboard/stats` | `{total_videos, total_anomalies, pending_reviews, reviewed_cases}` |
| `GET /api/dashboard/distribution` | `[{class, count, percentage}]` |
| `GET /api/dashboard/recent-alerts?limit=10&class=&date_from=&date_to=` | segments gần nhất + severity |
| `GET /api/dashboard/top-detections` | `[{class, count}]` sort DESC limit 6 |
| `GET /api/dashboard/recent-investigations?limit=5` | videos gần nhất + segment đầu tiên |
| `GET /api/dashboard/recent-activity?limit=5` | từ activity_log |

**Acceptance Criteria**:
- AC39: 4 summary cards hiển thị đúng số liệu từ DB
- AC40: Donut chart đúng phân phối theo class
- AC41: Recent Alerts 10 rows, Severity tính đúng từ anomaly_score
- AC42: Top Detections bar chart đúng data
- AC43: Recent Investigations 5 rows + "Load more" +5
- AC44: Filter (class + date) áp dụng cho Alerts + Top Detections + Investigations
- AC45: Row click → navigate đúng `/videos/:id`

---

### FR08 — Export Report

**Trigger**: Nút "Export Report" trên Video Detail Page (khả dụng ở mọi status trừ WAITING).

**Output**: Download file `report_{video_id}.json`

```json
{
  "video": {
    "id": "...", "name": "...", "location": "...",
    "duration": 312.5, "status": "COMPLETED", "created_at": "..."
  },
  "summary": {
    "total_segments": 5,
    "total_anomalies": 4,
    "feedback_submitted": 3,
    "pending_review": 2
  },
  "segments": [
    {
      "segment_id": "SEG-0001",
      "time_range": "02:14 - 02:25",
      "predicted_class": "Fighting",
      "confidence_score": 0.94,
      "anomaly_score": 0.87,
      "review_status": "LOGGED",
      "is_correct": true,
      "verified_label": "Fighting",
      "other_description": null,
      "investigator_comment": "Two individuals near column B3",
      "feedback_submitted_at": "2025-01-15T10:45:00"
    }
  ]
}
```

**Acceptance Criteria**:
- AC43: File download khi click Export Report
- AC44: File chứa đúng data video + tất cả segments + feedback hiện tại

---

## Navigation (5 items)

| Nav Item | Route | Ghi chú |
|---|---|---|
| Dashboard | `/dashboard` | |
| Upload Video | `/` | Trang chính |
| Queue Analyze | `/queue` | Xem batch queue |
| Alerts | `/alerts` | Placeholder: "Coming Soon" |
| Profile | `/profile` | Stats + Activity + Settings (FR09) |

---

## Non-Functional Requirements

- App chạy được trên local machine / lab server
- Không cần HTTPS, không cần authentication
- Response time API < 2s (trừ upload và AI processing)
- Lỗi phải có thông báo rõ ràng trên UI, không để màn hình trắng

---

### FR09 — Profile Page

**Route**: `/profile`  
**Trigger**: User click "Profile" trên nav.

> ⚠️ Đây là trang có chức năng thật — không phải placeholder.

**Layout**: 2 cột (LEFT ~65% / RIGHT ~35%), gap 32px.

---

**Header Card** (full width, bg white, border-radius 16px):
- Avatar: 128×128px, border-radius 16px, border `2px #2563EB`
- Camera icon badge góc dưới phải avatar: bg `#004AC6`, border-radius 12px
- Tên: "Officer James Miller" — hard-code cho demo
- Role: "Senior Security Investigator" — hard-code
- 3 badges: ID (`#SOC-882`), Location (`Sector A, London`), Current Shift (`14h Current Shift`)
- Nút **Edit Profile** (primary, bg `#004AC6`)
- Nút **Export Activity** (outline, bg white, border `#737686`)

> Edit Profile và Export Activity: **UI only** cho demo — không cần implement chức năng thật. Click thì hiện toast "Coming soon".

---

**CỘT TRÁI**:

**Stats Cards** (3 cards ngang, equal width):
| Card | Label | Value | Sub-text |
|---|---|---|---|
| 1 | VIDEOS UPLOADED | Đếm từ DB | "+12% vs LW" (hard-code) |
| 2 | CASES REVIEWED | Đếm segment đã feedback | "84% Completion" (hard-code) |
| 3 | FEEDBACK SUBMITTED | Đếm feedback có submitted_at | "98% Avg Score" (hard-code) |

- Value VIDEOS UPLOADED: `COUNT(*) FROM videos`
- Value CASES REVIEWED: `COUNT(*) FROM anomaly_segments WHERE feedback_submitted_at IS NOT NULL`
- Value FEEDBACK SUBMITTED: giống CASES REVIEWED (hoặc tính riêng)
- Sub-text và % là hard-code (demo)

**Recent Activity Card** (full width cột trái):
- Icon `~` + title "Recent Activity"
- Danh sách 10 activity gần nhất từ bảng `activity_log`
- Mỗi activity: icon theo type + title + description + timestamp relative ("2h ago", "Yesterday")
- Activity types (và icon màu tương ứng):
  - `UPLOAD`: icon upload, bg `#004AC6`
  - `REVIEW_COMPLETE`: icon clipboard, bg `#737686`
  - `FLAG`: icon alert-triangle, bg `#F59E0B` (vàng cam)
- Click activity có `video_id` → navigate `/videos/:video_id`
- Nút "View Full Activity History" ở cuối (UI only, toast "Coming soon")

---

**CỘT PHẢI**:

**Account Settings Card**:
- Title: "Account Settings" + icon users
- Row 1: Email Address → `j.miller@ssis.hq.com` (hard-code) + chevron
- Row 2: Language & Region → `English (UK) • UTC +0` (hard-code) + chevron
- Click rows: toast "Coming soon"

**Security Card**:
- Title: "Security" + icon shield
- Row 1: "Change Password" + "90 days ago" badge + chevron
- Row 2: "2FA Verification" + badge "RECOMMENDED" + toggle switch (UI only, không lưu state)
- Click: toast "Coming soon"

**Notifications Card**:
- Title: "Notifications" + icon bell
- 3 toggle rows:
  - "Critical Alerts" + "Immediate push to mobile & email" → toggle ON (default)
  - "Case Updates" + "Daily digest of reviewed materials" → toggle OFF (default)
  - "Login History" + "Notify on new device sign-in" → toggle ON (default)
- Toggle state: **lưu vào localStorage** (không cần API — demo)

**Log Out from Session** (button cuối, text đỏ `#BA1A1A`, outline đỏ):
- Click: toast "Coming soon" (không có auth nên không cần thật)

---

**API Endpoints cần thêm**:

| Endpoint | Mô tả |
|---|---|
| `GET /api/profile/stats` | Trả về videos_uploaded, cases_reviewed, feedback_submitted |
| `GET /api/profile/activity` | Trả về 10 activity gần nhất từ `activity_log` |

---

**Activity Log — khi nào ghi**:
- Upload video thành công → ghi 1 activity type=`UPLOAD`
- Submit Feedback lần đầu cho video → ghi 1 activity type=`REVIEW_COMPLETE`
- Video có segment với anomaly_score > 0.8 → ghi 1 activity type=`FLAG` (khi worker xong)

**Acceptance Criteria**:
- AC45: Stats cards hiển thị đúng số liệu từ DB
- AC46: Recent Activity hiển thị đúng 10 activities gần nhất
- AC47: Click activity có video_id → navigate đúng trang
- AC48: Notification toggles lưu state vào localStorage
- AC49: Edit Profile / Export Activity / Settings rows → toast "Coming soon"
