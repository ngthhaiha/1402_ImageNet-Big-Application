# UI_FLOW.md
# Frontend Screen Flow — Human Anomaly Detection System (v2)

## Routes

```
/              → Upload Page
/queue         → Queue Analyze Page
/videos        → Video List Page (optional, có thể merge vào /queue)
/videos/:id    → Video Detail Page
/dashboard     → Dashboard Page
/alerts        → Placeholder (hiện "Coming soon")
/profile       → Placeholder (hiện "Coming soon")
```

**Navigation** (sidebar, luôn hiển thị):
- Dashboard
- Upload Video → `/`
- Queue Analyze → `/queue`
- Alerts → `/alerts`
- Profile → `/profile`
- Nút **+ New Investigation** (primary CTA ở cuối sidebar)

---

## Screen 1: Upload Page (`/`)

**Tham khảo**: Figma ảnh 1.

### Drag & Drop Zone
- Icon upload + text "Drop videos here"
- Sub-text: "Support MP4, AVI, MOV. Maximum 3 videos per batch, total batch size up to 300 MB, maximum 5 minutes per video."
- Nút **+ Browse Files**
- Hỗ trợ drag-and-drop nhiều file cùng lúc

### Queue Table (hiện khi đã chọn ít nhất 1 file)

Header: "Queue (N selected)" — bên phải: "Total: X MB"

Cột:
| Cột | Nội dung |
|---|---|
| File Name | Icon video + tên file |
| Size | MB hoặc GB |
| Duration | HH:MM:SS (nếu đọc được), "--" nếu không |
| Status | Badge: Ready (xanh) / Invalid Format (đỏ) |
| Action | Icon thùng rác (xóa khỏi queue) |

**Validate realtime** khi user thêm file vào queue:
- Format không hỗ trợ → Status = "Invalid Format" (không thể upload file này)
- Quá 3 video → toast "Maximum 3 videos per batch"
- Tổng > 300 MB → toast "Total batch size exceeds 300 MB"
- Video > 5 phút → Status = "Invalid Format" (kèm tooltip "Exceeds 5 min limit")

### Footer Bar
- Info text: "Videos will be processed sequentially in the order uploaded. Estimated processing time: X minutes."
- Nút **Cancel** + Nút **Upload & Analyze** (disabled nếu không có file Ready)

### Upload Progress State (sau khi nhấn Upload & Analyze)
- Mỗi file: thay Status badge bằng progress bar (%)
- Sau khi tất cả upload xong → redirect tự động đến `/queue`

---

## Screen 2: Queue Analyze Page (`/queue`)

**Tham khảo**: Figma ảnh 2.

### Active Batch Panel
- Badge "ACTIVE BATCH" + Batch ID
- Tên batch
- "Uploaded: {date} • {time}"
- Completion % (số to, prominant)
- "X of Y Videos Processed"
- Progress bar tổng

### 4 Summary Cards (ngang)
| Card | Badge | Số |
|---|---|---|
| DONE | xanh | Số COMPLETED |
| ACTIVE | xanh nhạt | Số PROCESSING |
| QUEUED | xám | Số WAITING |
| ERRORS | đỏ | Số FAILED |

### Queue Details Table

Header: "Queue Details" — Actions: "Download Report" | "Cancel All"

Cột:
| Cột | Nội dung |
|---|---|
| Video Name | Thumbnail + tên file |
| Status | Badge màu |
| Progress | Progress bar (% khi PROCESSING, full khi COMPLETED, empty khi WAITING, màu đỏ khi FAILED kèm error text) |
| Duration | HH:MM:SS |
| Submitted | HH:MM:SS |
| Action | Nút tùy status |

**Status Badge Colors**:
| Status | Màu | Label hiển thị |
|---|---|---|
| `WAITING` | Gray | Waiting |
| `PROCESSING` | Yellow/Blue | Processing... |
| `PENDING_CONFIRM` | Orange | Pending Review |
| `COMPLETED` | Green | Completed |
| `FAILED` | Red | Failed |

**Action buttons theo status**:
- `WAITING` → "—" (disabled)
- `PROCESSING` → nút **Monitor** → `/videos/:id`
- `PENDING_CONFIRM` → nút **Review** (highlighted, CTA chính) → `/videos/:id`
- `COMPLETED` → nút **View Detail** → `/videos/:id`
- `FAILED` → nút **Retry**

**Pagination**: 10 rows / page, hiện "Showing X to Y of Z results"

**Auto-refresh**: polling mỗi 5 giây (dừng khi tất cả COMPLETED/FAILED)

---

## Screen 3: Video Detail Page (`/videos/:id`)

**Tham khảo**: Figma ảnh 3.

### Header
- Breadcrumb: "Analysis Queue / Segment Review"
- Tiêu đề: "Video Investigation"
- Nút **Archive Case** (outline) + Nút **Export Report** (primary)

### Layout: 2 cột

---

**CỘT TRÁI** (~ 65% width):

#### Video Player
- HTML5 video player, source từ `/uploads/{video_id}`
- Overlay badge (góc trên trái): `● ANOMALY DETECTED: {CLASS} ({CONFIDENCE}%)`
  - Chỉ hiện khi currentTime nằm trong range [start_time, end_time] của 1 segment
  - Màu đỏ

#### Analysis Timeline (hiện khi PENDING_CONFIRM hoặc COMPLETED)
- Header: "Analysis Timeline" + legend (Normal = xám, Detected Anomaly = đỏ)
- Thanh ngang, full width
- Nền xám = toàn video
- Block đỏ = anomaly segments (position theo %)
- Timestamp markers đều nhau: 00:00, 01:00, 02:00, ...
- Click vào block đỏ → video seek đến start_time của segment đó

#### Detected Segments Table (hiện khi PENDING_CONFIRM hoặc COMPLETED)
Header: "Detected Segments" + icon filter + icon download

Cột:
| Cột | Nội dung |
|---|---|
| Time Range | `MM:SS - MM:SS` |
| Predicted Activity | Label badge (màu khác nhau theo class) |
| Confidence | `XX%` |
| Review Status | Badge: Pending Review / Label Correct / Corrected / Logged |

- **Click vào row** → cập nhật panel bên phải (Investigation Summary + Feedback) với dữ liệu segment đó
- Row đang chọn được highlight

#### Processing State (khi chưa COMPLETED)
- Stepper progress thay cho Timeline và Table:
  ```
  ✓ Uploaded → ⟳ Feature Extraction → Detection → Classification → Completed
  ```
- Polling mỗi 3 giây

---

**CỘT PHẢI** (~ 35% width):

#### Investigation Summary Panel
```
┌─────────────────────────────────┐
│ Investigation Summary  [VERIFIED AI badge] │
│                                 │
│ Video Name    PRK_NORTH_CAM04.mp4│
│ Segment ID    #SEG-4421          │
│ Predicted     ● Fighting         │
│ Confidence    ████████░░  94%    │
│ Timestamp     02:14 - 02:25      │
│ Status        [Pending Review]   │
└─────────────────────────────────┘
```
- Cập nhật ngay khi user click row trong Segments Table
- Mặc định hiển thị segment đầu tiên

#### Feedback & Validation Panel (ngay dưới Investigation Summary)

Panel có **2 state** tùy thuộc segment đã feedback chưa:

---

**STATE A — Form** (segment chưa feedback, hoặc đang Edit):

```
Feedback & Validation
─────────────────────────────────────────────
Is the detected anomaly segment correct?
  [✓ Correct]   [✗ Incorrect]

Is the predicted activity correct?
  [✓ Label Correct]   [✎ Edit Label]
  └─ (nếu Edit Label: dropdown 15 options)
  └─ (nếu Other: textarea "Describe the activity" — bắt buộc nhập)

INVESTIGATOR COMMENTS
[Describe findings, involved parties...    ]
(optional)

Submit button:
  - Lần đầu submit:  [Submit Feedback]    ← primary, disabled cho đến khi chọn ≥1 option
  - Khi Edit:        [Save Changes]  [Cancel]
```

**Sau khi Submit / Save Changes**:
- Lưu feedback vào DB
- Backend kiểm tra nếu TẤT CẢ segments đã feedback → update video.status = COMPLETED
- Panel chuyển ngay sang STATE B (không reload page)

---

**STATE B — Feedback Detail** (sau submit, hoặc load segment đã có feedback):

```
Feedback Detail
─────────────────────────────────────────────
Segment Detect    ✓ Correct   /   ✗ Incorrect
Verified Label    {verified_label}
                  └─ (nếu Other: "{other_description}")
Investigator      {investigator_comment}
Comments          (hiển thị "—" nếu không có)
Submitted At      {DD/MM/YYYY HH:mm}

[           ✎ Edit Feedback           ]
```

**Edit Feedback**:
- Click → panel quay về STATE A, pre-fill toàn bộ giá trị cũ
- Cancel → giữ nguyên STATE B với data cũ
- Save Changes → ghi đè DB, quay về STATE B với data mới

---

## Screen 4: Dashboard Page (`/dashboard`)

### Row 1 — Summary Cards (4 ngang)
- Total Videos
- Total Anomaly Segments
- Total Feedback Submitted
- (Slot trống hoặc "Accuracy Rate" nếu muốn)

### Row 2 — Filters
- Dropdown: Anomaly Class (All + 15 options)
- Date range picker
- Nút Apply / Reset

### Row 3 — Charts
- Bar Chart (Recharts): số segments theo class
- (Optional) Pie Chart: tỷ lệ %

### Row 4 — Recent Alerts Table
- 10 segment mới nhất
- Cột: Video Name, Class, Confidence, Score, Time
- Click row → `/videos/:video_id`

---

## Components dùng chung

| Component | Dùng ở |
|---|---|
| StatusBadge | Video List, Queue, Video Detail |
| Toast | Toàn app (success/error/info) |
| Sidebar Nav | Toàn app |
| LoadingSpinner | Khi đang call API |
| ProgressBar | Upload page, Queue page |

---

## API Calls từ Frontend

| Page | Action | Endpoint |
|---|---|---|
| Upload | Upload files | `POST /api/videos/upload` (multipart, nhiều file) |
| Queue | Load queue | `GET /api/batches/:id` hoặc `GET /api/videos?batch_id=X` |
| Queue | Retry failed | `POST /api/videos/:id/retry` |
| Queue | Polling | `GET /api/batches/:id/status` (mỗi 5s) |
| Video Detail | Load video | `GET /api/videos/:id` |
| Video Detail | Polling | `GET /api/videos/:id` (mỗi 3s khi đang xử lý) |
| Video Detail | Submit feedback | `POST /api/segments/:id/feedback` |
| Video Detail | Export report | `GET /api/videos/:id/export` (download JSON) |
| Dashboard | Stats | `GET /api/dashboard/stats` |
| Dashboard | Recent alerts | `GET /api/dashboard/alerts` |
