# PROGRESS.md
# Build Progress — Human Anomaly Detection System (v2)

> AI agent: Đọc file này khi bắt đầu session mới để biết đang ở đâu.
> Cập nhật file này sau khi hoàn thành mỗi task.

---

## Current Status

**Đang ở task**: Chưa bắt đầu  
**Vấn đề hiện tại**: Không có  
**Ghi chú**: Labels đã chốt — xem REQUIREMENTS.md phần Anomaly Labels

---

## Task List

### Phase 1 — Setup

- [x] **Task 1**: Khởi tạo project structure
  - Backend: `pip install fastapi uvicorn sqlalchemy python-multipart`
  - Frontend: `npm create vite@latest frontend -- --template react-ts`
  - Frontend packages: `axios tailwindcss recharts react-router-dom`
  - Tạo đúng thư mục theo AGENTS.md
  - Tạo `uploads/` folder, `.env.example`

- [x] **Task 2**: Setup database
  - Tạo `models.py`: Video, Batch, ProcessingJob, AnomalySegment (đúng theo DATABASE_SCHEMA.md)
  - Tạo `database.py`: engine + `get_db()`
  - Tạo `schemas.py`: Pydantic schemas
  - Tạo `utils.py`: `generate_video_id()`, `generate_batch_id()`, `format_time()`
  - Chạy `create_all()`, verify bảng

---

### Phase 2 — Backend Core

- [x] **Task 3**: Upload Batch API
  - `POST /api/videos/upload` — nhận nhiều file (multipart)
  - Validate batch rules: max 3 video, max 300 MB total, max 5 phút / video
  - Tạo Batch record
  - Lưu từng file, gen Video ID, tạo Video record (status=WAITING, progress_step=WAITING)
  - Tạo ProcessingJob cho từng video (status=PENDING)
  - Trigger worker loop (BackgroundTask)

- [x] **Task 4**: Worker với concurrency = 1
  - `worker.py`: `start_worker_loop()` poll job PENDING theo FIFO
  - Update status + progress_step đúng thứ tự:
    - WAITING → PROCESSING, progress_step = PHASE1_START
    - Sau Phase 1: progress_step = PHASE1_DONE
    - Sau Phase 2: progress_step = PHASE2_DONE
    - Xong: status = PENDING_CONFIRM, progress_step = PENDING_CONFIRM
    - Exception: status = FAILED + error_message
  - Worker poll job tiếp theo sau PENDING_CONFIRM hoặc FAILED (không chờ COMPLETED)
  - Stub Phase 1 và Phase 2 (mock data)
  - Handle exception → FAILED + error_message
  - Test: upload 3 video, verify chỉ 1 video PROCESSING tại một thời điểm

- [x] **Task 5**: Video & Batch APIs
  - `GET /api/videos` — danh sách
  - `GET /api/videos/:id` — chi tiết + segments
  - `GET /api/batches/:id` — batch info + tất cả videos trong batch
  - `POST /api/videos/:id/retry` — tạo lại job cho video FAILED
  - Static: `GET /uploads/{filename}`

- [x] **Task 6**: Feedback + Export + Dashboard APIs
  - `POST /api/segments/:id/feedback` — body: `{is_correct, verified_label, other_description?, investigator_comment?}`
  - Tự tính `review_status` khi lưu (logic ưu tiên: LOGGED > CORRECTED > LABEL_CORRECT)
  - Sau mỗi feedback: kiểm tra nếu TẤT CẢ segments của video đã có `feedback_submitted_at` → update `video.status = COMPLETED`
  - `GET /api/videos/:id/export` — trả về JSON report (download)
  - `GET /api/dashboard/stats`
  - `GET /api/dashboard/alerts`

---

### Phase 3 — Frontend

- [x] **Task 7**: Setup + Shared components + Routing
  - Cấu hình Tailwind, React Router
  - `api/api.ts` với axios instance
  - `types/types.ts` với `ANOMALY_LABELS` constant và tất cả interfaces
  - Sidebar Nav (5 items)
  - Toast component, StatusBadge, LoadingSpinner, ProgressBar
  - Routes: `/`, `/queue`, `/videos/:id`, `/dashboard`, `/alerts`, `/profile`

- [x] **Task 8**: Upload Page (`/`)
  - Drag & Drop zone
  - Queue table: filename, size, duration, status badge (Ready/Invalid Format), xóa
  - Validate realtime (format, max 3, max 300MB, max 5 phút)
  - Progress bar mỗi file khi uploading
  - Redirect sang `/queue` sau khi upload xong

- [x] **Task 9**: Queue Analyze Page (`/queue`)
  - Active Batch panel: completion %, progress bar, "X of Y processed"
  - 4 summary cards: DONE/ACTIVE/QUEUED/ERRORS
  - Queue Details table với thumbnail, status badge, progress bar, action buttons
  - Retry cho FAILED video
  - Pagination 10 rows
  - Auto-polling mỗi 5 giây

- [x] **Task 10**: Video Detail Page — Cột trái (`/videos/:id`)
  - Video player HTML5 + overlay badge ANOMALY DETECTED
  - Processing progress bar (đọc progress_step → map sang %) khi status = PROCESSING
  - Analysis Timeline và Segments Table chỉ hiện khi status = PENDING_CONFIRM hoặc COMPLETED
  - Detected Segments table (click row → cập nhật panel phải)
  - Polling 3 giây khi đang PROCESSING (dừng khi PENDING_CONFIRM hoặc FAILED)

- [x] **Task 11**: Video Detail Page — Cột phải
  - Investigation Summary panel (cập nhật khi click row)
  - Feedback panel State A (Form): 2 câu hỏi + dropdown + Other textarea + comments
  - Feedback panel State B (Feedback Detail): hiển thị data đã submit + nút Edit Feedback
  - Edit Feedback → pre-fill form → Save Changes / Cancel
  - Submit Feedback → optimistic update Review Status trong table
  - Nút Export Report (download JSON)

- [x] **Task 12**: Dashboard Page (`/dashboard`)
  - Đọc DASHBOARD_UI_SPEC.md trước khi code
  - Welcome banner + nút Upload New Video → navigate `/`
  - 4 summary cards: data từ `GET /api/dashboard/stats`
  - Anomaly Distribution: Recharts PieChart donut, data từ `GET /api/dashboard/distribution`
  - Recent Activity (5 items): data từ `GET /api/dashboard/recent-activity`
  - Recent Alerts table (10 rows): data từ `GET /api/dashboard/recent-alerts`, Severity từ anomaly_score
  - Top Detections: Recharts BarChart horizontal, data từ `GET /api/dashboard/top-detections`
  - Recent Investigations table (5 rows + Load more): data từ `GET /api/dashboard/recent-investigations`
  - Filter panel (class + date): áp dụng cho Alerts + Top Detections + Investigations
  - Row click → navigate `/videos/:id`
  - Backend: thêm 5 API endpoints mới vào `main.py` (stats, distribution, recent-alerts, top-detections, recent-investigations, recent-activity)

---

- [x] **Task 16**: Profile Backend APIs
  - Tạo bảng `activity_log` (thêm vào `models.py` và `create_all()`)
  - Ghi activity khi: upload thành công, worker xong PENDING_CONFIRM (nếu có FLAG), user submit feedback lần đầu
  - `GET /api/profile/stats` — trả về videos_uploaded, cases_reviewed, feedback_submitted (đếm từ DB)
  - `GET /api/profile/activity` — trả về 10 activity gần nhất

- [x] **Task 17**: Profile Page (`/profile`)
  - Đọc PROFILE_UI_SPEC.md trước khi code
  - Header card: avatar, tên, role, 3 badges, Edit Profile + Export Activity (toast "Coming soon")
  - Stats cards (3): data từ `GET /api/profile/stats`
  - Recent Activity: data từ `GET /api/profile/activity`, click → navigate `/videos/:id`
  - Account Settings card: hard-code data, click rows → toast "Coming soon"
  - Security card: hard-code, toggle 2FA UI only
  - Notifications card: 3 toggles, lưu state vào localStorage
  - Log Out button: toast "Coming soon"

---

- [x] **Task 18**: Alerts Backend APIs
  - `GET /api/alerts/stats` — 4 summary cards
  - `GET /api/alerts/log?name=&activity=&severity=&status=&date=&page=1&limit=10` — có filter + pagination
  - `GET /api/alerts/distribution` — top 5 class + count + percentage
  - `GET /api/alerts/critical?limit=10` — segments có anomaly_score ≥ 0.85
  - Tạo `backend/routers/alerts.py`, include vào `main.py`
  - Severity tính ở backend từ anomaly_score (≥0.85=HIGH, 0.65-0.84=MEDIUM, <0.65=LOW)

- [x] **Task 19**: Alerts Page (`/alerts`) + Video Detail query param
  - Đọc ALERTS_UI_SPEC.md trước khi code
  - 4 summary cards: data từ `GET /api/alerts/stats`
  - Filter bar: search name + 3 dropdowns + date picker + Reset
  - Alert Log table: data từ `GET /api/alerts/log`, pagination 10 rows
  - Alert Distribution card: bar chart CSS thuần (không Recharts)
  - Speed Up Analysis card: static, hard-code
  - Recent Critical Alerts table: data từ `GET /api/alerts/critical`
  - Click "View Detail" / "View Investigation" → navigate `/videos/:id?segment=:seg_id`
  - Cập nhật VideoDetail.tsx: đọc `?segment=` query param → auto-select segment + seek video

- [x] **Task 20**: Notification Backend
  - Thêm `Notification` model vào `models.py` (đúng theo DATABASE_SCHEMA.md)
  - Chạy `create_all()` tạo bảng `notifications`
  - Tạo helper `create_notification(db, type, title, message, target_url, video_id)` trong `utils.py`
  - Gọi helper trong `worker.py` đúng 5 trigger points (PENDING_CONFIRM có/không segment, FAILED, batch complete, low confidence)
  - Tạo `backend/routers/notifications.py` với 4 endpoints
  - Include router vào `main.py`
  - Thêm Pydantic schemas vào `schemas.py`

- [x] **Task 21**: Notification Frontend
  - Đọc `NOTIFICATION_UI_SPEC.md` trước khi code
  - Tạo `NotificationContext` ở App level (shared state)
  - Polling `GET /api/notifications?is_read=0` mỗi 10 giây
  - Build `NotificationStack` component (góc dưới trái, hover expand/collapse)
  - Build `NotificationBell` component (header, dropdown giống Facebook)
  - Cập nhật `Header` component: tích hợp bell + badge
  - Cập nhật `App.tsx`: mount `NotificationStack` + `NotificationProvider`
  - Cập nhật `VideoDetail.tsx`: không ảnh hưởng gì (navigate vẫn như cũ)
  - Thêm API functions vào `api.ts`
  - Thêm TypeScript interfaces vào `types.ts`


- [x] **Task 22**: Auth Backend
  - `pip install passlib[bcrypt] python-jose[cryptography]`
  - Thêm `User` model vào `models.py` (đúng theo DATABASE_SCHEMA.md)
  - Chạy `create_all()` tạo bảng `users`
  - Tạo `backend/auth.py`: hash_password, verify_password, create_access_token, get_current_user
  - Tạo `backend/routers/auth.py`: POST /register, POST /login, GET /me
  - Thêm schemas: RegisterRequest, LoginRequest, AuthResponse, UserResponse
  - Include router vào main.py
  - **CHƯA** thêm `get_current_user` vào các routes khác — để Task 23 làm riêng

- [x] **Task 23**: Protect existing API routes
  - Thêm `current_user: User = Depends(get_current_user)` vào TẤT CẢ routes hiện có
    (videos, batches, segments, dashboard, alerts, profile, notifications)
  - KHÔNG thêm vào: `/api/auth/*`, static file serving `/uploads/*`
  - Test: gọi 1 API không có token → verify 401
  - Test: gọi với token hợp lệ → verify vẫn hoạt động như cũ

- [x] **Task 24**: Auth Frontend
  - Đọc `AUTH_UI_SPEC.md` trước khi code
  - `context/AuthContext.tsx`: user, token, login(), register(), logout()
  - `components/ProtectedRoute.tsx`
  - `pages/Login.tsx`, `pages/Register.tsx` — layout riêng, không Sidebar/Header
  - Axios interceptors: request thêm Bearer token, response 401 → logout + redirect
  - Cập nhật `App.tsx`: wrap routes với ProtectedRoute
  - Cập nhật Sidebar "Logout" button: gọi logout()
  - Cập nhật Profile "Log Out from Session": gọi logout() (không còn toast)
  - Header: hiển thị username từ AuthContext
  - `npm.cmd run build` pass, không có TypeScript errors


### Phase 4 — Integration & Polish

- [x] **Task 13**: Integrate AI pipeline thật
  - Thay stub trong worker.py bằng import pipeline PyTorch thật
  - Test end-to-end với video thật

- [x] **Task 14**: Seed data
  - `seed.py`: insert 2–3 video COMPLETED với segments và feedback mẫu
  - `python seed.py` cho ra data demo ngay

- [ ] **Task 15**: Final checklist
  - [ ] App chạy: `uvicorn backend.main:app` + `npm run dev`
  - [ ] `npm run build` không có TypeScript errors
  - [ ] Upload 3 video → chỉ 1 video PROCESSING tại một thời điểm, 2 video còn lại WAITING
  - [ ] Worker xong → video status = PENDING_CONFIRM (không phải COMPLETED)
  - [ ] Queue page: status badge đúng (WAITING/PROCESSING/PENDING_CONFIRM/COMPLETED/FAILED), progress bar đúng %
  - [ ] Queue page: nút Review hiển thị cho video PENDING_CONFIRM
  - [ ] Retry video FAILED → job mới được tạo, video về WAITING
  - [ ] Video Detail: player, overlay badge, progress bar (khi PROCESSING), timeline, segments table hoạt động
  - [ ] Timeline và Segments Table chỉ hiện khi PENDING_CONFIRM hoặc COMPLETED
  - [ ] Click row → cập nhật Investigation Summary + Feedback panel
  - [ ] Submit Feedback → panel chuyển sang Feedback Detail ngay
  - [ ] Edit Feedback → Cancel → giữ data cũ; Save → ghi đè, hiển thị data mới
  - [ ] Submit feedback Other → bắt buộc nhập description
  - [ ] Sau khi TẤT CẢ segments feedback → video.status = COMPLETED
  - [ ] Export Report → download JSON đúng format
  - [ ] Dashboard hiển thị đúng số liệu + filter hoạt động
  - [ ] README có hướng dẫn setup đầy đủ
  - [ ] Ghi chú TODO trong PROGRESS.md về "confirm lại 13 nhãn" đã xóa (đã chốt)

---

## Decisions Log

| Quyết định | Lý do |
|---|---|
| SQLite | Demo đơn giản, không cần server DB |
| BackgroundTasks + worker loop | Đủ cho demo, không cần Celery/Redis |
| Concurrency = 1 | GPU chỉ xử lý 1 video tại một thời điểm |
| Worker poll FIFO | Đơn giản, fair queue |
| Polling thay WebSocket | Dễ implement, đủ UX cho demo |
| JSON export thay PDF | Đơn giản, không cần thư viện ngoài |
| HTML5 video player native | Không cần thư viện ngoài |
| Stub Phase 1 + 2 | Build UI trước khi integrate model thật |
| 15 labels (13 anomaly + Normal + Other) | Theo yêu cầu, Other có text input |
