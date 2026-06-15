# AGENTS.md
# AI Agent Rules — Human Anomaly Detection System

> File này dành cho AI agent (Claude Code, Cursor, Copilot...).
> Đọc file này TRƯỚC KHI bắt đầu bất kỳ task nào.
> Tuân thủ toàn bộ — không tự ý điều chỉnh.

---

## Nguyên tắc tối thượng

1. **Chỉ implement những gì được yêu cầu trong REQUIREMENTS.md**. Không tự thêm tính năng.
2. **Không thay đổi DATABASE_SCHEMA.md**. Schema đã chốt. Nếu cần thêm cột, hỏi lại trước.
3. **Không thay đổi tech stack**. Stack đã chốt trong REQUIREMENTS.md.
4. **Ưu tiên chạy được và đơn giản** hơn là elegant hay clever.
5. **Mỗi task làm một việc**. Không làm nhiều task trong một lần.

---

## Project Structure (bắt buộc tuân theo)

```
project/
├── backend/
│   ├── main.py              # FastAPI app, routes, CORS, static files
│   ├── database.py          # SQLAlchemy setup, get_db()
│   ├── models.py            # SQLAlchemy ORM models (Video, Batch, ProcessingJob, AnomalySegment, User, ...)
│   ├── schemas.py           # Pydantic schemas (request/response)
│   ├── worker.py            # AI pipeline worker, poll job queue, ghi vào SQLite
│   ├── utils.py             # generate_video_id(), generate_batch_id(), format_time()
│   ├── auth.py               # JWT encode/decode, password hash, get_current_user dependency
│   ├── routers/
│   │   └── auth.py           # POST /api/auth/register, /login, GET /api/auth/me
│   └── uploads/             # Thư mục lưu video
├── frontend/
│   ├── src/
│   │   ├── pages/           # Upload.tsx, Queue.tsx, VideoDetail.tsx, Dashboard.tsx,
│   │   │                    # Alerts.tsx, Profile.tsx, Login.tsx, Register.tsx
│   │   ├── components/      # StatusBadge, Toast, Sidebar, Timeline,
│   │   │                    # SegmentsTable, InvestigationPanel, FeedbackPanel,
│   │   │                    # ProtectedRoute.tsx
│   │   ├── context/         # NotificationContext.tsx, AuthContext.tsx
│   │   ├── api/             # api.ts — tất cả axios/fetch calls (kèm interceptor JWT)
│   │   └── types/           # types.ts — TypeScript interfaces
│   └── ...
├── REQUIREMENTS.md
├── DATABASE_SCHEMA.md
├── UI_FLOW.md
├── AGENTS.md
├── PROGRESS.md
└── .env.example
```

---

## Backend Rules

### FastAPI
- Dùng `APIRouter` để nhóm routes, không nhét tất cả vào `main.py`
- Tất cả routes có prefix `/api`
- CORS: allow all origins (demo, không có production)
- Static files: mount `uploads/` tại `/uploads`
- Không dùng async SQLAlchemy — dùng sync thông thường với `get_db()` dependency

### Authentication (JWT)
- File `backend/auth.py` chứa:
  - `hash_password(password: str) -> str` (dùng `passlib` bcrypt)
  - `verify_password(password: str, hashed: str) -> bool`
  - `create_access_token(data: dict) -> str` (dùng `python-jose`, exp = 7 ngày)
  - `get_current_user(token: str = Depends(oauth2_scheme), db = Depends(get_db)) -> User` — dependency dùng cho mọi protected route
- `JWT_SECRET_KEY` đọc từ `.env`, KHÔNG hardcode trong code
- Tất cả routes trong `routers/` (trừ `routers/auth.py`) thêm dependency:
  ```python
  @router.get("/...")
  def some_route(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
      ...
  ```
- `routers/auth.py`: `/register`, `/login` KHÔNG cần `get_current_user`. `/me` cần.
- Static file serving (`/uploads/...`) KHÔNG cần auth (video player cần load trực tiếp)
- 401 response khi token invalid/expired — FastAPI tự raise qua `HTTPException(401)`

### Database
- Dùng SQLAlchemy với SQLite, file `backend/anomaly.db`
- Tên bảng và cột phải khớp chính xác với DATABASE_SCHEMA.md
- Dùng `datetime.utcnow().isoformat()` cho tất cả timestamp, lưu dạng TEXT

### Worker
- File `worker.py` chứa function `run_pipeline(video_id: str, db_path: str)` và `start_worker_loop(db_path: str)`
- Worker nhận `db_path` để tạo connection riêng (không share session với FastAPI)
- Worker poll job queue: `SELECT * FROM processing_jobs WHERE status='PENDING' ORDER BY created_at ASC LIMIT 1`
- **Concurrency = 1**: không bao giờ xử lý 2 video cùng lúc
- Worker loop: sau khi 1 job xong (COMPLETED hoặc FAILED) → poll job tiếp theo ngay
- Worker update `videos.status` và `videos.progress_step` theo đúng thứ tự:
  - WAITING → PROCESSING, progress_step = PHASE1_START
  - Sau Phase 1: progress_step = PHASE1_DONE
  - Sau Phase 2: progress_step = PHASE2_DONE
  - Xong: status = PENDING_CONFIRM, progress_step = PENDING_CONFIRM
  - Exception: status = FAILED, lưu error_message
- **Worker loop**: sau khi job đạt `PENDING_CONFIRM` hoặc `FAILED` → poll job tiếp theo. KHÔNG chờ COMPLETED (COMPLETED do user trigger qua Submit Feedback)
- Video mới thêm vào queue: status = WAITING nếu đã có video đang PROCESSING
- Wrap toàn bộ logic trong try/except, bắt lỗi → update status = FAILED + error_message
- AI pipeline được import từ module ngoài — nếu chưa có, tạo stub function trả về mock data

### Video ID Generation
- Format: `{YYYYMMDD_HHMMSS}_{xxxx}`
- `xxxx` là số thứ tự 4 chữ số trong cùng giây, bắt đầu từ 0001
- Implement trong `utils.py`

### File Upload
- Lưu file với tên = `{video_id}.{ext}` vào thư mục `uploads/`
- Validate extension: chỉ chấp nhận `.mp4`, `.avi`, `.mov`
- Không cần validate MIME type — extension check là đủ cho demo

### API Response Format
Tất cả response dùng format:
```json
{
  "success": true,
  "data": { ... },
  "message": "..."
}
```
Lỗi:
```json
{
  "success": false,
  "data": null,
  "message": "Mô tả lỗi"
}
```

---

## Frontend Rules

### TypeScript
- Không dùng `any` — khai báo interface cho tất cả data từ API trong `types/types.ts`
- Không bỏ qua TypeScript errors — fix trước khi báo task xong

### API Calls
- Tất cả API calls viết trong `api/api.ts`, export functions
- Dùng `axios` (không dùng fetch raw)
- Base URL đọc từ env: `VITE_API_URL` (default: `http://localhost:8000`)
- Không hardcode URL trong component
- **Axios interceptor** (request): tự động thêm header `Authorization: Bearer {token}` từ `localStorage.getItem('token')` vào mọi request
- **Axios interceptor** (response): nếu response status = 401 → xóa token khỏi `localStorage`, redirect `/login`

### Authentication (Frontend)
- `context/AuthContext.tsx`: Provider quản lý `user`, `token`, `login()`, `register()`, `logout()`
- `components/ProtectedRoute.tsx`: wrapper component, kiểm tra `localStorage.getItem('token')`
  - Không có token → `<Navigate to="/login" replace />`
  - Có token → render children
- Routing trong `App.tsx`:
  ```tsx
  <Route path="/login" element={<Login />} />
  <Route path="/register" element={<Register />} />
  <Route path="/" element={<ProtectedRoute><Upload /></ProtectedRoute>} />
  // ... tất cả routes khác wrap tương tự
  ```
- `/login` và `/register` KHÔNG có Sidebar/Header — layout riêng, centered form
- Logout: gọi `logout()` từ AuthContext → xóa token → `navigate('/login', {replace: true})`

### State Management
- Dùng React built-in hooks (useState, useEffect) — không cần Redux hay Zustand
- Polling (Video Detail): dùng `setInterval` trong `useEffect`, cleanup khi unmount hoặc status terminal

### Component Rules
- Mỗi page là một file riêng trong `pages/`
- Component tái sử dụng đặt trong `components/`
- Không viết CSS inline dài — dùng Tailwind classes
- Không tự thêm UI library ngoài những gì đã có (Recharts cho chart là đủ)

### Timeline Component
- Dùng `<div>` với CSS positioning — không cần thư viện ngoài
- Width của highlight segment = `(end_time - start_time) / duration * 100%`
- Left offset = `start_time / duration * 100%`

### Formatting
- Time display: `MM:SS` — viết helper `formatTime(seconds: number): string`
- Confidence: `(score * 100).toFixed(1) + '%'`
- Score: `score.toFixed(2)`

---

## Những điều AI KHÔNG được làm

- ❌ Không thêm bảng mới vào database khi chưa có trong SCHEMA
- ❌ Không cài thêm package nếu chưa thảo luận
- ❌ Không thêm màn hình mới ngoài các màn hình trong UI_FLOW.md
- ❌ Không implement tính năng nằm trong Out of Scope của REQUIREMENTS.md
- ❌ Không refactor code của task khác trong khi làm task hiện tại
- ❌ Không xóa comment hiện có
- ❌ Không tự thêm labels ngoài 15 labels đã liệt kê trong REQUIREMENTS.md
- ❌ Không implement xử lý song song nhiều video (concurrency phải = 1)
- ❌ Không tự generate seed data phức tạp — chỉ tạo nếu task yêu cầu
- ❌ Không thêm phân quyền theo role hoặc OAuth — chỉ Register/Login JWT đơn giản theo FR12
- ❌ Không thêm `user_id` vào các bảng videos/segments/... — mọi user thấy chung data

## Anomaly Labels (15 options — không thay đổi)

```typescript
// types/types.ts — dùng constant này ở mọi nơi, không tự ý thêm bớt
export const ANOMALY_LABELS = [
  "Abuse", "Arrest", "Arson", "Assault", "Burglary",
  "Explosion", "Fighting", "RoadAccidents", "Robbery",
  "Shooting", "Shoplifting", "Stealing", "Vandalism",
  "Normal", "Other"
] as const;

export type AnomalyLabel = typeof ANOMALY_LABELS[number];
```

---

## Definition of Done (mỗi task)

Task chỉ được coi là DONE khi:
- [ ] Code chạy không có lỗi (backend start được, frontend build được)
- [ ] Không có TypeScript errors
- [ ] Feature hoạt động đúng theo Acceptance Criteria trong REQUIREMENTS.md
- [ ] PROGRESS.md được update (task đó tick Done)
- [ ] Không có TODO comment còn lại trong code của task đó

---

## Cách handle AI pipeline stub

Vì AI pipeline (PyTorch) là module bên ngoài, khi chưa integrate:

```python
# worker.py — stub khi chưa có pipeline thật
def run_phase1(video_path: str) -> list[dict]:
    """
    Returns list of segments: [{start_time, end_time, anomaly_score}]
    REPLACE THIS with actual pipeline import when ready.
    """
    return [
        {"start_time": 10.5, "end_time": 25.0, "anomaly_score": 0.87},
        {"start_time": 78.2, "end_time": 91.4, "anomaly_score": 0.73},
    ]

def run_phase2(video_path: str, segment: dict) -> dict:
    """
    Returns: {predicted_class, confidence_score}
    REPLACE THIS with actual pipeline import when ready.
    """
    return {"predicted_class": "Fighting", "confidence_score": 0.91}
```

Khi integrate pipeline thật: chỉ thay thế 2 function này, không đổi gì khác.
