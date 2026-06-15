# NOTIFICATION_UI_SPEC.md
# UI Specification — Notification System (Task 20 + 21)

## Tổng quan kiến trúc

```
App.tsx
├── NotificationProvider (Context + polling)
│   ├── Header
│   │   └── NotificationBell (icon + badge + dropdown)
│   ├── [tất cả pages]
│   └── NotificationStack (fixed, góc dưới trái)
```

Shared state qua Context — bell và stack dùng chung, không duplicate call API.

---

## Design Tokens

```
-- Type colors --
success:  dot/accent #059669  bg #D1FAE5  (xanh lá)
error:    dot/accent #BA1A1A  bg #FEE2E2  (đỏ)
warning:  dot/accent #D97706  bg #FEF3C7  (vàng cam)
info:     dot/accent #004AC6  bg #DBEAFE  (xanh nhạt)

-- Stack card --
Card bg:          white
Card border:      1px rgba(195,198,215,0.60)
Card shadow:      0px 4px 12px rgba(0,0,0,0.10)
Card border-radius: 12px
Card padding:     12px 16px
Card width:       320px
Card hover bg:    rgba(242,243,255,0.80)

-- Stack collapsed --
Card offset Y:    8px per card (card sau thấp hơn 8px)
Card offset X:    4px per card (card sau thu hẹp vào 4px mỗi bên)
Card opacity:     1.0 → 0.85 → 0.70 cho card thứ 1, 2, 3+
Max visible in stack: 3 cards (card 4+ ẩn, chỉ hiện "+N more" badge)

-- Bell dropdown --
Dropdown width:   380px
Dropdown max-height: 480px (scroll nếu nhiều hơn)
Dropdown bg:      white
Dropdown shadow:  0px 8px 24px rgba(0,0,0,0.12)
Dropdown border-radius: 12px
Item unread bg:   rgba(219,234,254,0.30)  (xanh rất nhạt)
Item read bg:     white
Item hover bg:    rgba(242,243,255,0.80)
Item border-bottom: 1px rgba(195,198,215,0.30)

-- Badge --
Badge bg:         #BA1A1A
Badge text:       white
Badge size:       18px × 18px, border-radius 9999
Badge font:       10px weight 700
Badge position:   absolute top-0 right-0, translate(-25%, 25%)

-- Timestamp --
Color: #737686  12px weight 400
```

---

## 1. NotificationContext

File: `frontend/src/context/NotificationContext.tsx`

```typescript
interface Notification {
  id: number
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  target_url: string | null
  video_id: string | null
  is_read: boolean
  created_at: string
}

interface NotificationContextValue {
  notifications: Notification[]        // tất cả (đã + chưa đọc)
  unreadCount: number                   // computed từ notifications
  markRead: (id: number) => void        // mark 1 notification
  markAllRead: () => void               // mark tất cả
  loadMore: () => void                  // load thêm 10
  isLoading: boolean
}
```

**Polling logic**:
```typescript
// Poll mỗi 10 giây
useEffect(() => {
  fetchNotifications()  // load lần đầu
  const interval = setInterval(fetchNotifications, 10000)
  return () => clearInterval(interval)
}, [])

const fetchNotifications = async () => {
  const data = await getNotifications({ limit: 20, offset: 0 })
  setNotifications(data.items)
}
```

**markRead** (optimistic update):
```typescript
const markRead = async (id: number) => {
  // Cập nhật local state ngay
  setNotifications(prev =>
    prev.map(n => n.id === id ? { ...n, is_read: true } : n)
  )
  // Gọi API sau
  await patchNotificationRead(id)
}
```

**markAllRead**:
```typescript
const markAllRead = async () => {
  setNotifications(prev => prev.map(n => ({ ...n, is_read: true })))
  await patchNotificationReadAll()
}
```

---

## 2. NotificationStack

File: `frontend/src/components/notifications/NotificationStack.tsx`

**Vị trí**: `fixed bottom-6 left-6 z-50`

Chỉ render notifications có `is_read = false`, max 5.

### Collapsed state (default)

```
[Card 0 — newest, full opacity]
  [Card 1 — offset 8px down, scale nhỏ hơn]
    [Card 2 — offset 16px down, scale nhỏ hơn nữa]
      [+N more badge nếu có card thứ 4+]
```

CSS cho stack effect:
```tsx
// Card thứ i (0-based, 0 = newest = trên cùng)
<div
  style={{
    position: 'absolute',
    bottom: `${i * 8}px`,        // card sau lùi xuống
    left: `${i * 4}px`,          // thu hẹp vào
    right: `${i * 4}px`,
    opacity: Math.max(0.7, 1 - i * 0.15),
    zIndex: 50 - i,
    transform: `scale(${1 - i * 0.02})`,
    transformOrigin: 'bottom center',
    transition: 'all 0.3s ease',
  }}
>
  <NotificationCard notification={notifications[i]} />
</div>
```

Container height khi collapsed:
```tsx
// Đủ cao để chứa card đầu + offset của các card sau
height: `${CARD_HEIGHT + (Math.min(visibleCount, 3) - 1) * 8}px`
```

**+N more badge** (khi có > 3 unread):
```tsx
<div className="absolute -top-2 -right-2 bg-[#BA1A1A] text-white
                text-[10px] font-bold px-2 py-0.5 rounded-full z-50">
  +{unreadCount - 3} more
</div>
```

### Expanded state (on hover)

```
[Card 0 — newest]
[Card 1]
[Card 2]
[Card 3]
[Card 4]
```

Khi `isExpanded = true`: list bình thường, `flex flex-col gap-3`, không dùng absolute positioning.

Transition: `transition-all duration-300 ease-in-out`

```tsx
const [isExpanded, setIsExpanded] = useState(false)

<div
  onMouseEnter={() => setIsExpanded(true)}
  onMouseLeave={() => setIsExpanded(false)}
  className={`fixed bottom-6 left-6 z-50 w-80
    ${isExpanded ? 'flex flex-col gap-3' : 'relative'}`}
>
  {isExpanded
    ? notifications.filter(n => !n.is_read).slice(0, 5).map((n, i) => (
        <NotificationCard key={n.id} notification={n} onRead={markRead} />
      ))
    : /* collapsed stack */ ...
  }
</div>
```

### NotificationCard component

File: `frontend/src/components/notifications/NotificationCard.tsx`

```tsx
<div
  className="w-80 bg-white rounded-xl border border-[rgba(195,198,215,0.60)]
             shadow-lg p-3 cursor-pointer hover:bg-[rgba(242,243,255,0.80)]
             transition-colors select-none"
  onClick={() => {
    markRead(notification.id)
    if (notification.target_url) navigate(notification.target_url)
  }}
>
  <div className="flex items-start gap-3">
    {/* Type indicator dot */}
    <div className="w-2 h-2 rounded-full mt-1.5 shrink-0"
         style={{ background: TYPE_COLORS[notification.type] }} />
    <div className="flex-1 min-w-0">
      <p className="text-sm font-semibold text-[#131B2E] leading-snug truncate">
        {notification.title}
      </p>
      <p className="text-xs text-[#505F76] mt-0.5 line-clamp-2">
        {notification.message}
      </p>
      <p className="text-xs text-[#737686] mt-1">
        {getRelativeTime(notification.created_at)}
      </p>
    </div>
    {/* Type accent bar */}
    <div className="w-1 self-stretch rounded-full shrink-0"
         style={{ background: TYPE_COLORS[notification.type] }} />
  </div>
</div>
```

```typescript
const TYPE_COLORS = {
  success: '#059669',
  error:   '#BA1A1A',
  warning: '#D97706',
  info:    '#004AC6',
}
```

---

## 3. NotificationBell

File: `frontend/src/components/notifications/NotificationBell.tsx`

### Bell icon + badge

```tsx
<div className="relative">
  <button
    className="p-1 rounded-full hover:bg-[#F2F3FF] transition-colors"
    onClick={() => setIsOpen(prev => !prev)}
  >
    <Bell size={20} className="text-[#434655]" />
  </button>

  {/* Badge */}
  {unreadCount > 0 && (
    <div className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px]
                    bg-[#BA1A1A] text-white text-[10px] font-bold
                    rounded-full flex items-center justify-center px-1">
      {unreadCount > 99 ? '99+' : unreadCount}
    </div>
  )}
</div>
```

### Dropdown panel

Đóng khi click ngoài:
```typescript
// useEffect click outside
useEffect(() => {
  const handler = (e: MouseEvent) => {
    if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
      setIsOpen(false)
    }
  }
  document.addEventListener('mousedown', handler)
  return () => document.removeEventListener('mousedown', handler)
}, [])
```

Dropdown layout:
```tsx
{isOpen && (
  <div
    ref={dropdownRef}
    className="absolute right-0 top-full mt-2 w-96 bg-white rounded-xl
               shadow-2xl border border-[#C3C6D7] overflow-hidden z-50"
  >
    {/* Header */}
    <div className="flex items-center justify-between px-4 py-3
                    border-b border-[#C3C6D7]">
      <h3 className="text-base font-semibold text-[#131B2E]">Notifications</h3>
      <button
        className="text-xs text-[#004AC6] hover:underline font-medium"
        onClick={markAllRead}
      >
        Mark all as read
      </button>
    </div>

    {/* List */}
    <div className="overflow-y-auto max-h-[420px]">
      {notifications.length === 0 ? (
        <div className="py-8 text-center text-[#737686] text-sm">
          No notifications
        </div>
      ) : (
        notifications.map(n => (
          <NotificationDropdownItem
            key={n.id}
            notification={n}
            onRead={handleItemClick}
          />
        ))
      )}
    </div>

    {/* Load more */}
    <div className="border-t border-[#C3C6D7] p-3 text-center">
      <button
        className="text-sm text-[#004AC6] hover:underline font-medium"
        onClick={loadMore}
      >
        Load more Notifications
      </button>
    </div>
  </div>
)}
```

### NotificationDropdownItem component

File: `frontend/src/components/notifications/NotificationDropdownItem.tsx`

```tsx
<div
  className={`flex items-start gap-3 px-4 py-3 cursor-pointer
    hover:bg-[rgba(242,243,255,0.80)] transition-colors
    border-b border-[rgba(195,198,215,0.30)]
    ${!notification.is_read ? 'bg-[rgba(219,234,254,0.30)]' : 'bg-white'}`}
  onClick={() => {
    onRead(notification.id)
    if (notification.target_url) navigate(notification.target_url)
    setIsOpen(false)
  }}
>
  {/* Unread dot */}
  <div className="w-2 h-2 rounded-full mt-1.5 shrink-0"
       style={{
         background: !notification.is_read
           ? TYPE_COLORS[notification.type]
           : 'transparent',
         outline: notification.is_read ? `2px solid ${TYPE_COLORS[notification.type]}` : 'none'
       }} />

  <div className="flex-1 min-w-0">
    {/* Video name as title */}
    <p className={`text-sm leading-snug
      ${!notification.is_read ? 'font-semibold text-[#131B2E]' : 'font-medium text-[#434655]'}`}>
      {notification.title}
    </p>
    {/* Message */}
    <p className="text-xs text-[#505F76] mt-0.5 line-clamp-2">
      {notification.message}
    </p>
    {/* Timestamp */}
    <p className="text-xs text-[#737686] mt-1">
      {getRelativeTime(notification.created_at)}
    </p>
  </div>
</div>
```

---

## 4. Tích hợp vào App

### App.tsx

```tsx
import { NotificationProvider } from './context/NotificationContext'
import { NotificationStack } from './components/notifications/NotificationStack'

function App() {
  return (
    <NotificationProvider>
      <Router>
        <Sidebar />
        <Header />  {/* Header chứa NotificationBell */}
        <Routes>...</Routes>
        <NotificationStack />  {/* Mount 1 lần ở App level */}
      </Router>
    </NotificationProvider>
  )
}
```

### Header component

Tìm chỗ hiển thị icon chuông trong Header, thay div màu bằng:
```tsx
import { NotificationBell } from '../components/notifications/NotificationBell'

// Trong header icons row:
<NotificationBell />
```

---

## 5. API Functions (thêm vào api.ts)

```typescript
// GET /api/notifications?is_read=&limit=20&offset=0
export const getNotifications = async (params?: {
  is_read?: boolean
  limit?: number
  offset?: number
}): Promise<{ items: Notification[], total: number }> => {
  const res = await api.get('/api/notifications', { params })
  return res.data.data
}

// GET /api/notifications/unread-count
export const getUnreadCount = async (): Promise<number> => {
  const res = await api.get('/api/notifications/unread-count')
  return res.data.data.count
}

// PATCH /api/notifications/:id/read
export const patchNotificationRead = async (id: number): Promise<void> => {
  await api.patch(`/api/notifications/${id}/read`)
}

// PATCH /api/notifications/read-all
export const patchNotificationReadAll = async (): Promise<void> => {
  await api.patch('/api/notifications/read-all')
}
```

---

## 6. TypeScript interfaces (thêm vào types.ts)

```typescript
export interface Notification {
  id: number
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  target_url: string | null
  video_id: string | null
  is_read: boolean
  created_at: string
}

export interface NotificationListResponse {
  items: Notification[]
  total: number
}

export const TYPE_COLORS: Record<string, string> = {
  success: '#059669',
  error:   '#BA1A1A',
  warning: '#D97706',
  info:    '#004AC6',
}
```

---

## 7. Component Files

```
frontend/src/
├── context/
│   └── NotificationContext.tsx      ← Provider + polling + shared state
├── components/notifications/
│   ├── NotificationStack.tsx        ← Fixed stack góc dưới trái
│   ├── NotificationCard.tsx         ← Card trong stack
│   ├── NotificationBell.tsx         ← Bell icon + badge + dropdown
│   └── NotificationDropdownItem.tsx ← Item trong dropdown
```

---

## 8. Backend — create_notification helper

Thêm vào `backend/utils.py`:

```python
def create_notification(
    db,
    notification_type: str,    # "success" | "error" | "warning" | "info"
    title: str,
    message: str,
    target_url: str = None,
    video_id: str = None,
):
    from models import Notification
    from datetime import datetime
    notif = Notification(
        type=notification_type,
        title=title,
        message=message,
        target_url=target_url,
        video_id=video_id,
        is_read=0,
        created_at=datetime.utcnow().isoformat(),
    )
    db.add(notif)
    db.commit()
```

---

## 9. Worker trigger points

Trong `worker.py`, gọi `create_notification` đúng các điểm sau:

```python
# Sau khi ghi segments, trước khi update PENDING_CONFIRM:
if len(segments) > 0:
    create_notification(db,
        notification_type="success",
        title="Video detected as abnormal",
        message=f"Video {video.filename} has {len(segments)} anomaly segment(s) waiting for review.",
        target_url=f"/videos/{video_id}",
        video_id=video_id,
    )
    # Check low confidence
    low_conf = [s for s in segments if s.confidence_score < 0.6]
    if low_conf:
        create_notification(db,
            notification_type="warning",
            title="Low confidence detection",
            message=f"Video {video.filename} has {len(low_conf)} segment(s) with low confidence. Manual review recommended.",
            target_url=f"/videos/{video_id}",
            video_id=video_id,
        )
else:
    create_notification(db,
        notification_type="info",
        title="Video processing complete",
        message=f"Video {video.filename} processed with no anomaly detected.",
        target_url=f"/videos/{video_id}",
        video_id=video_id,
    )

# Nếu FAILED:
create_notification(db,
    notification_type="error",
    title="Video processing failed",
    message=f"Video {video.filename} failed during processing. {error_message}",
    target_url=f"/videos/{video_id}",
    video_id=video_id,
)

# Sau khi batch hoàn tất (tất cả video trong batch đạt terminal status):
# Đếm success/fail trong batch, ghi 1 notification tổng
success_count = ...
total_count = ...
create_notification(db,
    notification_type="info",
    title="Batch processing complete",
    message=f"{success_count} of {total_count} videos processed successfully.",
    target_url=f"/queue",
    video_id=None,
)
```

---

## 10. Acceptance Criteria (test checklist)

```
Backend:
- [ ] Bảng notifications tồn tại trong DB
- [ ] Upload + xử lý video thành công → tạo notification type=success
- [ ] Upload + xử lý thất bại → tạo notification type=error
- [ ] Segment confidence < 0.6 → tạo notification type=warning
- [ ] GET /api/notifications trả về đúng format
- [ ] PATCH /api/notifications/:id/read cập nhật is_read=1
- [ ] PATCH /api/notifications/read-all cập nhật tất cả

Frontend:
- [ ] Bell badge hiển thị đúng số unread, ẩn khi = 0
- [ ] Click bell → dropdown mở, click ngoài → đóng
- [ ] Dropdown list đúng thứ tự (mới nhất trước)
- [ ] Unread items có background highlight, dot đầy màu
- [ ] Click item dropdown → navigate + mark read + badge giảm
- [ ] "Mark all as read" → badge = 0, stack trống
- [ ] Stack hiện ở góc dưới trái, không che nội dung
- [ ] Stack collapsed: cards xếp chồng có offset
- [ ] Stack hover → expand thành list, mouse leave → collapse
- [ ] Click card stack → navigate + biến khỏi stack
- [ ] "+N more" badge khi có > 3 unread
- [ ] Polling 10s cập nhật notification mới (test bằng upload video)
- [ ] Stack + bell đồng bộ state (mark read ở bell → biến khỏi stack và ngược lại)
- [ ] npm run build không có TypeScript errors
```
