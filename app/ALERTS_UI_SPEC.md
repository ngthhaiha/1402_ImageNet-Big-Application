# ALERTS_UI_SPEC.md
# UI Specification — Alerts Page (Task 18 + 19)
# Nguồn: Figma export (alerts.txt) + ảnh screenshot

## Layout tổng quan

```
[Sidebar 240px fixed]  |  [Content area — padding 24px, bg #FAF8FF]
                           ├─ Header (breadcrumb + title)
                           ├─ 4 Summary Cards (grid 4 cột)
                           ├─ Filter Bar (full width)
                           ├─ Row A: Alert Log (LEFT ~65%) | Distribution + Promo (RIGHT ~35%)
                           └─ Row B: Recent Critical Alerts (full width)
```

Gap giữa sections: `24px`
Card style: `bg white`, `border-radius 12px`, `outline 1px #C3C6D7`, `padding 24px`

---

## Design Tokens (Alerts-specific)

```
-- Summary Cards --
Card 1 sub-text:    #16A34A  (xanh lá, icon trend-up)
Card 2 sub-text:    #BA1A1A  (đỏ, icon alert-triangle)
Card 4 sub-text:    #004AC6  (xanh)

-- Severity badges (khác Dashboard!) --
HIGH:    bg #FFDAD6   text #93000A   border-radius 9999  padding 4px 8px
MEDIUM:  bg #BC4800   text #FFEDE6   border-radius 9999  padding 4px 8px
LOW:     bg #DAE2FD   text #434655   border-radius 9999  padding 4px 8px

-- Status dots --
Unreviewed (PENDING_REVIEW):  dot #737686  text #434655
Reviewed (others):            dot #004AC6  text #004AC6

-- Alert Log table --
Header bg:    #F2F3FF
Row border:   1px #C3C6D7 solid (top)
"View Investigation": text #004AC6  16px weight 400  (link text, no bg)

-- Recent Critical Alerts --
Header bg:    rgba(255, 218, 214, 0.20)
Header title: #BA1A1A
"Clear Feed": #BA1A1A weight 700
Activity icon container: 32×32 bg rgba(186,26,26,0.10) border-radius 4px
Confidence text: #BA1A1A weight 700
"ACTIVE" status: dot #BA1A1A + text #BA1A1A weight 700
"View Detail": bg #BA1A1A text white border-radius 4px  (primary button, đỏ)
  hoặc: outline #BA1A1A text #BA1A1A (secondary)

-- Alert Distribution card --
Bar track: #E2E7FF  height 8px  border-radius 9999
Bar fill:  #004AC6
Label: #434655 16px weight 400
% text: #131B2E 16px weight 700

-- Speed Up Analysis card --
bg: #2563EB
Title: #EEEFFF 16px weight 400
Body: #EEEFFF 16px weight 400 opacity 0.90
Button "Try Now": bg #EEEFFF text #2563EB border-radius 12px padding 4px 16px

-- Filter bar --
Search input: bg white, outline 1px #C3C6D7, border-radius 12px, icon search #737686
Dropdown: bg white, outline 1px #C3C6D7, border-radius 12px, chevron icon #6B7280
Date picker: bg white, outline 1px #C3C6D7, border-radius 12px
Reset Filters: outline 1px #737686, text #434655, border-radius 12px

-- Pagination --
Active page: bg #2563EB text #EEEFFF border-radius 4px
Inactive: text #131B2E
Prev/Next: opacity 0.30 khi disabled

-- "Showing X of Y": #434655 16px weight 400
```

---

## Section 1: Header

```
Cases  >  Alert                    [Emergency Call] [bell] [settings] [?] [avatar]
──────────────────────────────────────────────────────────────────────────────────
System Alerts
Monitor and review abnormal events detected from uploaded surveillance videos.
```

- "Cases": color `#434655` 14px
- "Alert": color `#004AC6` 14px weight 700, border-bottom `2px #004AC6`
- Title: `#131B2E` 16px weight 400 (nhỏ hơn Dashboard)
- Sub-text: `#434655` 16px weight 400

---

## Section 2: Summary Cards (4 ngang)

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ TOTAL ALERTS  📊│ │ HIGH SEVERITY  !│ │ PENDING         │ │ REVIEWED        │
│                 │ │                 │ │ REVIEWS      📋 │ │ ALERTS       ✓  │
│ 1,284           │ │ 42              │ │ 18              │ │ 1,224           │
│ ↑ +12% from    │ │ △ 5 active now  │ │ Awaiting human  │ │ 95.3% accuracy  │
│   last week     │ │                 │ │ validation      │ │ rate            │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

Card structure (mỗi card):
```tsx
<div className="bg-white rounded-xl border border-[#C3C6D7] p-6 flex flex-col justify-between">
  {/* Top row: label + icon */}
  <div className="flex justify-between items-start">
    <p className="text-[#434655] text-base uppercase tracking-wider">{label}</p>
    {icon}  {/* lucide icon, màu theo card */}
  </div>
  {/* Value */}
  <div className="mt-4">
    <p className="text-[#131B2E] text-base">{value.toLocaleString()}</p>
    {/* Sub-text */}
    <div className="flex items-center gap-1 mt-1">
      {trendIcon}
      <p style={{ color: subColor }}>{subText}</p>
    </div>
  </div>
</div>
```

Icon per card:
- Card 1: `<BarChart2>` color `#004AC6`
- Card 2: `<AlertCircle>` color `#BA1A1A`
- Card 3: `<ClipboardList>` color `#505F76`
- Card 4: `<CheckCircle>` color `#004AC6`

API: `GET /api/alerts/stats` →
```typescript
interface AlertStats {
  total_alerts: number
  high_severity: number
  pending_reviews: number
  reviewed_alerts: number
}
```

---

## Section 3: Filter Bar

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [🔍 Filter by video name...]  [Activity Type ▾]  [Severity ▾]  [Status ▾]  │
│ [mm/dd/yyyy 📅]  [Reset Filters]                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

Filter state:
```typescript
interface AlertFilter {
  name: string           // search by video filename
  activity: string       // '' = All, else predicted_class
  severity: string       // '' | 'HIGH' | 'MEDIUM' | 'LOW'
  status: string         // '' | 'PENDING_REVIEW' | 'REVIEWED'
  date: string           // 'YYYY-MM-DD' hoặc ''
}
```

- Search input: full-text, debounce 300ms trước khi gọi API
- 3 dropdowns: mỗi cái có "All" option đầu tiên
- Date picker: `<input type="date">` styled theo Figma
- Reset: clear toàn bộ filter state, gọi lại API

---

## Section 4 (Row A): Alert Log | Distribution + Promo

Layout: `flex gap-6`
LEFT: `flex-1` (chiếm ~65%)
RIGHT: `w-72 shrink-0` (chiếm ~35%)

### LEFT — Alert Log

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Alert Log                                              [≡] [↓]           │
│ ─────────────────────────────────────────────────────────────────────── │
│ [#F2F3FF] TIME  VIDEO NAME  ACTIVITY TYPE  CONFIDENCE  SEVERITY  STATUS  ACTION │
│ ─────────────────────────────────────────────────────────────────────── │
│ 14:24:02  Cam_04_North  Road Accident  94.2%  [HIGH]  ● Unreviewed  View Investigation │
│ 13:58:15  St_Cross_01   Shoplifting    88.7%  [MED]   ● Unreviewed  View Investigation │
│ 12:44:30  Store_Retail  Fighting       91.5%  [HIGH]  ● Reviewed    View Investigation │
│ 10:12:05  Parking_Lot   Burglary       76.3%  [LOW]   ● Reviewed    View Investigation │
│ ─────────────────────────────────────────────────────────────────────── │
│ [#F2F3FF] Showing 4 of 1,284 results                [<] [1] [>]         │
└──────────────────────────────────────────────────────────────────────────┘
```

Table header: `bg-[#F2F3FF]` `border-b border-[#C3C6D7]` padding 16px
Header text: `#434655` 16px weight 700 uppercase

Row: `border-t border-[#C3C6D7]` padding `26px 24px`
Row hover: `bg-gray-50`

Column widths (approximate):
- TIME: 115px
- VIDEO NAME: 160px
- ACTIVITY TYPE: 130px
- CONFIDENCE: 150px
- SEVERITY: 125px
- STATUS: 125px
- ACTION: 145px

"View Investigation" click → `navigate('/videos/${video_id}?segment=${segment_id}')` 

Footer: `bg-[#F2F3FF]` `border-t border-[#C3C6D7]` padding 16px
- "Showing X of Y results": left
- Pagination buttons: right

Pagination:
```tsx
<div className="flex items-center gap-2">
  <button disabled={page===1} className="p-2 opacity-30">‹</button>
  {pageNumbers.map(p => (
    <button key={p}
      className={p === page
        ? 'px-2 py-1 bg-[#2563EB] text-[#EEEFFF] rounded font-bold'
        : 'px-2 py-1 text-[#131B2E]'}
      onClick={() => setPage(p)}>{p}</button>
  ))}
  <button disabled={page===totalPages} className="p-2">›</button>
</div>
```

API: `GET /api/alerts/log?name=&activity=&severity=&status=&date=&page=1&limit=10`

Response:
```typescript
interface AlertLogResponse {
  items: AlertLogItem[]
  total: number
  page: number
  total_pages: number
}
interface AlertLogItem {
  segment_id: number
  video_id: string
  video_name: string        // videos.filename
  time: string              // HH:MM:SS
  activity_type: string     // predicted_class
  confidence: number
  anomaly_score: number
  severity: 'HIGH' | 'MEDIUM' | 'LOW'   // tính ở backend
  review_status: string
}
```

---

### RIGHT — 2 cards xếp dọc (gap 24px)

**Card 1: Alert Distribution**

```
┌──────────────────────────────────┐
│ Alert Distribution            ⋮  │
│ ──────────────────────────────── │
│ Road Accident          35%       │
│ [████████░░░░░░░░░░░░░░]         │
│ Shoplifting            25%       │
│ [██████░░░░░░░░░░░░░░░░]         │
│ Robbery                20%       │
│ [█████░░░░░░░░░░░░░░░░░]         │
│ Fighting               15%       │
│ [███░░░░░░░░░░░░░░░░░░░]         │
│ Burglary               5%        │
│ [█░░░░░░░░░░░░░░░░░░░░░]         │
└──────────────────────────────────┘
```

Không dùng Recharts — CSS div thuần:
```tsx
{distribution.map(item => (
  <div key={item.class} className="flex flex-col gap-1">
    <div className="flex justify-between">
      <span className="text-[#434655] text-base">{item.class}</span>
      <span className="text-[#131B2E] text-base font-bold">{item.percentage}%</span>
    </div>
    <div className="h-2 bg-[#E2E7FF] rounded-full overflow-hidden">
      <div
        className="h-full bg-[#004AC6] rounded-full"
        style={{ width: `${item.percentage}%` }}
      />
    </div>
  </div>
))}
```

API: `GET /api/alerts/distribution` → `[{class, count, percentage}]` top 5 sort DESC

**Card 2: Speed Up Analysis** (STATIC — không có data)

```tsx
<div className="bg-[#2563EB] rounded-xl p-6 relative overflow-hidden">
  <h3 className="text-[#EEEFFF] text-base">Speed Up Analysis</h3>
  <p className="text-[#EEEFFF] text-base opacity-90 mt-1 mb-3">
    Use the 'Auto-Validate' feature for Low severity events to focus on critical threats.
  </p>
  <button
    className="bg-[#EEEFFF] text-[#2563EB] px-4 py-1 rounded-xl"
    onClick={() => toast('Coming soon')}
  >
    Try Now
  </button>
</div>
```

---

## Section 5: Recent Critical Alerts (full width)

Data: segments có `anomaly_score ≥ 0.85`, sort DESC, limit 10

```
┌──────────────────────────────────────────────────────────────────────────┐
│ [△] Recent Critical Alerts                            [Clear Feed]       │
│ ─────────────────────────────────────────────────────────────────────── │
│ [#F2F3FF] TIME  ACTIVITY  CONFIDENCE  STATUS  ACTION                    │
│ ─────────────────────────────────────────────────────────────────────── │
│ 14:24:02  [icon] Road Accident   94.2%   ● ACTIVE   [View Detail]       │
│ 12:44:30  [icon] Fighting        91.5%   ESCALATED  [View Detail]       │
└──────────────────────────────────────────────────────────────────────────┘
```

Header:
```tsx
<div className="bg-[rgba(255,218,214,0.20)] border-b border-[#C3C6D7] p-4 flex justify-between">
  <div className="flex items-center gap-2">
    <AlertTriangle size={22} className="text-[#BA1A1A]" />
    <span className="text-[#BA1A1A] text-base">Recent Critical Alerts</span>
  </div>
  <button onClick={() => toast('Coming soon')}
    className="text-[#BA1A1A] font-bold text-base">
    Clear Feed
  </button>
</div>
```

Column widths:
- TIME: 145px
- ACTIVITY: 250px (icon 32×32 + text)
- CONFIDENCE: 190px
- STATUS: 175px
- ACTION: 210px

Activity cell:
```tsx
<div className="flex items-center gap-2">
  <div className="w-8 h-8 bg-[rgba(186,26,26,0.10)] rounded flex items-center justify-center">
    <AlertTriangle size={16} className="text-[#BA1A1A]" />
  </div>
  <span className="text-[#131B2E] text-base">{activity_type}</span>
</div>
```

Confidence: `text-[#BA1A1A] font-bold`

Status mapping (Critical Alerts — đơn giản hơn):
- PENDING_REVIEW: dot `#BA1A1A` animate-pulse + "ACTIVE" text `#BA1A1A` weight 700
- Đã feedback: text "ESCALATED" màu `#434655`

Action button:
```tsx
// Nếu PENDING_REVIEW: primary đỏ
<button className="bg-[#BA1A1A] text-white px-4 py-1 rounded"
  onClick={() => navigate(`/videos/${video_id}?segment=${segment_id}`)}>
  View Detail
</button>

// Nếu đã reviewed: outline đỏ
<button className="border border-[#BA1A1A] text-[#BA1A1A] px-4 py-1 rounded"
  onClick={() => navigate(`/videos/${video_id}?segment=${segment_id}`)}>
  View Detail
</button>
```

API: `GET /api/alerts/critical?limit=10`

Response:
```typescript
interface CriticalAlertItem {
  segment_id: number
  video_id: string
  time: string
  activity_type: string
  confidence: number
  anomaly_score: number
  review_status: string
}
```

---

## Video Detail — xử lý `?segment=` query param

Khi navigate từ Alerts page: `/videos/:video_id?segment=:segment_id`

Cập nhật VideoDetail.tsx:
```typescript
// Đọc query param
const [searchParams] = useSearchParams()
const segmentParam = searchParams.get('segment')  // segment_id dạng string

// Sau khi load segments:
useEffect(() => {
  if (segmentParam && segments.length > 0) {
    const targetId = parseInt(segmentParam)
    const found = segments.find(s => s.id === targetId)
    if (found) {
      setSelectedSegmentId(targetId)
      // Seek video đến start_time
      if (videoRef.current) {
        videoRef.current.currentTime = found.start_time
      }
      // Scroll segment table row vào view
      document.getElementById(`segment-row-${targetId}`)?.scrollIntoView({ behavior: 'smooth' })
    }
  }
}, [segmentParam, segments])
```

Thêm `id={`segment-row-${segment.id}`}` vào mỗi `<tr>` trong SegmentsTable.

---

## Component Files

```
frontend/src/pages/Alerts.tsx                ← page chính
frontend/src/components/alerts/
  ├── AlertStatsCard.tsx                     ← 1 summary card
  ├── AlertFilterBar.tsx                     ← filter controls
  ├── AlertLogTable.tsx                      ← table + pagination
  ├── AlertDistribution.tsx                  ← CSS bar chart
  └── CriticalAlertsTable.tsx               ← recent critical table
```

---

## API Endpoints

| Endpoint | Backend file |
|---|---|
| `GET /api/alerts/stats` | `backend/routers/alerts.py` |
| `GET /api/alerts/log` | `backend/routers/alerts.py` |
| `GET /api/alerts/distribution` | `backend/routers/alerts.py` |
| `GET /api/alerts/critical` | `backend/routers/alerts.py` |
