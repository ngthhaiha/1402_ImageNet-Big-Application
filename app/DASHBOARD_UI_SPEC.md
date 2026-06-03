# DASHBOARD_UI_SPEC.md
# UI Specification — Dashboard Page (Task 12)
# Nguồn: Figma export (dashboard.txt) + Dashboard.pdf

## Layout tổng quan

```
[Sidebar 240px fixed — đã có sẵn]  |  [Content area]
                                        padding: 32px
                                        background: #FAF8FF
```

Thứ tự sections từ trên xuống:
```
1. Welcome Banner
2. Stats Cards (4 ngang)
3. Row A: Anomaly Distribution (LEFT ~58%) | Recent Activity (RIGHT ~40%)
4. Row B: Recent Alerts (LEFT ~58%) | Top Detections (RIGHT ~40%)
5. Row C: Recent Investigations (full width)
```

Gap giữa các section: `32px`  
Card style chung: `bg white`, `border-radius 12px`, `border 1px rgba(195,198,215,0.30)`, `box-shadow 0px 1px 2px rgba(0,0,0,0.05)`

---

## Design Tokens

### Colors
```
Background page:       #FAF8FF
Card bg:               white
Card border:           1px rgba(195,198,215,0.30)  ← mờ hơn các page khác

-- Stats badges --
Badge +12%:            bg rgba(0,74,198,0.10)    text #004AC6
Badge +3%:             bg rgba(186,26,26,0.10)   text #BA1A1A
Badge 24 New:          bg rgba(148,55,0,0.10)    text #943700
Badge 98% Acc.:        bg rgba(80,95,118,0.10)   text #505F76

-- Stats values --
Card 1 value:          #131B2E  (xanh đen — bình thường)
Card 2 value:          #BA1A1A  (đỏ — anomaly count)
Card 3 value:          #131B2E
Card 4 value:          #131B2E

-- Donut chart colors (theo thứ tự trong PDF) --
Fighting:              #BA1A1A  (đỏ)
Robbery:               #004AC6  (xanh đậm)
Road Accident:         #4CAF50  (xanh lá)
Shooting:              #F59E0B  (vàng cam)
Burglary:              #9C27B0  (tím)
Other:                 #9E9E9E  (xám)

-- Recent Activity dots --
UPLOAD (Video uploaded):          #004AC6  (xanh)
REVIEW_COMPLETE (Analysis/Feedback): #4CAF50  (xanh lá)
FLAG (Feedback submitted):        #BA1A1A  (đỏ cam)
EXPORT (Report exported):         #9E9E9E  (xám)

-- Severity badges --
HIGH:    bg rgba(186,26,26,0.10)   text #BA1A1A   border-radius 4px
MEDIUM:  bg rgba(245,158,11,0.10)  text #D97706   border-radius 4px
LOW:     bg rgba(80,95,118,0.10)   text #505F76   border-radius 4px

-- Alert Status --
Unreviewed:    dot #943700  text #943700
Processing:    text #505F76  (không có dot)
Validated:     dot #059669  text #059669  (có icon check)
False Positive: text #BA1A1A (không có dot, chữ đỏ)

-- Confidence bar (Recent Investigations) --
Track:   #E2E7FF
HIGH ≥85%:  fill #BA1A1A  text #BA1A1A  weight 700
MED 65-84%: fill #F59E0B  text #D97706  weight 700
LOW <65%:   fill #004AC6  text #004AC6  weight 700

-- Investigation Review Status badges --
HIGH ALERT:  bg rgba(186,26,26,0.10)  text #BA1A1A  border-radius 9999
IN REVIEW:   bg rgba(80,95,118,0.10)  text #505F76  border-radius 9999
VALIDATED:   bg rgba(0,74,198,0.10)   text #004AC6  border-radius 9999  (estimate từ PDF)

-- Top Detections --
Bar fill:    #004AC6
Bar track:   #E2E7FF

-- Buttons --
"View Investigation": bg #E2E7FF  text #131B2E  border-radius 8px  padding 8px 16px
"View All":           text #004AC6  12px weight 500  letterSpacing 0.6
"Load more":          text #505F76  centered  có chevron down icon
"Upload New Video":   bg #004AC6  text white  icon Cloud  border-radius 8px
"Filter":             outline border #C3C6D7  text #434655  icon Filter
"Export Data":        outline border #C3C6D7  text #434655  icon Download
```

### Typography
```
Welcome title:         32px  weight 700  #131B2E
Welcome subtitle:      16px  weight 400  #505F76
Stats badge:           12px  weight 700
Stats value:           48px  weight 700  lineHeight 56px
Stats label:           16px  weight 600  #434655
Stats description:     12px  weight 500  letterSpacing 0.6  #737686
Section title:         20px  weight 600  #131B2E  (Recent Alerts, Recent Investigations)
Section title small:   18px  weight 600  (Anomaly Distribution, Top Detections, Recent Activity)
Table header:          12px  weight 500  letterSpacing 0.6  #737686
Table body primary:    16px  weight 600  #131B2E  (Activity Type, Video Name)
Table body secondary:  16px  weight 400  #131B2E  (Time, Confidence %)
Severity badge text:   10px  weight 700  uppercase
Status text:           12px  weight 400  lineHeight 16px
Top Detections label:  14px  weight 600  #131B2E
Top Detections count:  14px  weight 600  #131B2E  right-aligned
Donut legend label:    12px  weight 500  #434655
Donut legend pct:      12px  weight 400  #737686
Donut center value:    32px  weight 700  #131B2E
Donut center label:    12px  weight 500  #737686  uppercase
Recent Activity title: 14px  weight 600  #131B2E
Recent Activity sub:   12px  weight 400  #737686
Investigation sub:     14px  weight 400  #505F76  (sub-text dưới title)
Video filename:        16px  weight 600  #131B2E
Video meta:            10px  weight 400  #737686  (1080p • 2:45 min)
```

---

## Section 1: Welcome Banner

```
┌──────────────────────────────────────────────────────────────────┐
│ Welcome back                              [☁ Upload New Video]   │
│ Monitor abnormal activities detected from                        │
│ surveillance videos and review investigation results.            │
└──────────────────────────────────────────────────────────────────┘
```

- Padding: 24px–32px
- Title: 32px weight 700
- Subtitle: 16px weight 400 `#505F76`, 2 dòng
- Nút "Upload New Video": bg `#004AC6`, text white, icon `<Cloud>` lucide, border-radius 8px, padding `12px 24px`
- Nút nằm vertical-center bên phải

---

## Section 2: Stats Cards

Grid 4 cột ngang, gap `16px` (estimate — có thể 0 tùy design).

### Card structure (mỗi card):
```
┌─────────────────────────┐
│ [icon 36px]    [+12%]   │  ← icon trái, badge phải
│                         │
│ 1,284                   │  ← value 48px bold
│ Total Videos            │  ← label 16px weight 600
│ Analyzed                │
│ Total uploaded videos   │  ← description 12px #737686
│ processed by AI         │
└─────────────────────────┘
```

| # | Icon bg | Badge | Value color | Label | Description |
|---|---|---|---|---|---|
| 1 | `rgba(0,74,198,0.10)` icon `#004AC6` | `+12%` bg `rgba(0,74,198,0.10)` text `#004AC6` | `#131B2E` | Total Videos Analyzed | Total uploaded videos processed by AI |
| 2 | `rgba(186,26,26,0.10)` icon `#BA1A1A` | `+3%` bg `rgba(186,26,26,0.10)` text `#BA1A1A` | **`#BA1A1A`** | Abnormal Events Detected | Total abnormal segments detected |
| 3 | `rgba(148,55,0,0.10)` icon `#943700` | `24 New` bg `rgba(148,55,0,0.10)` text `#943700` | `#131B2E` | Pending Reviews | Segments waiting for user validation |
| 4 | `rgba(80,95,118,0.10)` icon `#505F76` | `98% Acc.` bg `rgba(80,95,118,0.10)` text `#505F76` | `#131B2E` | Reviewed Cases | Validated anomaly events |

Badge: padding `4px 8px`, border-radius `4px`  
Icon container: `36px × 36px`, border-radius `8px`  
Badge text: 12px weight 700  
Badges và % trend: **hard-code** (demo, không tính thật)

### Data từ API `GET /api/dashboard/stats`:
```typescript
interface DashboardStats {
  total_videos: number         // COUNT(*) FROM videos
  total_anomalies: number      // COUNT(*) FROM anomaly_segments (chỉ predicted_class != 'Normal')
  pending_reviews: number      // COUNT(*) FROM anomaly_segments WHERE feedback_submitted_at IS NULL
                               //   AND video.status IN ('PENDING_CONFIRM')
  reviewed_cases: number       // COUNT(*) FROM anomaly_segments WHERE feedback_submitted_at IS NOT NULL
}
```

---

## Section 3 (Row A): Anomaly Distribution | Recent Activity

Layout: `flex gap-6`, LEFT `flex-1`, RIGHT `flex-shrink-0 w-[340px]` (estimate)

### LEFT — Anomaly Distribution Card

```
┌────────────────────────────────────────────────────────────┐
│ Anomaly Distribution                                  [···] │
│                                                            │
│   ┌──────┐    ● Fighting      32% (50)  ● Robbery  24%(38) │
│   │  156 │    ● Road Accident 18% (28)  ● Shooting 12%(19) │
│   │ TOTAL│    ● Burglary      10% (16)  ● Other     4% (5) │
│   └──────┘                                                  │
└────────────────────────────────────────────────────────────┘
```

**Donut chart (Recharts)**:
```tsx
<PieChart width={180} height={180}>
  <Pie
    data={distribution}
    cx="50%" cy="50%"
    innerRadius={65}
    outerRadius={90}
    dataKey="count"
    paddingAngle={2}
  >
    {distribution.map((entry, i) => (
      <Cell key={i} fill={CLASS_COLORS[entry.class]} />
    ))}
  </Pie>
</PieChart>
```

**Center label** — absolute positioned div chồng lên chart:
```tsx
<div className="absolute inset-0 flex flex-col items-center justify-center">
  <span className="text-3xl font-bold text-[#131B2E]">{total}</span>
  <span className="text-xs font-medium text-[#737686] uppercase tracking-wider">TOTAL</span>
</div>
```

**Legend** — bên phải chart, 2 cột grid:
```
● Fighting      32% (50)    ● Robbery    24% (38)
● Road Accident 18% (28)    ● Shooting   12% (19)
● Burglary      10% (16)    ● Other       4%  (5)
```
- Dot: 8px circle, màu theo class
- Label: 12px weight 500 `#434655`
- Pct + count: 12px weight 400 `#737686`

**Icon `...`** góc phải title: `<MoreHorizontal>` lucide, 16px `#737686`

**Data từ API `GET /api/dashboard/distribution`**:
```typescript
interface DistributionItem {
  class: string       // predicted_class
  count: number
  percentage: number  // làm tròn 1 chữ số thập phân
}
```

**CLASS_COLORS constant** (dùng cả trong chart lẫn legend):
```typescript
export const CLASS_COLORS: Record<string, string> = {
  Fighting:     '#BA1A1A',
  Robbery:      '#004AC6',
  RoadAccidents:'#4CAF50',
  Shooting:     '#F59E0B',
  Burglary:     '#9C27B0',
  Abuse:        '#E91E63',
  Arrest:       '#00BCD4',
  Arson:        '#FF5722',
  Assault:      '#795548',
  Burglary:     '#9C27B0',
  Stealing:     '#607D8B',
  Vandalism:    '#FF9800',
  Other:        '#9E9E9E',
  Normal:       '#4CAF50',
}
```

---

### RIGHT — Recent Activity Card

```
┌──────────────────────────────────────────┐
│ Recent Activity                          │
│ ──────────────────────────────────────── │
│ ● Video uploaded                         │
│   North Gate Camera 04 • 2m ago          │
│                                          │
│ ● Analysis completed                     │
│   Parking Lot West Exit • 14m ago        │
│                                          │
│ ● Feedback submitted                     │
│   Incident ID #4928 • 1h ago             │
│                                          │
│ ● Report exported                        │
│   Daily Summary - Oct 24 • 3h ago        │
└──────────────────────────────────────────┘
```

- 4–5 items từ `activity_log` (hoặc ghép query nếu chưa có bảng)
- Mỗi item: dot 8px circle + title 14px weight 600 + sub-text 12px `#737686`
- Sub-text format: `{detail} • {relative_time}`
- Không có divider giữa items, gap `16px`
- Không click (display only)

**Dot colors**:
```
UPLOAD:          #004AC6
REVIEW_COMPLETE: #4CAF50
FLAG:            #BA1A1A  (Feedback submitted)
EXPORT:          #9E9E9E  (Report exported — hard-code nếu chưa có type này)
```

**Data từ API `GET /api/dashboard/recent-activity?limit=5`**:
```typescript
interface ActivityItem {
  type: 'UPLOAD' | 'REVIEW_COMPLETE' | 'FLAG'
  title: string     // "Video uploaded", "Analysis completed", v.v.
  detail: string    // "North Gate Camera 04", "Incident ID #4928"
  created_at: string
}
```

---

## Section 4 (Row B): Recent Alerts | Top Detections

Layout: `flex gap-6`, LEFT `flex-1`, RIGHT `flex-shrink-0 w-[320px]`

### LEFT — Recent Alerts

```
┌────────────────────────────────────────────────────────────┐
│ Recent Alerts                                    View All  │
│ ──────────────────────────────────────────────────────────│
│ TIME      ACTIVITY TYPE  CONFIDENCE  SEVERITY  STATUS      │
│ ──────────────────────────────────────────────────────────│
│ 14:24:02  Fighting        94.2%      [HIGH]    ●Unreviewed │
│ 13:58:15  Road Accident   88.7%      [MEDIUM]  Processing  │
│ 12:44:30  Burglary        91.5%      [HIGH]    ✓Validated  │
│ 10:12:05  Other           76.3%      [LOW]     False Pos.  │
└────────────────────────────────────────────────────────────┘
```

**Table header row**: bg `rgba(242,243,255,0.50)`, padding `16px`  
**Table row**: border-top `1px rgba(195,198,215,0.20)`, padding `16px`  
**Row click**: navigate `/videos/:video_id`  
**"View All"**: text `#004AC6` 12px weight 500 — UI only, toast "Coming soon"

**Columns**:

| Column | Width | Content |
|---|---|---|
| Time | ~100px | `HH:MM:SS` từ `anomaly_segments.created_at`, 16px weight 400 |
| Activity Type | ~150px | `predicted_class`, 16px weight 600 |
| Confidence | ~110px | `confidence_score` format `XX.X%`, 16px weight 400 |
| Severity | ~90px | Badge tính từ `anomaly_score` (xem bảng dưới) |
| Status | ~115px | Dot + text tính từ `review_status` |

**Severity từ anomaly_score** (tính ở backend):
```python
def get_severity(anomaly_score: float) -> str:
    if anomaly_score >= 0.85: return 'HIGH'
    if anomaly_score >= 0.65: return 'MEDIUM'
    return 'LOW'
```

**Status display từ review_status**:
```typescript
const STATUS_DISPLAY = {
  'PENDING_REVIEW': { dot: '#943700', text: 'Unreviewed' },
  'LABEL_CORRECT':  { dot: '#059669', text: 'Validated', icon: 'check' },
  'CORRECTED':      { dot: null,      text: 'False Positive', color: '#BA1A1A' },
  'LOGGED':         { dot: '#004AC6', text: 'Logged' },
  // video đang processing (chưa có review_status):
  'PROCESSING':     { dot: null,      text: 'Processing', color: '#505F76' },
}
```

**Filter params** (apply khi có filter): `?class=Fighting&date_from=2024-01-01&date_to=2024-12-31&limit=10`

**Data từ API `GET /api/dashboard/recent-alerts`**:
```typescript
interface AlertItem {
  id: number
  video_id: string
  time: string           // "HH:MM:SS"
  activity_type: string  // predicted_class
  confidence: number     // 0–1
  anomaly_score: number  // 0–1
  severity: 'HIGH' | 'MEDIUM' | 'LOW'  // tính ở backend
  review_status: string
  is_correct: boolean | null
}
```

---

### RIGHT — Top Detections

```
┌──────────────────────────────────────────┐
│ Top Detections                           │
│ ──────────────────────────────────────── │
│ Fighting    [████████████████████]  50   │
│ Robbery     [████████████████]      38   │
│ Road Accident [████████████]        28   │
│ Shooting    [████████]              19   │
│ Burglary    [███████]               16   │
│ Other       [█]                      5   │
└──────────────────────────────────────────┘
```

**Recharts BarChart**:
```tsx
<BarChart
  layout="vertical"
  data={topDetections}
  margin={{ left: 0, right: 40, top: 0, bottom: 0 }}
>
  <XAxis type="number" hide />
  <YAxis
    type="category"
    dataKey="class"
    width={110}
    tick={{ fontSize: 14, fontWeight: 600, fill: '#131B2E' }}
    axisLine={false}
    tickLine={false}
  />
  <Bar dataKey="count" fill="#004AC6" radius={4} background={{ fill: '#E2E7FF', radius: 4 }}>
    <LabelList dataKey="count" position="right" style={{ fontSize: 14, fontWeight: 600, fill: '#131B2E' }} />
  </Bar>
</BarChart>
```
- `ResponsiveContainer width="100%" height={240}`
- Không có gridlines, không có tooltip

**Data từ API `GET /api/dashboard/top-detections`**:
```typescript
interface TopDetection {
  class: string   // predicted_class
  count: number
}
// Sort DESC, limit 6
```

---

## Section 5 (Row C): Recent Investigations

Full width card.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Recent Investigations                              [Filter] [↓ Export Data]│
│ Detailed log of AI-assisted surveillance analysis                         │
│ ──────────────────────────────────────────────────────────────────────── │
│ VIDEO NAME          DETECTED ACTIVITY  CONFIDENCE  REVIEW STATUS  CREATED │
│ ──────────────────────────────────────────────────────────────────────── │
│ [thumb] Cam_04...   Physical           [▓▓▓▓] 94%  [HIGH ALERT]  Today   [View] │
│         1080p•2:45  Altercation                                   14:24         │
│ ──────────────────────────────────────────────────────────────────────── │
│ [thumb] St_Cross... Vehicle            [▓▓▓] 88%   [IN REVIEW]   Today   [View] │
│         4K•1:12     Collision                                      13:58         │
│ ──────────────────────────────────────────────────────────────────────── │
│ [thumb] Store...    Unauthorized       [▓▓▓▓] 91%  [VALIDATED]   Today   [View] │
│         1080p•5:30  Entry                                          12:44         │
│ ──────────────────────────────────────────────────────────────────────── │
│                    Load more investigations ∨                             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Header**:
- Title: "Recent Investigations" 20px weight 600
- Sub-text: "Detailed log of AI-assisted surveillance analysis" 14px `#505F76`
- Buttons góc phải: Filter (outline) + Export Data (outline + icon Download)
- Filter button → toggle filter panel (class dropdown + date range)
- Export Data → toast "Coming soon"

**Table header row**: bg `rgba(242,243,255,0.50)` hoặc transparent, text `#737686` 12px uppercase

**Columns**:

| Column | Nội dung |
|---|---|
| Video Name | Thumbnail 48×48 (bg `#E2E8F0`, border-radius 4px, `<img src placeholder>`) + tên file 16px weight 600 + meta 10px `#737686` (`1080p • 2:45 min`) |
| Detected Activity | predicted_class của segment đầu tiên, có thể 2 dòng (16px weight 400) |
| Confidence | Mini bar: `div relative w-16 h-1.5 bg-[#E2E7FF] rounded-full` + `div absolute left-0 top-0 h-full rounded-full` fill theo severity + % text 16px weight 700 màu theo severity |
| Review Status | Badge border-radius 9999, text 10px weight 700 uppercase |
| Created Time | "Today, HH:MM" hoặc "DD/MM" — 16px weight 400 `#434655` |
| Action | Nút "View Investigation" → navigate `/videos/:video_id` |

**Confidence bar width**: `width: ${confidence * 100}%` — nhưng clamp vào 64px container  
**Row**: padding `32px`, border-top `1px rgba(195,198,215,0.20)`

**"Load more investigations"**:
- Text button centered, 14px `#505F76`
- Icon `<ChevronDown>` lucide bên phải
- Click → load thêm 5 items (offset += 5)
- Ẩn khi đã load hết

**Investigation Review Status** — tính từ worst segment:
```typescript
const getInvestigationStatus = (segments: Segment[]) => {
  if (segments.some(s => s.review_status === 'PENDING_REVIEW')) return 'HIGH ALERT'
  if (segments.some(s => s.review_status === 'CORRECTED')) return 'IN REVIEW'
  return 'VALIDATED'
}
```

**Data từ API `GET /api/dashboard/recent-investigations?limit=5&offset=0`**:
```typescript
interface InvestigationItem {
  video_id: string
  filename: string
  duration: number | null       // giây
  file_size: number | null      // bytes — để tính "1080p" thì cần thêm resolution field hoặc hard-code
  detected_activity: string     // predicted_class segment đầu tiên
  confidence: number            // confidence_score segment đầu tiên
  anomaly_score: number         // anomaly_score segment đầu tiên
  investigation_status: 'HIGH ALERT' | 'IN REVIEW' | 'VALIDATED'  // tính ở backend
  created_at: string
}
```

> **Note**: "1080p", "4K" trong Figma là hard-code demo. Backend không có resolution field — hiển thị `duration` dạng "X:XX min" là đủ.

---

## Filter State

```typescript
interface DashboardFilter {
  anomaly_class: string   // '' = All
  date_from: string       // 'YYYY-MM-DD' hoặc ''
  date_to: string         // 'YYYY-MM-DD' hoặc ''
}
```

Filter áp dụng cho: Recent Alerts + Top Detections + Recent Investigations  
Filter KHÔNG áp dụng cho: Stats cards, Anomaly Distribution, Recent Activity

---

## API Endpoints Summary

| Endpoint | Params | Dùng cho |
|---|---|---|
| `GET /api/dashboard/stats` | — | 4 summary cards |
| `GET /api/dashboard/distribution` | `class?, date_from?, date_to?` | Donut chart |
| `GET /api/dashboard/recent-alerts` | `class?, date_from?, date_to?, limit=10` | Recent Alerts table |
| `GET /api/dashboard/top-detections` | `class?, date_from?, date_to?` | Top Detections bar |
| `GET /api/dashboard/recent-investigations` | `class?, date_from?, date_to?, limit=5, offset=0` | Investigations table |
| `GET /api/dashboard/recent-activity` | `limit=5` | Recent Activity list |

---

## Component Files

```
frontend/src/pages/Dashboard.tsx          ← page chính
frontend/src/components/
  ├── dashboard/
  │   ├── StatsCard.tsx                   ← 1 card, nhận props
  │   ├── AnomalyDonut.tsx                ← PieChart + legend
  │   ├── RecentActivity.tsx              ← list items
  │   ├── RecentAlerts.tsx                ← table
  │   ├── TopDetections.tsx               ← BarChart horizontal
  │   ├── RecentInvestigations.tsx        ← table + load more
  │   └── DashboardFilter.tsx             ← filter panel (dropdown + date)
```

---

## Acceptance Criteria

- AC39: 4 summary cards hiển thị đúng số liệu từ DB, badge % hard-code
- AC40: Donut chart đúng màu theo CLASS_COLORS, center label đúng tổng, legend 2 cột
- AC41: Recent Activity 4–5 items, dot màu đúng theo type
- AC42: Recent Alerts 10 rows, Severity đúng từ anomaly_score, Status đúng từ review_status
- AC43: Top Detections bar nằm ngang, sort DESC, LabelList số bên phải
- AC44: Recent Investigations 5 rows mặc định, Load more +5, confidence bar màu đúng
- AC45: Filter (class + date) apply cho Alerts + Top Detections + Investigations
- AC46: Row click trong Alerts và Investigations → navigate `/videos/:video_id`
- AC47: "Upload New Video" → navigate `/`
- AC48: "Export Data", "View All" → toast "Coming soon"
