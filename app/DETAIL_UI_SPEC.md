# DETAIL_UI_SPEC.md
# UI Specification — Video Detail Page (Task 10 + Task 11)

## Nguồn gốc
Export từ Figma + ảnh screenshot giao diện thực.
Dùng làm DESIGN REFERENCE — extract màu, layout, component structure.
Không copy code trực tiếp vào component.

---

## Design Tokens (Detail page — bổ sung thêm so với shared tokens)

### Colors mới (không có ở Upload/Queue)
- Anomaly badge bg: `#BA1A1A` (đỏ đậm)
- Anomaly badge text: `white`
- Anomaly badge dot: `white`, 8px, border-radius 9999
- Video overlay bg: `rgba(0,0,0,0.20)`
- Video controls bg: `rgba(0,0,0,0.60)`, backdropFilter blur(6px)
- Segment ID color: `#004AC6` (clickable blue)
- Predicted Activity "Fighting": `#BA1A1A` (đỏ đậm — màu anomaly)
- Confidence bar fill: `#004AC6`
- Confidence bar track: `#E2E5F0` (estimate)
- Status badge "Pending Review": bg `#F2F3FF`, text `#434655`, border radius 9999
- Status badge "VERIFIED AI": bg `#DBE1FF`, text `#004AC6`
- Timeline bar bg: `#E2E5F0` (xám nhạt = normal)
- Timeline anomaly highlight: `#BA1A1A` (đỏ đậm)
- Timeline normal legend: `#C3C6D7`
- Row selected bg: estimate `#F2F3FF`
- Label badge "FIGHTING": bg `#FEE2E2`, text `#BA1A1A` (đỏ nhạt nền, đỏ đậm text)
- Label badge "NORMAL": bg `#F2F3FF`, text `#434655`
- Review status "Pending Review": dot `#737686`, text `#434655`
- Review status "Logged": dot `#BA1A1A` (đỏ), text `#434655`
- Submit Feedback button: bg `#004AC6`, text white, full width, border-radius 8px
- Feedback panel border-top: `1px #C3C6D7`
- Archive Case button: outline `#737686`, text `#434655`, border-radius 12px
- Export Report button: bg `#004AC6`, text white, border-radius 12px

### Typography mới
- Page title "Video Investigation": 32px, weight 600, Inter
- Breadcrumb secondary: 14px, weight 400, color `#737686`, font Nimbus Sans
- Breadcrumb active "Segment Review": 14px, weight 500, color `#004AC6`
- Investigation Summary label: 12px, weight 500, color `#737686`
- Investigation Summary value: 14px, weight 600, color `#131B2E`
- Segment ID value: 14px, weight 600, color `#004AC6`
- Predicted Activity value: 14px, weight 700, color `#BA1A1A`
- Confidence %: 16px, weight 700
- "Feedback & Validation" title: 18px, weight 600
- "INVESTIGATOR COMMENTS" label: 11px, weight 600, uppercase, letterSpacing 0.6
- Submit Feedback: 16px, weight 600
- Anomaly overlay badge: 12px, weight 700, color white
- Timeline label "Analysis Timeline": 14px, weight 600
- Legend text: 12px, weight 400, color `#737686`
- Table header: 11px, weight 600, uppercase, color `#737686`, letterSpacing
- Table body time range: 14px, weight 500, mono-like

---

## Layout Structure

```
[Sidebar 240px fixed] | [Content area]
                        ├─ [Header: title + breadcrumb + Archive/Export buttons]
                        └─ [2-column body]
                             ├─ LEFT (~65%):
                             │   ├─ Video Player card
                             │   ├─ Analysis Timeline card
                             │   └─ Detected Segments Table card
                             └─ RIGHT (~35%):
                                 ├─ Investigation Summary card
                                 └─ Feedback & Validation card
```

Gap between 2 columns: 24px  
Gap between cards (vertical): 24px  
Card: bg white, border-radius 12px, border 1px `#C3C6D7`, box-shadow `0px 1px 2px rgba(0,0,0,0.05)`

---

## Component Breakdown

### 1. Header
```
Video Investigation                    [Archive Case]  [Export Report]
Analysis Queue / Segment Review
```
- "Analysis Queue": color `#737686`, clickable → `/queue`
- "/" separator: color `#737686`
- "Segment Review": color `#004AC6`
- Archive Case: outline button, border `#737686`, radius 12px, icon + text
- Export Report: bg `#004AC6`, radius 12px, icon (document) + text white

---

### 2. Video Player Card (cột trái, card 1)
```
┌─────────────────────────────────────────────┐
│ [● ANOMALY DETECTED: FIGHTING (94%)]        │  ← badge absolute top-left
│                                             │
│         [video frame / placeholder]         │
│                                             │
│ [play] [pause] [vol] 02:18 / 05:00 [full]  │  ← controls overlay bottom
└─────────────────────────────────────────────┘
```
- Card height: ~366px, overflow hidden
- Overlay badge: position absolute, top 17px, left 17px
  - bg `#BA1A1A`, border-radius 9999, padding `4px 12px`
  - dot: 8px circle white + text "ANOMALY DETECTED: {CLASS} ({CONFIDENCE}%)"
  - Chỉ hiện khi `currentTime` nằm trong range segment đang active
- Video controls overlay: position absolute bottom, opacity 0 → 1 khi hover
  - bg `rgba(0,0,0,0.60)`, blur, border-radius 12px
  - Icons: Play/Pause, Volume, timestamp `MM:SS / MM:SS`, Fullscreen

---

### 3. Analysis Timeline Card (cột trái, card 2)
```
┌──────────────────────────────────────────────────────────┐
│ ~ Analysis Timeline      □ Normal  ■ Detected Anomaly    │
│ ──────────────────────────────────────────────────────── │
│ [░░░░░░░░░░░░░░░▓▓░░░░░░░░░░░░░░░░░░░░░░░▓▓▓░░░░░░░░░] │
│ 00:00   01:00   02:00   03:00   04:00   05:00            │
└──────────────────────────────────────────────────────────┘
```
- Card: bg white, padding 24px
- Header: icon (trend line) + "Analysis Timeline" + legend bên phải
- Legend: `□ Normal` (bg `#C3C6D7`) + `■ Detected Anomaly` (bg `#BA1A1A`)
- Timeline bar: height ~24px, border-radius 4px, bg `#E2E5F0`
- Anomaly blocks: absolute positioned, bg `#BA1A1A`, height 100%
  - Width = `(end_time - start_time) / duration * 100%`
  - Left = `start_time / duration * 100%`
- Timestamp markers: đều nhau bên dưới bar, font mono 11px, color `#737686`
- Click block → video seek đến start_time

---

### 4. Detected Segments Table (cột trái, card 3)
```
┌──────────────────────────────────────────────────────────┐
│ Detected Segments                          [filter] [↓]  │
│ TIME RANGE    PREDICTED ACTIVITY  CONFIDENCE  REVIEW STATUS│
│ ─────────────────────────────────────────────────────────│
│ 02:14 - 02:25  [FIGHTING]          94%       ● Pending   │
│ 04:45 - 04:58  [FIGHTING]          81%       ● Pending   │
│ 00:12 - 00:45  [NORMAL]            12%       ● Logged    │
└──────────────────────────────────────────────────────────┘
```
- Header: "Detected Segments" (16px bold) + icon filter + icon download
- Table header: uppercase 11px, color `#737686`, letterSpacing
- Columns: TIME RANGE | PREDICTED ACTIVITY | CONFIDENCE | REVIEW STATUS
- Row click → highlight row + cập nhật Investigation Summary panel bên phải
- Row mặc định: chọn row đầu tiên khi load
- Label badge colors:
  - FIGHTING/anomaly: bg `#FEE2E2`, text `#BA1A1A`, border-radius 4px, padding `2px 8px`
  - NORMAL: bg `#F2F3FF`, text `#434655`
- Review Status:
  - "Pending Review": dot `#737686` + text
  - "Logged": dot `#BA1A1A` + text
  - "Label Correct": dot xanh lá + text
  - "Corrected": dot cam + text

---

### 5. Investigation Summary Card (cột phải, card 1)
```
┌──────────────────────────────────┐
│ Investigation Summary  [badge]   │
│ ─────────────────────────────── │
│ Video Name   PRK_NORTH_CAM04.mp4 │
│ Segment ID   #SEG-4421           │
│ Predicted    ● Fighting          │
│ Confidence   [████████░░]  94%   │
│ Timestamp    02:14 - 02:25       │
│ Status       [Pending Review]    │
└──────────────────────────────────┘
```
- Card: bg white, padding 24px, border-radius 12px
- Title: "Investigation Summary" (18px, weight 600) + badge góc phải
- Badge "VERIFIED AI": bg `#DBE1FF`, text `#004AC6`, 10px bold uppercase
- Badge "Pending Review" (status): bg `#F2F3FF`, border-radius 9999
- Label rows: left = label (12px, `#737686`), right = value (14px, `#131B2E`)
- Segment ID: color `#004AC6`
- Predicted Activity: dot `#BA1A1A` + text `#BA1A1A`, weight 700
- Confidence bar: height 6px, track `#E2E5F0`, fill `#004AC6`, border-radius 9999

---

### 6. Feedback & Validation Card (cột phải, card 2)

**STATE A — Form:**
```
┌──────────────────────────────────┐
│ Feedback & Validation            │
│ ─────────────────────────────── │
│ Is the detected anomaly          │
│ segment correct?                 │
│ [✓ Correct]  [✗ Incorrect]       │
│                                  │
│ Is the predicted activity        │
│ correct?                         │
│ [✓ Label Correct] [✎ Edit Label] │
│ (dropdown nếu Edit Label)        │
│ (textarea nếu Other)             │
│                                  │
│ INVESTIGATOR COMMENTS            │
│ [Describe findings...      ]     │
│                                  │
│ [      Submit Feedback     ]     │
└──────────────────────────────────┘
```
- Title: "Feedback & Validation" (18px, weight 600)
- Divider: `1px #C3C6D7` dưới title
- Question text: 14px, weight 400, `#131B2E`
- Nút Correct/Incorrect: outline, border-radius 8px, icon + text
  - Selected: bg `#004AC6`, text white
  - Unselected: bg white, border `#C3C6D7`, text `#434655`
- Nút Label Correct/Edit Label: tương tự
- "INVESTIGATOR COMMENTS": 11px uppercase, letterSpacing, color `#737686`
- Textarea: border `#C3C6D7`, border-radius 8px, placeholder màu `#B0B3C1`
- Submit Feedback: full width, bg `#004AC6`, text white, border-radius 8px, 16px weight 600
  - disabled state: opacity 0.5

**STATE B — Feedback Detail:**
```
┌──────────────────────────────────┐
│ Feedback Detail                  │
│ ─────────────────────────────── │
│ Segment Detect   ✓ Correct       │
│ Verified Label   Fighting        │
│ Comments         "Two individuals│
│                   near B3"       │
│ Submitted At     15/01/2025 10:45│
│                                  │
│ [      ✎ Edit Feedback      ]    │
└──────────────────────────────────┘
```
- Edit Feedback: outline button, full width, icon pencil + text

---

## Mapping → React Components

| UI Element | Component / file |
|---|---|
| Toàn bộ page | `VideoDetail.tsx` |
| Cột trái | `VideoDetailLeft.tsx` (hoặc inline trong VideoDetail) |
| Video Player + badge | `VideoPlayer.tsx` — `<video>` HTML5 |
| Anomaly badge overlay | Conditional render dựa trên `currentTime` |
| Analysis Timeline | `AnomalyTimeline.tsx` — div positioning |
| Detected Segments Table | `SegmentsTable.tsx` |
| Label badge | `ActivityBadge.tsx` (màu theo class) |
| Review status | `ReviewStatusBadge.tsx` |
| Cột phải | `VideoDetailRight.tsx` |
| Investigation Summary | `InvestigationPanel.tsx` |
| Confidence bar | Tailwind div width % |
| Feedback Form (State A) | `FeedbackForm.tsx` |
| Feedback Detail (State B) | `FeedbackDetail.tsx` |
| Controlled bởi | `feedbackState: 'form' \| 'detail'` trong VideoDetail |

## API Calls (Task 10 + 11)

| Action | Endpoint |
|---|---|
| Load video + segments | `GET /api/videos/:id` |
| Polling khi PROCESSING | `GET /api/videos/:id` mỗi 3s |
| Submit Feedback | `POST /api/segments/:id/feedback` |
| Export Report | `GET /api/videos/:id/export` (trigger download) |

## State cần quản lý trong VideoDetail.tsx

```typescript
const [video, setVideo] = useState<Video | null>(null)
const [segments, setSegments] = useState<Segment[]>([])
const [selectedSegmentId, setSelectedSegmentId] = useState<number | null>(null)
const [currentTime, setCurrentTime] = useState(0)          // từ video player
const [feedbackState, setFeedbackState] = useState<'form' | 'detail'>('form')
const videoRef = useRef<HTMLVideoElement>(null)
```

- `selectedSegmentId` → drive Investigation Summary + Feedback panel
- `currentTime` → drive anomaly overlay badge visibility
- Khi click row: `setSelectedSegmentId(segment.id)` + `videoRef.current.currentTime = segment.start_time`
- Khi click timeline block: tương tự
