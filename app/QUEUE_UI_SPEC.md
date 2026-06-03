# QUEUE_UI_SPEC.md
# UI Specification — Queue Analyze Page

## Nguồn gốc
Export từ Figma. Dùng làm DESIGN REFERENCE — extract màu, spacing, layout.
Không copy code trực tiếp vào component.

---

## Design Tokens (từ Queue page)

### Shared với Upload page (giống nhau — dùng từ UPLOAD_UI_SPEC.md)
- Primary blue: `#004AC6`
- Background page: `#FAF8FF`
- Sidebar bg: `#F2F3FF`
- Nav active bg: `#D0E1FB`
- Border: `#C3C6D7`
- Text primary: `#131B2E`
- Text secondary: `#434655`

### Queue-specific colors
- Card background: `white`
- Card border: `1px #C3C6D7`
- Summary card badge bg: `#DBE1FF`
- Summary card badge text DONE: `#004AC6`
- Summary card badge text ACTIVE: `#003EA8`
- Card left accent border (ACTIVE card): `4px #C3C6D7 solid` (left only)
- Progress bar track: `#E2E5F0` (estimate, not in code)
- Progress bar fill PROCESSING: `#004AC6`
- Progress bar fill COMPLETED: `#004AC6` (full)
- Status badge COMPLETED: green tones
- Status badge PROCESSING: blue/yellow tones
- Status badge WAITING: gray
- Status badge FAILED: red

### Typography — Queue specific
- Page title "Analysis Queue": 32px, weight 600
- Breadcrumb: 12px, weight 400/500
- Summary card number: 32px, weight 700
- Summary card label: 12px, weight 500
- Summary card badge: 12px, weight 700
- Refresh Queue button: 16px, weight 600
- Table header: estimate 12px, weight 600, uppercase
- Table body: 14px, weight 400–500

### Layout — Queue specific
- Content padding: 32px (top 64px để tránh header)
- Gap between sections: 32px
- Card border radius: 12px
- Card padding: 16px
- Summary cards: grid 4 cột ngang, equal width
- Active card có left border accent 4px (phân biệt với DONE card)
- Batch panel: full width, border radius 12px, background white

---

## Component Breakdown (từ Figma code)

### 1. Header Row
```
[Breadcrumb: Investigations > Analysis Queue]    [Refresh Queue button]
[Title: "Analysis Queue" — 32px bold]
[Subtitle: "Real-time AI telemetry..."]
```
- Breadcrumb: text "Investigations" + chevron icon + "Analysis Queue" (màu #004AC6)
- Refresh Queue: outline button (bg white, border #737686), có icon refresh + text

### 2. Active Batch Panel
```
┌─────────────────────────────────────────────────────┐
│ [batch icon]  ACTIVE BATCH badge    [completion %]  │
│ Batch name                          [X% large text] │
│ "Uploaded: date • time"                             │
│                                                     │
│ "X of Y Videos Processed"    Processing...          │
│ [████████░░░░░░░░░░░░] progress bar                 │
└─────────────────────────────────────────────────────┘
```
- Background: white, border radius 12px, border 1px #C3C6D7
- Badge "ACTIVE BATCH": bg #DBE1FF, text #004AC6, border radius 4px
- Completion %: 32px, weight 700, text #131B2E
- Progress bar: full width, height ~8px

### 3. Summary Cards (4 cards, grid ngang)
| Card | Badge text | Badge color | Left border |
|---|---|---|---|
| DONE | DONE | #DBE1FF / #004AC6 | none |
| ACTIVE | ACTIVE | #DBE1FF / #003EA8 | 4px #C3C6D7 (accent) |
| QUEUED | QUEUED | (estimate gray) | none |
| ERRORS | ERRORS | (estimate red) | none |

Mỗi card:
- bg white, border radius 12px, padding 16px
- Icon nhỏ góc trên trái (placeholder trong Figma)
- Badge text góc trên phải
- Số lớn (32px bold) + label nhỏ bên dưới

### 4. Queue Details Table

**Table header row**: "Queue Details" (title) + "Download Report" | "Cancel All" (actions)

**Columns**:
| Column | Width estimate | Nội dung |
|---|---|---|
| Video Name | ~30% | Thumbnail (ảnh nhỏ) + tên file |
| Status | ~15% | Badge màu theo status |
| Progress | ~20% | Progress bar + % text (khi PROCESSING) |
| Duration | ~10% | HH:MM:SS |
| Submitted | ~10% | HH:MM:SS |
| Action | ~15% | Nút theo status |

**Row styles**:
- Border bottom giữa các row: `1px #C3C6D7`
- Row FAILED: có thể có background đỏ nhạt (cần confirm)
- Row hover: bg highlight nhẹ

**Status badges trong table**:
- COMPLETED: bg xanh lá nhạt, text xanh đậm, dot xanh
- PROCESSING: bg xanh nhạt, text xanh, dot xanh (animate)
- WAITING: bg xám nhạt, text xám, dot xám
- FAILED: bg đỏ nhạt, text đỏ, dot đỏ
- PENDING_CONFIRM: bg cam nhạt, text cam

**Action buttons theo status**:
- COMPLETED → text link "View Detail" (màu #004AC6, bold)
- PROCESSING → outline button "Monitor"
- WAITING → disabled text "View Detail"
- PENDING_CONFIRM → primary button "Review" (bg #004AC6, text white)
- FAILED → outline button "Retry" (border đỏ, text đỏ)

**Progress bar trong table**:
- COMPLETED: full width, màu #004AC6
- PROCESSING: partial width tương ứng %, màu #004AC6
- WAITING: empty (bg only)
- FAILED: có text lỗi thay vì bar (vd: "Error: Codec Incompatible")

### 5. Pagination
```
Showing 1 to 4 of 10 results     [<] [1] [2] [3] [>]
```
- Text "Showing X to Y of Z results": 14px, text secondary
- Page buttons: 32px × 32px, border radius 4px
- Active page: bg #004AC6, text white
- Inactive: bg white, border #C3C6D7

---

## Figma Raw Code
[Xem file queue.txt trong uploads — không copy vào component]

---

## Mapping sang Task 9

| Figma element | React component / logic |
|---|---|
| Active Batch Panel | `BatchPanel` component, data từ `GET /api/batches/:id` |
| Completion % | `(done + pending_confirm) / total * 100` |
| 4 Summary Cards | `SummaryCards` component, đếm từ video list |
| Queue table | `QueueTable` component, data từ `GET /api/videos?batch_id=X` |
| Progress bar trong row | Map `progress_step` → % theo bảng trong REQUIREMENTS.md |
| Status badge | Dùng `StatusBadge` component đã có từ Task 7 |
| Action button | Conditional render theo `video.status` |
| Retry button | `POST /api/videos/:id/retry` |
| Pagination | Client-side, 10 rows/page |
| Auto-refresh | `setInterval` 5000ms, clear khi tất cả terminal status |
