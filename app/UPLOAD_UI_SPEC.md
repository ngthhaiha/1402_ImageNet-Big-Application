# UPLOAD_UI_SPEC.md
# UI Specification — Upload Page

## Nguồn gốc
Export từ Figma (Google AI Studio / Stitch).
Dùng làm DESIGN REFERENCE — extract màu, spacing, layout.
Không copy code trực tiếp vào component.

---

## Design Tokens

### Colors
- Primary blue: `#004AC6`
- Background page: `#FAF8FF`
- Sidebar background: `#F2F3FF`
- Sidebar nav active bg: `#D0E1FB`
- Sidebar border: `#C3C6D7`
- Text primary: `#131B2E`
- Text secondary: `#434655`
- Info bar background: `#D3E4FE`
- Info bar border: `#B7C8E1`
- Info bar text: `#0B1C30`
- Card background: `#F2F3FF`
- Card border: `#C3C6D7`
- Button primary bg: `#004AC6`
- Button primary text: `white`
- Button cancel border: `#C3C6D7`
- Button cancel text: `#434655`
- Badge Ready: green (estimate `#D1FAE5` / `#059669`)
- Badge Invalid Format: red (estimate `#FEE2E2` / `#DC2626`)
- Drop zone border: dashed, `#C3C6D7`
- Drop zone background: white

### Typography (font: Inter)
- App name: 16px, weight 800
- Nav items inactive: 12px, weight 500, letterSpacing 0.6
- Nav item active: 12px, weight 700
- Page title: 24px, weight 700 (estimate)
- Breadcrumb: 14px, weight 400 (inactive) / weight 700 (active, màu #004AC6)
- Body text: 14px, weight 400
- Button primary: 16px, weight 600
- Button cancel: 16px, weight 600
- Info bar text: 14px, weight 400
- Card title: 16px, weight 600
- Card subtitle: 12px, weight 500, letterSpacing 0.6
- Table header: 12px, weight 600 (estimate uppercase)
- Table body: 14px, weight 400

### Spacing & Layout
- Sidebar width: 240px, fixed left, bg #F2F3FF
- Content padding: 32px
- Content max-width: 1200px
- Section gap: 32px
- Card border-radius: 12px
- Button border-radius: 8px
- Nav item border-radius: 8px
- Drop zone border-radius: 12px (estimate)
- Drop zone padding: 48px vertical (estimate)

---

## Component Breakdown

### 1. Sidebar (shared, dùng lại cho tất cả pages)
```
[Logo icon #004AC6] Video Anomaly Detection
───────────────────────────────
  Dashboard
▶ Upload Video        ← active: bg #D0E1FB, weight 700
  Queue Analyze
  Alerts
  Profile
───────────────────────────────
[+ New Investigation]  ← bg #004AC6, text white
───────────────────────────────
  Logout
```
- Active item: bg `#D0E1FB`, text weight 700
- Inactive item: transparent bg, text `#434655`, weight 500
- New Investigation button: full width, bg `#004AC6`, rounded 8px

### 2. Header bar (breadcrumb)
```
Cases  >  Upload Video     [Emergency Call btn] [bell] [settings] [?] [avatar]
```
- "Cases": text `#434655`, 14px
- "Upload Video": text `#004AC6`, 14px, weight 700, underline border-bottom `#004AC6`
- Right: action icons + avatar

### 3. Drop Zone
```
┌──────────────────────────────────────────┐  (dashed border)
│                                          │
│          [cloud upload icon]             │
│         Drop videos here                 │
│  Support for MP4, AVI, MOV up to 5 GB   │
│  Higher bitrates recommended...          │
│                                          │
│       [+ Browse Files]  ← primary btn    │
│                                          │
└──────────────────────────────────────────┘
```

### 4. Queue Table (sau khi chọn file)
Header: `Queue (N selected)` — bên phải: `Total: X.X B`

Columns: FILE NAME | SIZE | DURATION | STATUS | ACTION(trash icon)

Row states:
- Ready: status badge xanh lá
- Invalid Format: status badge đỏ, icon cảnh báo

### 5. Footer Bar
```
[ℹ info icon] Videos will be processed sequentially...    [Cancel] [Upload & Analyze →]
```
- Info bar: bg `#D3E4FE`, border `#B7C8E1`, border-radius 8px, padding 16px
- Cancel: outline button, border `#C3C6D7`
- Upload & Analyze: bg `#004AC6`, text white, có icon

### 6. Bottom Cards (3 cards ngang)
- End-to-End Encryption
- AI Optimization
- Retention Policy

Mỗi card: bg `#F2F3FF`, border `#C3C6D7`, border-radius 12px, padding 24px

---

## Mapping sang Task 8

| Figma element | React / logic |
|---|---|
| Drop zone | `onDrop`, `onDragOver` handlers + `<input type="file" multiple accept>` |
| "N selected" | `useState` đếm files trong queue |
| "Total: X MB" | Tính tổng `file.size` |
| Ready badge | File pass validate |
| Invalid Format badge | File fail validate (extension, size, duration) |
| Duration column | Đọc bằng `HTMLVideoElement.duration` trước khi upload |
| Progress bar (khi upload) | `axios onUploadProgress` callback |
| Upload & Analyze | `POST /api/videos/upload` multipart |
| Cancel | Clear queue state |
| Redirect sau upload | `navigate('/queue')` |

## Validate logic (từ REQUIREMENTS.md)
```typescript
const MAX_FILES = 3
const MAX_BATCH_MB = 300
const MAX_DURATION_SEC = 300  // 5 phút
const ALLOWED_EXT = ['.mp4', '.avi', '.mov']
```
