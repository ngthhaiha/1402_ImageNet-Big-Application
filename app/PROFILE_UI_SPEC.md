# PROFILE_UI_SPEC.md
# UI Specification — Profile Page (Task 17)

## Nguồn gốc
Export từ Figma + ảnh screenshot.
Dùng làm DESIGN REFERENCE — không copy code trực tiếp.

---

## Design Tokens (Profile-specific)

### Colors mới
- Avatar border: `2px #2563EB solid`
- Camera badge bg: `#004AC6`, border-radius 12px
- Name text: `#131B2E`, 32px, weight 600
- Role text: `#505F76`, 20px, weight 600
- Badge ID: bg `rgba(37,99,235,0.10)`, text `#004AC6`
- Badge Location: bg `rgba(208,225,251,0.20)`, text `#54647A`
- Badge Shift: bg `#FFDBCD`, text `#7D2D00` (cam đậm)
- Export Activity button text: `#505F76`
- Stats icon bg: `rgba(0,74,198,0.10)`, icon `#004AC6`
- Stats sub-text: `#505F76`, 12px, weight 500
- Stats value: `#131B2E`, 32px–40px, weight 700
- Stats label: `#737686`, 12px, uppercase, letterSpacing
- Activity icon UPLOAD: bg `#004AC6`
- Activity icon REVIEW: bg `#737686`
- Activity icon FLAG: bg `#F59E0B`
- Activity title link: `#004AC6`
- Activity timestamp: `#737686`, 12px
- Activity description: `#505F76`, 14px
- Settings row border: `1px #C3C6D7`
- Settings chevron: `#C3C6D7`
- Toggle ON: `#004AC6`
- Toggle OFF: `#C3C6D7`
- "RECOMMENDED" badge: bg `#FEE2E2`, text `#BA1A1A`, 10px bold uppercase
- Log Out button text + border: `#BA1A1A`
- Card border-radius: 16px (Profile cards lớn hơn các page khác — 12px)
- Card padding: 24px–32px

---

## Layout

```
[Sidebar 240px] | [Content 32px padding]
                  ├─ [Header Card — full width]
                  └─ [2 columns gap-32px]
                       ├─ LEFT ~65%:
                       │   ├─ Stats Cards (3 ngang)
                       │   └─ Recent Activity Card
                       └─ RIGHT ~35%:
                           ├─ Account Settings Card
                           ├─ Security Card
                           ├─ Notifications Card
                           └─ Log Out Button
```

---

## Component Breakdown

### 1. Header Card (full width)

```
┌────────────────────────────────────────────────────────────┐
│  [avatar 128px]  Officer James Miller          [Edit Profile]  │
│  [cam badge]     Senior Security Investigator  [Export Activity]│
│                  [ID: #SOC-882] [Sector A] [14h Shift]         │
└────────────────────────────────────────────────────────────┘
```
- Card: bg white, border-radius 16px, padding 32px, border 1px `#C3C6D7`
- Avatar: 128×128, border-radius 16px, border `2px #2563EB`
- Camera badge: absolute bottom-right của avatar, bg `#004AC6`, padding 8px, border-radius 12px, icon Camera white
- Edit Profile: bg `#004AC6`, text white, border-radius 12px, padding `16px`, icon Pencil
- Export Activity: bg white, outline `#737686`, text `#505F76`, border-radius 12px, icon Share

### 2. Stats Cards (3 cards ngang, equal flex)

Mỗi card:
```
┌──────────────────────┐
│ [icon bg blue]  +12% │  ← icon + sub-text (hard-code)
│                      │
│ VIDEOS UPLOADED      │  ← label uppercase
│ 142                  │  ← value từ API
└──────────────────────┘
```
- bg white, border-radius 16px, padding `24px 24px 40px`
- Icon container: 40×40, bg `rgba(0,74,198,0.10)`, border-radius 12px, icon `#004AC6`
- Sub-text (trend): right-aligned, 12px, `#505F76`
- Label: 12px uppercase, letterSpacing, `#737686`
- Value: 40px, weight 700, `#131B2E`

**Data mapping**:
```typescript
// GET /api/profile/stats response
{
  videos_uploaded: number,    // COUNT(*) FROM videos
  cases_reviewed: number,     // COUNT(*) FROM anomaly_segments WHERE feedback_submitted_at IS NOT NULL
  feedback_submitted: number  // same as cases_reviewed hoặc tính riêng
}
```

### 3. Recent Activity Card

```
┌────────────────────────────────────────────────────────────┐
│ ~ Recent Activity                                          │
│ ─────────────────────────────────────────────────────────│
│ [●blue] Uploaded Mall_Entrance_01.mp4           2h ago   │
│         Auto-tagging initiated for person detection...   │
│         ┌──────────────────────────────────┐             │
│         │ [thumb] Sector A4 - Zone 2       │             │
│         │         4K UHD • 1.2 GB          │             │
│         └──────────────────────────────────┘             │
│                                                           │
│ [●gray] Completed review for Case #INV-8821    5h ago   │
│         Evidence finalized. Final status: [CRITICAL]    │
│                                                           │
│ [▲amber] Flagged suspicious activity in Parking...      │
│                                                           │
│ ─────────────────────────────────────────────────────── │
│                [View Full Activity History]              │
└────────────────────────────────────────────────────────────┘
```

**Activity item structure**:
- Icon: 40×40 circle, màu theo type (UPLOAD=blue, REVIEW=gray, FLAG=amber)
- Title: 14px, weight 600. Phần link (filename/case): color `#004AC6`, underline hover
- Description: 14px, `#505F76`
- Timestamp: 12px, `#737686`, right-aligned
- UPLOAD type: có thể có sub-card thumbnail (optional, chỉ nếu có thumbnail)
- REVIEW_COMPLETE: có thể có status badge (hard-code "CRITICAL" cho demo)
- Click title/link → navigate `/videos/:video_id` (nếu `video_id` có trong activity)

**Timestamp format**:
```typescript
// relative time
const now = new Date()
const diff = now - new Date(created_at)
if (diff < 3600000) return `${Math.floor(diff/60000)}m ago`
if (diff < 86400000) return `${Math.floor(diff/3600000)}h ago`
if (diff < 172800000) return 'Yesterday'
return format(new Date(created_at), 'dd/MM/yyyy')
```

### 4. Account Settings Card (cột phải)

```
┌──────────────────────────────────┐
│ 👥 Account Settings              │
│ ───────────────────────────────│
│ ✉ Email Address           >     │
│   j.miller@ssis.hq.com          │
│ ───────────────────────────────│
│ 🌐 Language & Region       >    │
│   English (UK) • UTC +0         │
└──────────────────────────────────┘
```
- bg white, border-radius 16px, padding 24px, border 1px `#C3C6D7`
- Title: 16px, weight 600, icon Users
- Rows: padding 16px, border-bottom `1px #C3C6D7`, chevron `#C3C6D7` right
- Click row → toast "Coming soon"
- Data: **hard-code** (demo, không có auth)

### 5. Security Card

```
┌──────────────────────────────────┐
│ 🛡 Security                      │
│ ───────────────────────────────│
│ 🕐 Change Password    [90 days] >│
│ ☑ 2FA Verification              │
│   [RECOMMENDED]      [toggle ON] │
└──────────────────────────────────┘
```
- "90 days ago": badge outline, text `#737686`, 12px
- "RECOMMENDED": badge bg `#FEE2E2`, text `#BA1A1A`, 10px uppercase bold
- Toggle: UI only, không lưu state
- Click: toast "Coming soon"

### 6. Notifications Card

```
┌──────────────────────────────────┐
│ 🔔 Notifications                 │
│ ───────────────────────────────│
│ Critical Alerts         [ON ●]   │
│ Immediate push to mobile & email │
│ ───────────────────────────────│
│ Case Updates            [● OFF]  │
│ Daily digest of reviewed materials│
│ ───────────────────────────────│
│ Login History           [ON ●]   │
│ Notify on new device sign-in    │
└──────────────────────────────────┘
```
- Toggle ON: bg `#004AC6`, circle white bên phải
- Toggle OFF: bg `#C3C6D7`, circle white bên trái
- State lưu vào **localStorage** key `notification_prefs`:
  ```typescript
  {
    critical_alerts: true,   // default
    case_updates: false,     // default
    login_history: true      // default
  }
  ```

### 7. Log Out Button (cuối cột phải)

```
[→ Log Out from Session]
```
- Full width, bg white, border `1px #BA1A1A`, text `#BA1A1A`, border-radius 16px, padding 16px
- Icon LogOut màu `#BA1A1A`
- Click: toast "Coming soon"

---

## API Mapping

| Component | API |
|---|---|
| Stats cards | `GET /api/profile/stats` |
| Recent Activity | `GET /api/profile/activity` |
| Account Settings | Hard-code |
| Security | Hard-code |
| Notifications | localStorage |
| Edit Profile | Toast "Coming soon" |
| Export Activity | Toast "Coming soon" |
| Log Out | Toast "Coming soon" |

## Activity Log ghi từ Backend

| Trigger | File ghi | Code cần thêm |
|---|---|---|
| Upload thành công | `main.py` route upload | Sau khi tạo Video record |
| Worker xong PENDING_CONFIRM | `worker.py` | Sau bước 6 (update PENDING_CONFIRM), nếu có segment score > 0.8 → ghi FLAG |
| User submit feedback lần đầu | `main.py` route feedback | Kiểm tra `feedback_submitted_at` trước đó là null → ghi REVIEW_COMPLETE |
