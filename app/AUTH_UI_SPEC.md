# AUTH_UI_SPEC.md
# UI Specification — Authentication (Task 22-24)

## Layout — Login & Register

Không dùng Sidebar/Header. Layout centered, full-screen.

```
┌─────────────────────────────────────────┐
│              [App logo + name]            │
│                                          │
│         ┌──────────────────┐            │
│         │   Welcome back     │            │
│         │                    │            │
│         │  Username/Email    │            │
│         │  [____________]    │            │
│         │                    │            │
│         │  Password          │            │
│         │  [____________]    │            │
│         │                    │            │
│         │  [    Login    ]   │            │
│         │                    │            │
│         │  Don't have an     │            │
│         │  account? Register │            │
│         └──────────────────┘            │
└─────────────────────────────────────────┘
```

bg: `#FAF8FF`  
Card: `bg white`, `border-radius 16px`, `border 1px #C3C6D7`, `padding 40px`, `max-width 400px`, centered cả màn hình (`flex items-center justify-center min-h-screen`)

Logo: dùng icon shield (giống sidebar) + "Video Anomaly Detection", 16px weight 800

---

## Login.tsx

```tsx
<div className="min-h-screen flex items-center justify-center bg-[#FAF8FF]">
  <div className="w-full max-w-[400px] bg-white rounded-2xl border border-[#C3C6D7] p-10">
    {/* Logo */}
    <div className="flex items-center gap-3 justify-center mb-8">
      <div className="w-8 h-8 bg-[#004AC6] rounded-lg flex items-center justify-center">
        <Shield size={16} className="text-white" />
      </div>
      <span className="text-base font-extrabold text-[#131B2E]">Video Anomaly Detection</span>
    </div>

    <h1 className="text-2xl font-semibold text-[#131B2E] mb-6 text-center">
      Welcome back
    </h1>

    {error && (
      <div className="bg-[#FEE2E2] text-[#BA1A1A] text-sm rounded-lg px-4 py-3 mb-4">
        {error}
      </div>
    )}

    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div>
        <label className="text-sm font-medium text-[#434655] mb-1 block">
          Username or Email
        </label>
        <input
          type="text"
          className="w-full border border-[#C3C6D7] rounded-xl px-4 py-2.5
                     focus:outline-none focus:ring-2 focus:ring-[#004AC6]"
          value={usernameOrEmail}
          onChange={e => setUsernameOrEmail(e.target.value)}
          required
        />
      </div>

      <div>
        <label className="text-sm font-medium text-[#434655] mb-1 block">
          Password
        </label>
        <input
          type="password"
          className="w-full border border-[#C3C6D7] rounded-xl px-4 py-2.5
                     focus:outline-none focus:ring-2 focus:ring-[#004AC6]"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-[#004AC6] text-white rounded-xl py-2.5 font-semibold
                   mt-2 disabled:opacity-50 hover:bg-[#003a9e] transition-colors"
      >
        {isLoading ? 'Logging in...' : 'Login'}
      </button>
    </form>

    <p className="text-center text-sm text-[#737686] mt-6">
      Don't have an account?{' '}
      <Link to="/register" className="text-[#004AC6] font-medium hover:underline">
        Register
      </Link>
    </p>
  </div>
</div>
```

**Submit handler**:
```typescript
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault()
  setError(null)
  setIsLoading(true)
  try {
    await login(usernameOrEmail, password)  // từ AuthContext
    navigate('/', { replace: true })
  } catch (err) {
    setError('Invalid username or password')
  } finally {
    setIsLoading(false)
  }
}
```

---

## Register.tsx

Cấu trúc tương tự Login, thêm 2 field (Email, Confirm Password):

```tsx
<h1 className="text-2xl font-semibold text-[#131B2E] mb-6 text-center">
  Create an account
</h1>

{/* Username */}
{/* Email — type="email" */}
{/* Password */}
{/* Confirm Password */}

<button>Register</button>

<p className="text-center text-sm text-[#737686] mt-6">
  Already have an account?{' '}
  <Link to="/login" className="text-[#004AC6] font-medium hover:underline">
    Login
  </Link>
</p>
```

**Frontend validation trước khi gọi API**:
```typescript
const validate = (): string | null => {
  if (password.length < 6) return 'Password must be at least 6 characters'
  if (password !== confirmPassword) return 'Passwords do not match'
  if (!email.includes('@')) return 'Invalid email format'
  return null
}

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault()
  const validationError = validate()
  if (validationError) { setError(validationError); return }

  setError(null)
  setIsLoading(true)
  try {
    await register(username, email, password)  // từ AuthContext
    navigate('/', { replace: true })
  } catch (err: any) {
    setError(err.response?.data?.message || 'Registration failed')
  } finally {
    setIsLoading(false)
  }
}
```

---

## AuthContext.tsx

```typescript
interface User {
  id: number
  username: string
  email: string
}

interface AuthContextValue {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (usernameOrEmail: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem('token')
  )

  // Khi mount, nếu có token → fetch /api/auth/me để load user
  useEffect(() => {
    if (token) {
      getCurrentUser()
        .then(setUser)
        .catch(() => {
          localStorage.removeItem('token')
          setToken(null)
        })
    }
  }, [token])

  const login = async (usernameOrEmail: string, password: string) => {
    const data = await apiLogin(usernameOrEmail, password)
    localStorage.setItem('token', data.token)
    setToken(data.token)
    setUser(data.user)
  }

  const register = async (username: string, email: string, password: string) => {
    const data = await apiRegister(username, email, password)
    localStorage.setItem('token', data.token)
    setToken(data.token)
    setUser(data.user)
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{
      user, token, isAuthenticated: !!token, login, register, logout
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
```

---

## ProtectedRoute.tsx

```tsx
import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}
```

---

## Axios Interceptors (api.ts)

```typescript
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
})

// Request interceptor: thêm Bearer token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor: 401 → logout + redirect
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api
```

---

## App.tsx routing

```tsx
function App() {
  return (
    <AuthProvider>
      <NotificationProvider>
        <Router>
          <Routes>
            {/* Public routes — không Sidebar/Header */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* Protected routes — có Sidebar/Header */}
            <Route path="/*" element={
              <ProtectedRoute>
                <AppLayout />
              </ProtectedRoute>
            } />
          </Routes>
        </Router>
      </NotificationProvider>
    </AuthProvider>
  )
}

// AppLayout chứa Sidebar + Header + nested routes
function AppLayout() {
  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1">
        <Header />
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/queue" element={<Queue />} />
          <Route path="/videos/:id" element={<VideoDetail />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/profile" element={<Profile />} />
        </Routes>
        <NotificationStack />
      </div>
    </div>
  )
}
```

---

## Backend: auth.py

```python
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    from models import User
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user
```

---

## Backend: routers/auth.py

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import User
from schemas import RegisterRequest, LoginRequest, AuthResponse, UserResponse
from auth import hash_password, verify_password, create_access_token, get_current_user
from datetime import datetime

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(400, "Username already exists")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, "Email already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        created_at=datetime.utcnow().isoformat(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"user_id": user.id, "username": user.username})
    return AuthResponse(
        token=token,
        user=UserResponse(id=user.id, username=user.username, email=user.email)
    )


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = (
        db.query(User)
        .filter(
            (User.username == payload.username_or_email)
            | (User.email == payload.username_or_email)
        )
        .first()
    )
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")

    token = create_access_token({"user_id": user.id, "username": user.username})
    return AuthResponse(
        token=token,
        user=UserResponse(id=user.id, username=user.username, email=user.email)
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email
    )
```

---

## Schemas

```python
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username_or_email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        from_attributes = True

class AuthResponse(BaseModel):
    token: str
    user: UserResponse
```

---

## Header — hiển thị user (sau khi login)

Trong Header component, thay avatar placeholder bằng:
```tsx
const { user, logout } = useAuth()

<div className="relative">
  <button onClick={() => setShowUserMenu(prev => !prev)}
    className="w-8 h-8 bg-[#D0E1FB] rounded-full flex items-center justify-center
               text-sm font-bold text-[#004AC6]">
    {user?.username.charAt(0).toUpperCase()}
  </button>

  {showUserMenu && (
    <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-xl
                    shadow-lg border border-[#C3C6D7] p-2 z-50">
      <div className="px-3 py-2 text-sm font-medium text-[#131B2E] border-b border-[#C3C6D7]">
        {user?.username}
      </div>
      <button
        onClick={logout}
        className="w-full text-left px-3 py-2 text-sm text-[#BA1A1A]
                   hover:bg-[#FEE2E2] rounded-lg mt-1"
      >
        Logout
      </button>
    </div>
  )}
</div>
```

---

## Sidebar Logout button

Tìm nút "Logout" hiện có trong Sidebar (UI đã có, chỉ chưa có logic):
```tsx
const { logout } = useAuth()
const navigate = useNavigate()

const handleLogout = () => {
  logout()
  navigate('/login', { replace: true })
}

// onClick={handleLogout} vào nút Logout hiện có
```

---

## Profile "Log Out from Session" button

Thay `onClick={() => toast('Coming soon')}` bằng cùng `handleLogout` như trên.
```

---

## Acceptance Checklist

```
Backend:
- [ ] Bảng users tồn tại
- [ ] POST /api/auth/register tạo user, hash password, trả token
- [ ] Register username/email đã tồn tại → 400 error
- [ ] POST /api/auth/login đúng credential → trả token
- [ ] POST /api/auth/login sai → 401
- [ ] GET /api/auth/me với token hợp lệ → trả user info
- [ ] Tất cả routes khác (videos, dashboard, alerts...) yêu cầu Bearer token
- [ ] Gọi API không có token → 401
- [ ] /uploads/* KHÔNG yêu cầu token (video vẫn play được)

Frontend:
- [ ] /login và /register không có Sidebar/Header
- [ ] Register validation: password length, confirm match, email format
- [ ] Register thành công → auto login → redirect /
- [ ] Login thành công → redirect /
- [ ] Login sai → hiện lỗi rõ ràng
- [ ] Truy cập / khi chưa login → redirect /login
- [ ] Token invalid/expired → API 401 → tự redirect /login
- [ ] Logout (Sidebar) → xóa token → redirect /login
- [ ] Logout (Profile) → tương tự
- [ ] Header hiển thị username, dropdown có Logout
- [ ] npm run build không có TypeScript errors
```
