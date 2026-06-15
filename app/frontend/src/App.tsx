import { Navigate, Route, Routes } from 'react-router-dom'

import { NotificationStack } from './components/notifications/NotificationStack'
import { ProtectedRoute } from './components/ProtectedRoute'
import { Sidebar } from './components/Sidebar'
import { AuthProvider } from './context/AuthContext'
import { NotificationProvider } from './context/NotificationContext'
import { Alerts } from './pages/Alerts'
import { Dashboard } from './pages/Dashboard'
import { Login } from './pages/Login'
import { Profile } from './pages/Profile'
import { Queue } from './pages/Queue'
import { Register } from './pages/Register'
import { Upload } from './pages/Upload'
import { VideoDetail } from './pages/VideoDetail'

function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <AppLayout />
            </ProtectedRoute>
          }
        />
      </Routes>
    </AuthProvider>
  )
}

function AppLayout() {
  return (
    <NotificationProvider>
      <div className="min-h-screen bg-slate-100 text-slate-950">
        <div className="flex min-h-screen flex-col lg:flex-row">
          <Sidebar />
          <main className="min-w-0 flex-1">
            <Routes>
              <Route path="/" element={<Upload />} />
              <Route path="/queue" element={<Queue />} />
              <Route path="/videos/:id" element={<VideoDetail />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/profile" element={<Profile />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
          <NotificationStack />
        </div>
      </div>
    </NotificationProvider>
  )
}

export default App
