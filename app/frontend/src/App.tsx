import { Navigate, Route, Routes } from 'react-router-dom'

import { Sidebar } from './components/Sidebar'
import { Alerts } from './pages/Alerts'
import { Dashboard } from './pages/Dashboard'
import { Profile } from './pages/Profile'
import { Queue } from './pages/Queue'
import { Upload } from './pages/Upload'
import { VideoDetail } from './pages/VideoDetail'

function App() {
  return (
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
      </div>
    </div>
  )
}

export default App
