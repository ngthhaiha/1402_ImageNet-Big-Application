import { useState } from 'react'
import { CircleHelp, PhoneCall, Settings } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { useAuth } from '../context/AuthContext'
import { NotificationBell } from './notifications/NotificationBell'

interface PageHeaderProps {
  pageName: string
}

export function PageHeader({ pageName }: PageHeaderProps) {
  const navigate = useNavigate()
  const { user } = useAuth()
  const [callState, setCallState] = useState<'closed' | 'confirm' | 'calling'>('closed')

  function closeCallDialog() {
    setCallState('closed')
  }

  function startDemoCall() {
    setCallState('calling')
  }

  return (
    <>
      <div className="page-header-sticky">
        <div className="page-header-inner">
          <header className="page-header">
            <nav className="page-breadcrumb" aria-label="Breadcrumb">
              <span>Cases</span>
              <span>&gt;</span>
              <span>{pageName}</span>
            </nav>
            <div className="page-header-actions" aria-label="Page actions">
              <button
                type="button"
                className="page-emergency-button"
                onClick={() => setCallState('confirm')}
              >
                <PhoneCall className="page-header-icon" aria-hidden="true" />
                Emergency Call
              </button>
              <NotificationBell />
              <button type="button" className="page-header-icon-button" aria-label="Settings">
                <Settings className="page-header-icon" aria-hidden="true" />
              </button>
              <button type="button" className="page-header-icon-button" aria-label="Help">
                <CircleHelp className="page-header-icon" aria-hidden="true" />
              </button>
              <div className="page-user-menu-wrapper">
                <button
                  type="button"
                  onClick={() => navigate('/profile')}
                  className="page-user-menu-button"
                  aria-label="Profile"
                >
                  {user?.username.charAt(0).toUpperCase() ?? 'U'}
                </button>
              </div>
            </div>
          </header>
        </div>
      </div>

      {callState !== 'closed' ? (
        <div className="emergency-modal-backdrop" role="presentation" onClick={closeCallDialog}>
          <div
            className="emergency-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="emergency-call-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="emergency-modal-icon">
              <PhoneCall aria-hidden="true" />
            </div>
            {callState === 'confirm' ? (
              <>
                <h2 id="emergency-call-title">Emergency Call</h2>
                <p>Gọi khẩn cấp đến 115 để báo sự cố cần hỗ trợ y tế.</p>
                <div className="emergency-modal-number">115</div>
                <div className="emergency-modal-actions">
                  <button
                    type="button"
                    className="emergency-modal-secondary"
                    onClick={closeCallDialog}
                  >
                    Hủy
                  </button>
                  <button
                    type="button"
                    className="emergency-modal-primary"
                    onClick={startDemoCall}
                  >
                    Gọi 115
                  </button>
                </div>
              </>
            ) : (
              <>
                <h2 id="emergency-call-title">Đang gọi 115</h2>
                <p>Demo call đang được mô phỏng. Không có cuộc gọi thật được thực hiện.</p>
                <div className="emergency-call-status">
                  <span aria-hidden="true" />
                  Connected demo line
                </div>
                <div className="emergency-modal-actions">
                  <button
                    type="button"
                    className="emergency-modal-primary"
                    onClick={closeCallDialog}
                  >
                    Kết thúc
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      ) : null}
    </>
  )
}
