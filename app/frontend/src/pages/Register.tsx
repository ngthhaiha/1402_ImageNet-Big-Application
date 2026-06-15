import axios from 'axios'
import { useState } from 'react'
import { Shield } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'

import { useAuth } from '../context/AuthContext'

export function Register() {
  const navigate = useNavigate()
  const { register } = useAuth()
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const validate = (): string | null => {
    if (password.length < 6) {
      return 'Password must be at least 6 characters'
    }
    if (password !== confirmPassword) {
      return 'Passwords do not match'
    }
    if (!email.includes('@')) {
      return 'Invalid email format'
    }
    return null
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    const validationError = validate()
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    setIsLoading(true)
    try {
      await register(username, email, password)
      navigate('/', { replace: true })
    } catch (registerError) {
      if (axios.isAxiosError(registerError)) {
        const message = registerError.response?.data?.message
        setError(typeof message === 'string' ? message : 'Registration failed')
      } else {
        setError('Registration failed')
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#FAF8FF] px-4">
      <div className="w-full max-w-[400px] bg-white rounded-2xl border border-[#C3C6D7] p-10">
        <div className="flex items-center gap-3 justify-center mb-8">
          <div className="w-8 h-8 bg-[#004AC6] rounded-lg flex items-center justify-center">
            <Shield size={16} className="text-white" />
          </div>
          <span className="text-base font-extrabold text-[#131B2E]">
            Video Anomaly Detection
          </span>
        </div>

        <h1 className="text-2xl font-semibold text-[#131B2E] mb-6 text-center">
          Create an account
        </h1>

        {error ? (
          <div className="bg-[#FEE2E2] text-[#BA1A1A] text-sm rounded-lg px-4 py-3 mb-4">
            {error}
          </div>
        ) : null}

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label className="text-sm font-medium text-[#434655] mb-1 block">
              Username
            </label>
            <input
              type="text"
              className="w-full border border-[#C3C6D7] rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-[#004AC6]"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </div>

          <div>
            <label className="text-sm font-medium text-[#434655] mb-1 block">
              Email
            </label>
            <input
              type="email"
              className="w-full border border-[#C3C6D7] rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-[#004AC6]"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>

          <div>
            <label className="text-sm font-medium text-[#434655] mb-1 block">
              Password
            </label>
            <input
              type="password"
              className="w-full border border-[#C3C6D7] rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-[#004AC6]"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </div>

          <div>
            <label className="text-sm font-medium text-[#434655] mb-1 block">
              Confirm Password
            </label>
            <input
              type="password"
              className="w-full border border-[#C3C6D7] rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-[#004AC6]"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-[#004AC6] text-white rounded-xl py-2.5 font-semibold mt-2 disabled:opacity-50 hover:bg-[#003a9e] transition-colors"
          >
            {isLoading ? 'Registering...' : 'Register'}
          </button>
        </form>

        <p className="text-center text-sm text-[#737686] mt-6">
          Already have an account?{' '}
          <Link to="/login" className="text-[#004AC6] font-medium hover:underline">
            Login
          </Link>
        </p>
      </div>
    </div>
  )
}
