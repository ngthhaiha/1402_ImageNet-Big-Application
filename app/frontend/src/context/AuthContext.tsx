import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

import { apiLogin, apiRegister, getCurrentUser } from '../api/api'
import type { User } from '../types/types'

interface AuthContextValue {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (usernameOrEmail: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('token'))

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
    <AuthContext.Provider
      value={{ user, token, isAuthenticated: !!token, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}
