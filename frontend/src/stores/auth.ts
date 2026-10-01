import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// 角色取值与后端 users.role 的 CHECK 约束一一对应（见 scripts/init_db.sql）。
// 这三个角色同时是后端 require_role() 的入参，改这里必须同步改那边。
interface UserInfo {
  userId: string
  role: 'admin' | 'researcher' | 'risk_control'
  tenantId: string
  username?: string
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(localStorage.getItem('research-agent-token'))
  const user = ref<UserInfo | null>((() => {
    try {
      return JSON.parse(localStorage.getItem('research-agent-user') ?? 'null')
    } catch {
      return null
    }
  })())

  const role = computed(() => user.value?.role ?? '')
  const isLoggedIn = computed(() => !!token.value)
  const isResearcher = computed(() => role.value === 'researcher')
  const isRiskControl = computed(() => role.value === 'risk_control')
  const isAdmin = computed(() => role.value === 'admin')

  function login(accessToken: string, userInfo: UserInfo) {
    token.value = accessToken
    user.value = userInfo
    localStorage.setItem('research-agent-token', accessToken)
    localStorage.setItem('research-agent-user', JSON.stringify(userInfo))
  }

  function logout() {
    token.value = null
    user.value = null
    localStorage.removeItem('research-agent-token')
    localStorage.removeItem('research-agent-user')
  }

  return { token, user, role, isLoggedIn, isResearcher, isRiskControl, isAdmin, login, logout }
})
