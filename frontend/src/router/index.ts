import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { public: true },
    },
    {
      path: '/',
      // 外壳不动：侧边栏 + 顶栏 + <router-view/>。收敛后只剩工作台一个子路由，
      // 但外壳必须留着 —— 它承载了顶栏的退出登录与全局错误兜底（onErrorCaptured）。
      component: () => import('@/components/layout/AppLayout.vue'),
      meta: { requiresAuth: true },
      children: [
        {
          path: '',
          name: 'dashboard',
          component: () => import('@/views/DashboardView.vue'),
        },
        {
          path: 'qa',
          name: 'qa',
          component: () => import('@/views/QAChatView.vue'),
          // 研报问答面向三个角色开放（后端 get_current_user 只要求登录）。
          // keep-alive 在 AppLayout 中按组件名 QAChatView 缓存，切走再回来不丢会话状态。
        },
        {
          path: 'risk',
          name: 'risk-review',
          component: () => import('@/views/RiskReviewView.vue'),
          // P6 首次用上这个判定：风控审核台只有风控与管理员能进。
          // 研究员（提交方）不该看到风控给它的批注草稿 —— 职责分离在界面上也成立。
          meta: { requiresRole: ['risk_control', 'admin'] },
        },
        {
          path: 'monitor',
          name: 'task-monitor',
          component: () => import('@/views/TaskMonitorView.vue'),
          // 任务监控只有管理员能进：这里展示的是失败任务的完整 traceback，
          // 相当于把「内部排障栈」暴露给了界面 —— 研究员用不上，风险上也不该让。
          meta: { requiresRole: ['admin'] },
        },
        {
          path: 'tasks/:taskId/report',
          name: 'report',
          component: () => import('@/views/ReportView.vue'),
          // 不加 requiresRole：发布后的研报是对内的公开产物，
          // 三个角色都可读 —— 与 /risk（只有风控与管理员）刻意相反。
        },
      ],
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/',
    },
  ],
})

router.beforeEach((to) => {
  const auth = useAuthStore()

  if (to.meta.public) {
    // 已登录还去 /login 就直接回工作台 —— 否则用户点「后退」会看到登录页
    if (auth.isLoggedIn && to.name === 'login') return { name: 'dashboard' }
    return true
  }

  if (!auth.isLoggedIn) return '/login'

  // requiresRole：P6 起由 /risk（风控审核台）真正用上。不在名单里的角色直接送回工作台 ——
  // 不跳 403 页面，因为用户点侧边栏时根本看不到那个入口，走到这里只可能是手敲 URL。
  const required = to.meta.requiresRole as string[] | undefined
  if (required && !required.includes(auth.role)) return '/'

  return true
})

// ⚠️ 路由守卫只是 UX，不是安全边界。真正的鉴权在后端 require_role()：
// 任何人打开 devtools 改一下 localStorage 里的 token 或 user.role 就能绕过这里，
// 所以「谁能做什么」必须由后端拦住，前端这层只负责别让用户白跑一趟。
// 前端这个判定如果坏了，后果是「审核台进不去」；后端那个坏了才是「谁都能签字」。
export default router
