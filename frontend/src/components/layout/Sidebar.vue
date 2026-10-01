<template>
  <div class="sidebar">
    <div class="logo">
      <span>📈 投研助手</span>
    </div>
    <nav class="nav-list">
      <RouterLink to="/" class="nav-item" :class="{ 'nav-item--active': isActive('/') }">
        <el-icon><DataAnalysis /></el-icon>
        <span>投研工作台</span>
      </RouterLink>
      <!-- 研报问答：问已发布的研报，三角色都可见（后端只要求登录）。 -->
      <RouterLink to="/qa" class="nav-item" :class="{ 'nav-item--active': isActive('/qa') }">
        <el-icon><ChatDotRound /></el-icon>
        <span>研报问答</span>
      </RouterLink>
      <!-- 风控审核台只对风控与管理员显示。研究员看不到这个入口 ——
           职责分离在界面上就该是「看不见」，而不是「点进去被拒」。
           与 router 上 meta.requiresRole 用的是同一份角色名单。 -->
      <RouterLink
        v-if="canReview"
        to="/risk"
        class="nav-item"
        :class="{ 'nav-item--active': isActive('/risk') }"
      >
        <el-icon><Stamp /></el-icon>
        <span>风控审核</span>
      </RouterLink>
    </nav>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { DataAnalysis, Stamp, ChatDotRound } from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const auth = useAuthStore()

const canReview = computed(() => auth.isRiskControl || auth.isAdmin)

// 仅用于高亮（纯视觉反馈，不参与导航逻辑）
function isActive(prefix: string) {
  // 根路由必须精确匹配。原来的前缀写法（startsWith('/' + '/')）恰好也能得到正确结果
  // —— 因为 '//' 不会匹配 '/risk' —— 但那靠的是一个巧合，多一个入口就未必还成立。
  if (prefix === '/') return route.path === '/'
  return route.path === prefix || route.path.startsWith(prefix + '/')
}
</script>

<style scoped>
.sidebar {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  border-bottom: 1px solid #ffffff1a;
  flex-shrink: 0;
}

.nav-list {
  display: flex;
  flex-direction: column;
  padding: 4px 0;
  flex: 1;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 13px 20px;
  color: #ffffffa6;
  text-decoration: none;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.2s, color 0.2s;
  user-select: none;
}

.nav-item:hover {
  background-color: #ffffff14;
  color: #fff;
}

.nav-item--active {
  background-color: #1677ff;
  color: #fff;
}

.nav-item .el-icon {
  font-size: 16px;
  flex-shrink: 0;
}
</style>
