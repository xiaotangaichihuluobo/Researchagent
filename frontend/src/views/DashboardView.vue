<!-- frontend/src/views/DashboardView.vue -->
<!-- 投研工作台：提交任务 → 轮询进度 → 展示阶段时间线。 -->
<template>
  <div class="dashboard">
    <el-card class="panel">
      <template #header>
        <span>发起研究</span>
      </template>

      <!-- 后端 POST /research/tasks 刻意只放行 researcher 与 admin（风控不得发起
           研究 —— 自己提案自己审，签字那道闸门就形同虚设）。这里提前说清楚，
           免得风控账号点下去只拿到一个 403。真正的边界仍在后端 require_role()。 -->
      <el-alert
        v-if="!canSubmit"
        type="info"
        :closable="false"
        show-icon
        title="当前角色不能发起研究"
        description="只有研究员与管理员可以发起研究任务；风控角色负责审核，不参与提案。"
      />

      <el-form v-else inline @submit.prevent>
        <el-form-item label="研究标的">
          <el-select v-model="selectedCode" placeholder="选择标的" style="width: 260px">
            <el-option
              v-for="c in companies"
              :key="c.code"
              :label="`${c.name}（${c.code}）`"
              :value="c.code"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="submitting"
            :disabled="!selectedCode"
            @click="handleSubmit"
          >
            发起研究
          </el-button>
        </el-form-item>
      </el-form>

      <el-alert
        v-if="submitError"
        type="error"
        :title="submitError"
        :closable="false"
        show-icon
      />
    </el-card>

    <el-card v-if="task" class="panel">
      <template #header>
        <div class="panel-header">
          <span>研究任务 {{ task.task_id.slice(0, 8) }}</span>
          <el-tag :type="statusTagType">{{ STATUS_LABELS[task.status] ?? task.status }}</el-tag>
        </div>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item label="当前阶段">
          {{ STAGE_LABELS[task.current_stage] ?? task.current_stage }}
        </el-descriptions-item>
        <el-descriptions-item label="驳回重做次数">{{ task.redo_count }}</el-descriptions-item>
      </el-descriptions>

      <el-alert
        v-if="task.last_error"
        class="error-alert"
        type="error"
        :title="task.last_error"
        :closable="false"
        show-icon
      />

      <StageTimeline class="timeline" :events="task.stages" />

      <!-- 待审：非终态，所以轮询会继续（is_terminal=false）——
           签字之后这一页会自己刷新出「已发布」，用户不用手动重开。 -->
      <el-alert
        v-if="task.status === 'awaiting_risk_review'"
        type="warning"
        title="已提交风控审核，等待签字"
        :description="auth.isResearcher
          ? '研报草稿已生成，需要风控人员签字后才会发布。本页会自动刷新。'
          : '请到「风控审核」页签字。本页会自动刷新。'"
        :closable="false"
        show-icon
      />
      <el-alert
        v-else-if="task.status === 'approved'"
        type="info"
        title="风控已签字通过，正在发布"
        :closable="false"
        show-icon
      />
      <el-alert
        v-else-if="task.result_available"
        type="success"
        title="研报已发布"
        :closable="false"
        show-icon
      >
        <template #default>
          <el-button type="primary" link @click="openReport">
            查看研报详情 →
          </el-button>
        </template>
      </el-alert>
      <el-alert
        v-else-if="task.status === 'rejected'"
        type="error"
        title="驳回次数达上限，已转人工处理"
        :closable="false"
        show-icon
      />
      <el-alert
        v-else-if="task.status === 'failed'"
        type="error"
        title="本次研究失败，详见上方错误信息"
        :closable="false"
        show-icon
      />
    </el-card>

    <!-- 历史研究：研究员/管理员看【自己发起】的历次研究（后端已按 created_by 过滤）。
         「进度」把任意一条历史任务加载进上方卡片看它的阶段时间线；已发布的再加「研报」直达详情。 -->
    <el-card v-if="canSubmit" class="panel">
      <template #header>
        <div class="panel-header">
          <span>历史研究</span>
          <el-button size="small" :loading="loadingHistory" @click="loadHistory">刷新</el-button>
        </div>
      </template>

      <el-empty v-if="!loadingHistory && history.length === 0" description="还没有发起过研究" />

      <el-table v-else :data="history" size="small">
        <el-table-column label="标的" min-width="180">
          <template #default="{ row }">{{ row.company_name }}（{{ row.company_code }}）</template>
        </el-table-column>
        <el-table-column label="提交时间" min-width="160">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusTag(row.status)">{{ STATUS_LABELS[row.status] ?? row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button link type="primary" @click="loadTask(row.id)">进度</el-button>
            <el-button
              v-if="row.status === 'published'"
              link
              type="primary"
              @click="openReportById(row.id)"
            >研报</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import StageTimeline from '@/components/research/StageTimeline.vue'
import { useAuthStore } from '@/stores/auth'
import {
  STAGE_LABELS, STATUS_LABELS,
  createTask, getTask, listCompanies, listTasks,
  type Company, type TaskDetail, type TaskSummary,
} from '@/api/research'

const POLL_INTERVAL_MS = 5000          // 设计文档 §8.5：5 秒一次

const auth = useAuthStore()
const router = useRouter()
const companies = ref<Company[]>([])
const selectedCode = ref('')
const task = ref<TaskDetail | null>(null)
const submitting = ref(false)
const submitError = ref('')
const history = ref<TaskSummary[]>([])
const loadingHistory = ref(false)

// 与后端 require_role("researcher", "admin") 一致
const canSubmit = computed(() => auth.isResearcher || auth.isAdmin)

// 轮询句柄。用递归 setTimeout 而不是 setInterval：
// setInterval 不看上一发是否返回，网络慢时会堆积并发请求，任务越跑越慢。
let pollTimer: ReturnType<typeof setTimeout> | null = null

const statusTagType = computed(() => {
  switch (task.value?.status) {
    case 'published': return 'success'
    case 'failed':
    case 'rejected':  return 'danger'
    case 'awaiting_risk_review': return 'warning'
    default: return 'info'
  }
})

// 拦截器已经弹过一次 ElMessage，这里只负责给卡片内的 alert 一句准确的话。
// 直接用 e.message 会拿到 axios 的英文原文「Request failed with status code 404」。
function errorText(e: unknown, fallback: string): string {
  const detail = (e as { response?: { data?: { detail?: string } } })
    ?.response?.data?.detail
  if (detail) return detail
  return e instanceof Error ? e.message : fallback
}

function stopPolling() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
}

function openReport() {
  if (!task.value) return
  router.push({ name: 'report', params: { taskId: task.value.task_id } })
}

function openReportById(taskId: string) {
  router.push({ name: 'report', params: { taskId } })
}

// 行级状态标签配色，与顶部卡片的 statusTagType 共用同一套语义。
function statusTag(status: string) {
  switch (status) {
    case 'published': return 'success'
    case 'failed':
    case 'rejected':  return 'danger'
    case 'awaiting_risk_review': return 'warning'
    default: return 'info'
  }
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString('zh-CN')
}

async function loadHistory() {
  if (!canSubmit.value) return
  loadingHistory.value = true
  try {
    const { data } = await listTasks(50)
    history.value = data.items
  } catch {
    // 列表加载失败不打扰用户：顶部的「发起研究」仍可用，下次进页或点刷新重试。
    history.value = []
  } finally {
    loadingHistory.value = false
  }
}

// 把历史里的某条任务加载进顶部「研究任务」卡片（走同一套轮询），
// 这样能看到它完整的阶段时间线与结果，而不只是表格里的一行。
async function loadTask(taskId: string) {
  stopPolling()
  task.value = null
  await pollOnce(taskId)
}

async function pollOnce(taskId: string) {
  try {
    const { data } = await getTask(taskId)
    task.value = data
    // 终态就停：任务不会再变了，继续轮询只是白耗流量
    if (data.is_terminal) {
      stopPolling()
      if (data.status === 'published') ElMessage.success('研报已发布')
      else if (data.status === 'rejected') ElMessage.error('驳回次数达上限，已转人工处理')
      else if (data.status === 'failed') ElMessage.error('研究任务失败')
      return
    }
    // awaiting_risk_review 也走这里：它是**非终态**，必须继续轮询 ——
    // 若把它当成终态停在这里，签字通过之后这一页就永远停在「待审核」了。
    pollTimer = setTimeout(() => pollOnce(taskId), POLL_INTERVAL_MS)
  } catch {
    // 单次轮询失败不终止：网络抖动过后下一轮就能接上。
    // 但也不能无限重试 —— 交给 setTimeout 继续，用户离开页面时 onBeforeUnmount 会停掉。
    pollTimer = setTimeout(() => pollOnce(taskId), POLL_INTERVAL_MS)
  }
}

async function handleSubmit() {
  if (!selectedCode.value) return
  submitting.value = true
  submitError.value = ''
  stopPolling()
  task.value = null

  try {
    const { data } = await createTask(selectedCode.value)
    await pollOnce(data.task_id)          // 立即拉一次，不等 5 秒
    await loadHistory()                   // 刚发起的这条进历史列表
  } catch (e: unknown) {
    submitError.value = errorText(e, '提交失败')
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  if (!canSubmit.value) return
  try {
    const { data } = await listCompanies()
    companies.value = data.items
    if (data.items.length > 0) selectedCode.value = data.items[0].code
  } catch (e: unknown) {
    submitError.value = errorText(e, '标的列表加载失败')
  }
  await loadHistory()
})

// 组件卸载必须停掉轮询，否则用户切走页面后请求还在后台发
onBeforeUnmount(stopPolling)
</script>

<style scoped>
.dashboard { display: flex; flex-direction: column; gap: 16px; }
.panel-header { display: flex; justify-content: space-between; align-items: center; }
.error-alert, .timeline { margin-top: 16px; }
</style>
