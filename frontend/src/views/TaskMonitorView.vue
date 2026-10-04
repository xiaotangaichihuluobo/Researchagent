<!-- frontend/src/views/TaskMonitorView.vue -->
<!-- 任务监控（仅管理员）：列出全租户任务，重点看失败任务的完整 traceback。
     「失败过程」的数据（stage detail 里的 error + trace）后端本来就在
     task_stage_events.detail 里记录并随 /tasks/{id} 返回，只是工作台那个
     StageTimeline 把它折叠成状态徽标、从没展开过。本页把它摊开给管理员排障。 -->
<template>
  <div class="monitor">
    <el-page-header title="任务监控" content="全租户任务流水与失败排障" />

    <!-- 状态过滤器：后端 /tasks 已支持 ?status=，直接复用，服务端过滤。 -->
    <div class="filter-bar">
      <el-radio-group v-model="statusFilter" @change="reloadList">
        <el-radio-button label="">全部</el-radio-button>
        <el-radio-button label="failed">失败</el-radio-button>
        <el-radio-button label="awaiting_risk_review">待审</el-radio-button>
        <el-radio-button label="published">已发布</el-radio-button>
      </el-radio-group>
      <el-button :loading="loadingList" @click="reloadList">刷新</el-button>
    </div>

    <!-- 左：任务列表 -->
    <el-table
      :data="tasks"
      v-loading="loadingList"
      highlight-current-row
      @current-change="onSelectTask"
    >
      <el-table-column label="标的" min-width="150">
        <template #default="{ row }">
          <span class="mono">{{ row.company_code }}</span>
          &nbsp;{{ row.company_name }}
        </template>
      </el-table-column>
      <el-table-column label="状态" width="120">
        <template #default="{ row }">
          <el-tag :type="taskStatusTag(row.status)">{{ statusLabel(row.status) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="当前阶段" width="120">
        <template #default="{ row }">{{ stageLabel(row.current_stage) }}</template>
      </el-table-column>
      <el-table-column label="发起时间" width="170">
        <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
      </el-table-column>
    </el-table>

    <!-- 右：任务详情（失败过程） -->
    <el-card v-if="selectedTask" class="detail-card" shadow="never">
      <template #header>
        <div class="detail-header">
          <span class="mono">{{ selectedTask.company_code }}</span>
          <span>{{ selectedTask.company_name }}</span>
          <el-tag :type="taskStatusTag(selectedTask.status)">
            {{ statusLabel(selectedTask.status) }}
          </el-tag>
          <span class="muted">发起于 {{ formatTime(selectedTask.created_at) }}</span>
        </div>
      </template>

      <!-- 顶楼错误摘要 -->
      <el-alert
        v-if="selectedTask.last_error"
        class="last-error"
        type="error"
        :title="selectedTask.last_error"
        :closable="false"
        show-icon
      />
      <p v-else class="muted">该任务无 last_error（非 fail 终态或错误文本为空）。</p>

      <!-- 阶段级失败过程：每个 failed 阶段展开其 detail 里的 trace -->
      <h4 class="section-title">阶段流水</h4>
      <el-timeline class="stages">
        <el-timeline-item
          v-for="stage in stageRows"
          :key="stage.stage"
          :type="stage.type"
          :hollow="stage.type === 'info'"
          :timestamp="stage.timestamp"
        >
          <div class="stage-head">
            <span class="stage-name">{{ stage.label }}</span>
            <span class="stage-status">{{ stage.statusText }}</span>
          </div>

          <!-- 只有 failed 阶段才展开排障栈；success/started/skipped 没有 detail -->
          <template v-if="stage.failDetail">
            <div class="trace-block">
              <div class="trace-title">
                <el-icon><WarningFilled /></el-icon>
                <span>失败原因（detail.error）</span>
              </div>
              <p class="trace-error">{{ stage.failDetail.error || '（无 error 文本）' }}</p>
              <div class="trace-title">
                <el-icon><Document /></el-icon>
                <span>traceback（detail.trace）</span>
              </div>
              <pre class="trace">{{ stage.failDetail.trace || '（无 traceback）' }}</pre>
            </div>
          </template>
        </el-timeline-item>
      </el-timeline>
    </el-card>
    <el-empty v-else class="empty" description="在左侧选择一条任务查看失败过程" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import { Document, WarningFilled } from '@element-plus/icons-vue'
import {
  getTask,
  listTasks,
  STAGE_LABELS,
  STAGE_ORDER,
  STATUS_LABELS,
  type StageEvent,
  type TaskDetail,
  type TaskSummary,
} from '@/api/research'

// 详情对象 = 列表字段（company_code/company_name/created_at，来自列表行）
//           ∪ 详情字段（stages/last_error，来自 get_task）。
// 不直接继承 TaskSummary —— 它的 id 语义（列表行的主键）与详情接口返回的
// task_id 不是同一个键，继承下去要么缺 id 要么多出无意义字段，故显式列字段。
type TaskMonitorDetail = Omit<TaskSummary, 'id' | 'status'> & TaskDetail

// 本页是管理员排障台，不做 DashboardView 那种 5s 轮询：那个是为进行中任务
// （run→publish）连续更新，这里是查已失败 / 已终态的历史任务，点「刷新」手动拉即可。

const statusFilter = ref('')
const tasks = ref<TaskSummary[]>([])
const loadingList = ref(false)
const selectedTask = ref<TaskMonitorDetail | null>(null)
const selectedId = ref('')

function statusLabel(s: string): string {
  return STATUS_LABELS[s] ?? s
}

function stageLabel(s: string): string {
  return STAGE_LABELS[s] ?? s
}

function taskStatusTag(status: string) {
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

async function reloadList() {
  loadingList.value = true
  try {
    const { data } = await listTasks(50, statusFilter.value || undefined)
    tasks.value = data.items
  } catch {
    tasks.value = []
  } finally {
    loadingList.value = false
  }
}

// 点某一行 → 拉详情。当前行没变化就不用重复拉。
async function onSelectTask(row: TaskSummary | null) {
  if (!row) return
  if (selectedId.value === row.id) return
  selectedId.value = row.id
  try {
    const { data } = await getTask(row.id)
    // 详情接口没有 company_code/company_name/created_at，从选中的列表行补上，
    // 合并成一个大对象供模板一起渲染（见 TaskMonitorDetail 类型）。
    selectedTask.value = {
      company_code: row.company_code,
      company_name: row.company_name,
      created_at: row.created_at,
      ...data,
    }
  } catch {
    selectedTask.value = null
  }
}

// 阶段行：把只追加的事件流折叠成「每阶段当前状态」，并把 failed 阶段的
// detail（error + trace）原样透出。逻辑与 StageTimeline 对齐，但这里
// 不折叠丢弃 trace —— 那正是本页要点。
function statusRows(events: StageEvent[]) {
  const last: Record<string, StageEvent | undefined> = {}
  for (const e of events) last[e.stage] = e
  return last
}

const stageRows = computed(() => {
  const last = statusRows(selectedTask.value?.stages ?? [])
  return STAGE_ORDER.map((key) => {
    const e = last[key]
    let type: 'success' | 'danger' | 'primary' | 'info' = 'info'
    let statusText = '未开始'
    if (e) {
      if (e.status === 'success')      { type = 'success'; statusText = '已完成' }
      else if (e.status === 'failed')  { type = 'danger';  statusText = '失败' }
      else if (e.status === 'started') { type = 'primary'; statusText = '进行中' }
      else                             { type = 'info';    statusText = '已跳过' }
    }
    return {
      stage: key,
      label: STAGE_LABELS[key] ?? key,
      type,
      statusText,
      timestamp: e ? new Date(e.occurred_at).toLocaleTimeString('zh-CN') : '',
      failDetail: (e && e.status === 'failed' && e.detail && e.detail.trace)
        ? e.detail
        : null,
    }
  })
})

onMounted(reloadList)
</script>

<style scoped>
.monitor { display: flex; flex-direction: column; gap: 16px; }
.filter-bar { display: flex; align-items: center; gap: 12px; }
.mono { font-family: "Cascadia Mono", Consolas, monospace; }
.muted { color: var(--el-text-color-secondary); font-size: 13px; }

.detail-header { display: flex; align-items: center; gap: 8px; }
.last-error { margin-bottom: 12px; }
.section-title { margin: 8px 0 12px; color: var(--el-text-color-primary); }

.stage-head { display: flex; align-items: center; gap: 8px; }
.stage-name { font-weight: 500; }
.stage-status { color: var(--el-text-color-secondary); font-size: 13px; }

.trace-block {
  margin-top: 8px;
  padding: 10px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  background: var(--el-fill-color-lighter);
}
.trace-title {
  display: flex; align-items: center; gap: 6px;
  color: var(--el-text-color-regular); font-size: 13px; font-weight: 500;
  margin: 6px 0 4px;
}
.trace-title:first-child { margin-top: 0; }
.trace-error {
  margin: 0 0 6px;
  padding: 8px;
  border-radius: 4px;
  background: #fef3f2;
  color: #b42318;
  font-size: 13px;
  word-break: break-word;
}
.trace {
  margin: 0;
  padding: 10px;
  max-height: 320px;
  overflow: auto;
  border-radius: 4px;
  background: #0f172a;
  color: #e2e8f0;
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;      /* traceback 里较宽的路径/行号能换行而非撑破卡片 */
  word-break: break-word;
}
.empty { padding: 48px 0; }
</style>