<!-- frontend/src/views/RiskReviewView.vue -->
<!-- 风控审核台：待审列表 → 待审内容（草稿 + 预检清单 + 阶段时间线）→ 签字。
     「未签字不得发布」这条规则在人这一侧的入口就是这一页。 -->
<template>
  <div class="risk-review">
    <el-card class="panel">
      <template #header>
        <div class="panel-header">
          <span>待风控审核</span>
          <el-button size="small" :loading="loadingList" @click="loadList">刷新</el-button>
        </div>
      </template>

      <el-empty v-if="!loadingList && tasks.length === 0" description="当前没有待审的研究任务" />

      <el-table
        v-else
        :data="tasks"
        highlight-current-row
        :current-row-key="selectedId ?? undefined"
        row-key="id"
        @current-change="onSelect"
      >
        <el-table-column label="标的" min-width="180">
          <template #default="{ row }">
            {{ row.company_name }}（{{ row.company_code }}）
          </template>
        </el-table-column>
        <el-table-column label="提交时间" min-width="150">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="状态" width="130">
          <template #default="{ row }">
            <el-tag type="warning">{{ STATUS_LABELS[row.status] ?? row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="驳回次数" width="100" prop="redo_count" />
      </el-table>
    </el-card>

    <el-card v-if="detail" class="panel">
      <template #header>
        <div class="panel-header">
          <span>{{ detail.report?.title ?? '待审内容' }}</span>
          <el-tag :type="detail.reviewable ? 'warning' : 'info'">
            {{ STATUS_LABELS[detail.status] ?? detail.status }}
          </el-tag>
        </div>
      </template>

      <el-alert
        v-if="!detail.reviewable"
        type="info"
        :closable="false"
        show-icon
        :title="`这条任务当前不在待审状态（${STATUS_LABELS[detail.status] ?? detail.status}），只能查看`"
      />

      <el-descriptions :column="3" border class="meta">
        <el-descriptions-item label="评级">{{ detail.report?.rating ?? '未评级' }}</el-descriptions-item>
        <el-descriptions-item label="驳回重做次数">{{ detail.redo_count }}</el-descriptions-item>
        <el-descriptions-item label="草稿状态">{{ detail.report?.status ?? '—' }}</el-descriptions-item>
      </el-descriptions>

      <!-- 预检清单：只提供信息，不做裁决。blocking 项高亮，但**不**禁用签字 ——
           裁决权在人，这正是这个页面存在的理由。 -->
      <h4 class="section-title">自动预检</h4>
      <el-table :data="checklistItems" size="small" border>
        <el-table-column label="预检项" prop="label" min-width="160" />
        <el-table-column label="结果" width="110">
          <template #default="{ row }">
            <el-tag v-if="row.passed === true" type="success" size="small">通过</el-tag>
            <el-tag v-else-if="row.passed === false" type="danger" size="small">未通过</el-tag>
            <!-- passed === null：自动预检判不了，得签字人自己确认 -->
            <el-tag v-else type="info" size="small">待人工确认</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="说明" prop="note" min-width="280" />
      </el-table>

      <h4 class="section-title">研报草稿</h4>
      <pre class="draft">{{ detail.report?.content }}</pre>

      <template v-if="detail.report?.risk_disclosure">
        <h4 class="section-title">风险揭示</h4>
        <pre class="draft draft--risk">{{ detail.report.risk_disclosure }}</pre>
      </template>

      <h4 class="section-title">阶段时间线</h4>
      <StageTimeline :events="detail.stages" />

      <template v-if="detail.history.length > 0">
        <h4 class="section-title">历史审核意见</h4>
        <el-timeline>
          <el-timeline-item
            v-for="(h, i) in detail.history"
            :key="i"
            :timestamp="formatTime(h.signed_at)"
            :type="h.decision === 'reject' ? 'danger' : 'success'"
          >
            <b>{{ DECISION_LABELS[h.decision] ?? h.decision }}</b> —— {{ h.comments }}
          </el-timeline-item>
        </el-timeline>
      </template>

      <!-- 签字表单 -->
      <div v-if="detail.reviewable" class="sign-form">
        <el-divider />
        <h4 class="section-title">签字</h4>

        <el-form label-width="90px">
          <el-form-item label="决策">
            <el-radio-group v-model="form.decision">
              <el-radio value="approve">通过</el-radio>
              <el-radio value="modify">有条件通过</el-radio>
              <el-radio value="reject">驳回重做</el-radio>
            </el-radio-group>
          </el-form-item>

          <!-- modify = 有条件通过：意见随研报披露，但内容不改、不重跑。
               这一点必须写在界面上，否则审核人会以为选了它会触发返工。 -->
          <el-form-item v-if="form.decision === 'modify'" label=" ">
            <span class="hint">有条件通过：意见会随研报披露，内容不修改、不重跑。</span>
          </el-form-item>

          <template v-if="form.decision === 'reject'">
            <el-form-item label="重做阶段">
              <el-select v-model="form.redoStage" style="width: 200px">
                <el-option
                  v-for="o in REDO_STAGE_OPTIONS"
                  :key="o.value"
                  :label="o.label"
                  :value="o.value"
                />
              </el-select>
            </el-form-item>
            <el-form-item v-if="form.redoStage === 'analyze'" label="重做维度">
              <el-checkbox-group v-model="form.redoDimensions">
                <el-checkbox v-for="(label, key) in DIMENSION_LABELS" :key="key" :value="key">
                  {{ label }}
                </el-checkbox>
              </el-checkbox-group>
            </el-form-item>
          </template>

          <el-form-item label="意见">
            <el-input
              v-model="form.comments"
              type="textarea"
              :rows="3"
              :placeholder="`必填，不得短于 ${MIN_COMMENTS_LEN} 字（签字的合规要求：谁签字谁负责）`"
            />
            <span class="hint">{{ form.comments.trim().length }} / {{ MIN_COMMENTS_LEN }} 字</span>
          </el-form-item>

          <el-form-item>
            <el-button type="primary" :loading="submitting" @click="handleSign">提交签字</el-button>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import StageTimeline from '@/components/research/StageTimeline.vue'
import {
  DECISION_LABELS, DIMENSION_LABELS, MIN_COMMENTS_LEN, REDO_STAGE_OPTIONS, STATUS_LABELS,
  getReview, listTasks, submitRiskDecision,
  type ChecklistItem, type ReviewDetail, type TaskSummary,
} from '@/api/research'

const tasks = ref<TaskSummary[]>([])
const detail = ref<ReviewDetail | null>(null)
const selectedId = ref<string | null>(null)
const loadingList = ref(false)
const submitting = ref(false)

// 后端会拒绝不足 10 字的意见（422），前端先把按钮状态说清楚，省得用户白跑一趟。
// 真正的边界仍在后端：这里只是 UX。
const form = ref({
  decision: 'approve' as 'approve' | 'modify' | 'reject',
  comments: '',
  redoStage: 'analyze',
  redoDimensions: [] as string[],
})

const checklistItems = computed<ChecklistItem[]>(() => detail.value?.checklist?.items ?? [])
const commentsValid = computed(() => form.value.comments.trim().length >= MIN_COMMENTS_LEN)

function formatTime(value: string | null): string {
  return value ? new Date(value).toLocaleString('zh-CN') : '—'
}

function errorText(e: unknown, fallback: string): string {
  const detailMsg = (e as { response?: { data?: { detail?: string } } })
    ?.response?.data?.detail
  if (detailMsg) return detailMsg
  return e instanceof Error ? e.message : fallback
}

async function loadList() {
  loadingList.value = true
  try {
    // 只拉待审的：审核人每天打开的第一页不该是全租户的任务流水
    const { data } = await listTasks(50, 'awaiting_risk_review')
    tasks.value = data.items
  } catch (e: unknown) {
    ElMessage.error(errorText(e, '待审列表加载失败'))
  } finally {
    loadingList.value = false
  }
}

async function onSelect(row: TaskSummary | null) {
  if (!row) return
  selectedId.value = row.id
  // 换一条就清空草稿里的字，否则上一条的意见会跟着下一条一起提交出去
  form.value = { decision: 'approve', comments: '', redoStage: 'analyze', redoDimensions: [] }
  try {
    const { data } = await getReview(row.id)
    detail.value = data
  } catch (e: unknown) {
    detail.value = null
    ElMessage.error(errorText(e, '待审内容加载失败'))
  }
}

/**
 * 签字后要**等这条任务真的离开待审列表**，不能只刷新一次。
 *
 * POST 返回 202 只代表「已受理」—— 真正把状态翻走的是后台那次 resume。
 * 受理的那一瞬间任务还是 awaiting_risk_review，此时 loadList() 拉回来的仍是旧行；
 * 而之后再没有任何东西会去刷新它，那行就永远挂在待审列表上：审核人以为还没签，
 * 点进去只能拿到 409「任务当前不在待审状态」。
 *
 * 这与后端 mark_redo_started 修掉的那个竞态是同一个形状：**受理 ≠ 生效**。
 * 超时不当作失败 —— 任务确实已经签了，只是后台还没跑完；如实告诉用户去看刷新按钮，
 * 比假装成功或假装失败都强。
 */
async function loadListUntilGone(taskId: string, timeoutMs = 15000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    await loadList()
    if (!tasks.value.some((t) => t.id === taskId)) return true
    if (Date.now() >= deadline) return false
    await new Promise((r) => setTimeout(r, 700))
  }
}

async function handleSign() {
  if (!detail.value) return
  if (!commentsValid.value) {
    ElMessage.warning(`签字意见不得短于 ${MIN_COMMENTS_LEN} 字`)
    return
  }

  submitting.value = true
  try {
    const payload = {
      decision: form.value.decision,
      comments: form.value.comments.trim(),
      ...(form.value.decision === 'reject'
        ? {
            redo_targets: {
              stage: form.value.redoStage,
              ...(form.value.redoStage === 'analyze'
                ? { dimensions: form.value.redoDimensions }
                : {}),
            },
          }
        : {}),
    }
    const signedId = detail.value.task_id
    await submitRiskDecision(signedId, payload)
    ElMessage.success('签字已提交，任务继续流转')
    detail.value = null
    selectedId.value = null
    if (!(await loadListUntilGone(signedId))) {
      ElMessage.warning('签字已提交，但后台还在流转 —— 稍后点「刷新」确认这条已离开待审列表')
    }
  } catch (e: unknown) {
    // 409 的两种可能（不在待审状态 / 暂停态丢失）后端都把话说清楚了，直接显示原文
    ElMessage.error(errorText(e, '签字提交失败'))
  } finally {
    submitting.value = false
  }
}

onMounted(loadList)
</script>

<style scoped>
.risk-review { display: flex; flex-direction: column; gap: 16px; }
.panel-header { display: flex; justify-content: space-between; align-items: center; }
.meta { margin-bottom: 8px; }
.section-title { margin: 18px 0 8px; font-size: 14px; font-weight: 600; }
.draft {
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--el-fill-color-light);
  border-radius: 4px;
  padding: 12px;
  font-size: 13px;
  line-height: 1.7;
  max-height: 420px;
  overflow-y: auto;
  font-family: inherit;
  margin: 0;
}
.draft--risk { background: var(--el-color-warning-light-9); }
.sign-form { margin-top: 8px; }
.hint { color: var(--el-text-color-secondary); font-size: 12px; margin-left: 8px; }
</style>
