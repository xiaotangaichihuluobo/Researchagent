<!-- frontend/src/views/ReportView.vue -->
<!-- 研报详情页（只读）。研究员与风控看的是同一份已发布产物。
     数字一律取结构化字段，不解析正文 —— 正文给人读。 -->
<template>
  <div class="report-view">
    <el-card v-loading="loading" class="panel">
      <template #header>
        <div class="panel-header">
          <span>{{ report?.title ?? '研报详情' }}</span>
          <div>
            <el-tag v-if="report?.rating" type="success">{{ report.rating }}</el-tag>
            <el-button size="small" style="margin-left: 8px" @click="load">刷新</el-button>
          </div>
        </div>
      </template>

      <el-alert v-if="error" type="error" :title="error" :closable="false" show-icon />

      <template v-if="report">
        <el-descriptions :column="2" border class="meta">
          <el-descriptions-item label="发布时间">
            {{ report.published_at ? formatTime(report.published_at) : '—' }}
          </el-descriptions-item>
          <el-descriptions-item label="历史参照">
            {{ report.references.has_reference ? '有' : '无' }}
          </el-descriptions-item>
        </el-descriptions>

        <!-- 估值：结构化字段渲染。不可用时明确写「不可用」，
             绝不显示 0 或任何看起来像数字的占位。 -->
        <h4 class="section-title">估值</h4>
        <el-empty
          v-if="!valuation || !valuation.is_available"
          description="本次未产出可用估值 —— 估值失败绝不编造数字"
          :image-size="60"
        />
        <template v-else>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="估值方法">
              {{ VALUATION_METHOD_LABELS[valuation.method ?? ''] ?? '—' }}
            </el-descriptions-item>
            <el-descriptions-item label="股权价值区间">
              {{ fmt(valuation.equity_value_low) }} ~ {{ fmt(valuation.equity_value_high) }} 亿元
            </el-descriptions-item>
            <el-descriptions-item label="每股价值区间">
              <template v-if="valuation.per_share_low !== null && valuation.per_share_high !== null">
                {{ fmt(valuation.per_share_low) }} ~ {{ fmt(valuation.per_share_high) }} 元/股
              </template>
              <!-- 总股本没提取到：说明为什么没有，而不是留空 -->
              <span v-else class="muted">未取到总股本，不提供每股口径</span>
            </el-descriptions-item>
            <el-descriptions-item label="币种">{{ valuation.currency }}</el-descriptions-item>
          </el-descriptions>

          <p v-if="valuation.rationale" class="rationale">{{ valuation.rationale }}</p>

          <!-- 假设来源逐条展示。来源直接显示原始标识（config:xxx / derived:xxx），
               因为「这是配置里的经验值」与「这来自数据」对读者是两件不同的事。 -->
          <h4 class="section-title">估值假设与来源</h4>
          <el-table :data="valuation.assumptions" size="small" border>
            <el-table-column label="假设" prop="name" min-width="160" />
            <el-table-column label="取值" prop="value" width="120" />
            <el-table-column label="来源" prop="source" min-width="220" />
          </el-table>
        </template>

        <h4 class="section-title">风险揭示</h4>
        <pre class="prose prose--risk">{{ report.risk_disclosure || '（无）' }}</pre>

        <h4 class="section-title">研报正文</h4>
        <pre class="prose">{{ report.content }}</pre>
      </template>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { getReport, VALUATION_METHOD_LABELS, type ReportDetail } from '@/api/research'

const route = useRoute()
const report = ref<ReportDetail | null>(null)
const loading = ref(false)
const error = ref('')

const valuation = computed(() => report.value?.valuation ?? null)

function fmt(value: number | null): string {
  return value === null ? '—' : value.toFixed(2)
}

function formatTime(value: string): string {
  return new Date(value).toLocaleString('zh-CN')
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const { data } = await getReport(String(route.params.taskId))
    report.value = data
  } catch (e: unknown) {
    // 404 是正常业务分支（未发布、跨租户），不是意外
    const status = (e as { response?: { status?: number } })?.response?.status
    error.value = status === 404
      ? '该任务尚未发布研报（草稿只对风控审核台可见）'
      : '加载研报失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.report-view { display: flex; flex-direction: column; gap: 16px; }
.panel-header { display: flex; justify-content: space-between; align-items: center; }
.meta { margin-bottom: 8px; }
.section-title { margin: 18px 0 8px; font-size: 14px; color: #303133; }
.prose {
  white-space: pre-wrap;
  background: #fafafa;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 12px;
  font-size: 13px;
  line-height: 1.7;
  max-height: 480px;
  overflow: auto;
}
.prose--risk { background: #fff8f0; border-color: #f5dab1; }
.rationale { margin: 12px 0 0; color: #606266; font-size: 13px; }
.muted { color: #909399; }
</style>