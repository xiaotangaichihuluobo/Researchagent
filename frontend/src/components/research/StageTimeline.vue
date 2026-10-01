<!-- frontend/src/components/research/StageTimeline.vue -->
<!-- 阶段时间线：把后端返回的只追加事件流，折叠成「每个阶段当前是什么状态」。 -->
<template>
  <el-timeline class="stage-timeline">
    <el-timeline-item
      v-for="stage in stages"
      :key="stage.key"
      :type="stage.type"
      :hollow="stage.type === 'info'"
      :timestamp="stage.timestamp"
    >
      <span class="stage-name">{{ stage.label }}</span>
      <span class="stage-status">{{ stage.statusText }}</span>
    </el-timeline-item>
  </el-timeline>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { STAGE_LABELS, STAGE_ORDER, type StageEvent } from '@/api/research'

const props = defineProps<{ events: StageEvent[] }>()

// 每个阶段取【最后一条】事件作为它的当前状态。
// 事件是只追加的，同一个阶段会有 started 和 success 两条 —— 后者才代表结果。
function statusOf(stageKey: string): StageEvent | undefined {
  const of = props.events.filter((e) => e.stage === stageKey)
  return of[of.length - 1]
}

const stages = computed(() =>
  STAGE_ORDER.map((key) => {
    const last = statusOf(key)
    let type: 'success' | 'danger' | 'primary' | 'info' = 'info'
    let statusText = '未开始'
    if (last) {
      if (last.status === 'success') { type = 'success'; statusText = '已完成' }
      else if (last.status === 'failed') { type = 'danger'; statusText = '失败' }
      else if (last.status === 'started') { type = 'primary'; statusText = '进行中' }
      else { type = 'info'; statusText = '已跳过' }
    }
    return {
      key,
      label: STAGE_LABELS[key] ?? key,
      type,
      statusText,
      timestamp: last ? new Date(last.occurred_at).toLocaleTimeString('zh-CN') : '',
    }
  })
)
</script>

<style scoped>
.stage-timeline { padding-left: 4px; }
.stage-name { font-weight: 500; margin-right: 8px; }
.stage-status { color: var(--el-text-color-secondary); font-size: 13px; }
</style>
