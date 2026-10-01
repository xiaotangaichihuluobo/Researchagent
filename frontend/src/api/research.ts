// frontend/src/api/research.ts
// 投研域接口封装。client.ts 的 baseURL 已经带了 /api/v1，这里只写相对路径。

import client from './client'

export interface Company {
  id: string
  code: string
  name: string
  industry: string
}

export interface StageEvent {
  stage: string
  status: 'started' | 'success' | 'failed' | 'skipped'
  detail: Record<string, unknown>
  occurred_at: string
}

export interface TaskDetail {
  task_id: string
  status: 'pending' | 'running' | 'awaiting_risk_review' | 'approved' | 'rejected' | 'published' | 'failed'
  current_stage: string
  redo_count: number
  last_error: string | null
  is_terminal: boolean          // 后端算好，前端据此停止轮询
  result_available: boolean
  stages: StageEvent[]
}

export interface TaskSummary {
  id: string
  company_code: string
  company_name: string
  status: TaskDetail['status']
  current_stage: string
  created_at: string
}

// 阶段的中文名与展示顺序（与后端 current_stage 取值一一对应）
export const STAGE_LABELS: Record<string, string> = {
  collect:     '数据采集',
  analyze:     '多维分析',
  retrieve:    '研报检索',
  valuation:   '估值测算',
  risk_review: '风控审核',
  publish:     '报告发布',
}

export const STAGE_ORDER = ['collect', 'analyze', 'retrieve', 'valuation', 'risk_review', 'publish']

// 与后端 research_tasks.status 的 CHECK 约束一一对应（scripts/init_db.sql:59）
export const STATUS_LABELS: Record<string, string> = {
  pending:              '排队中',
  running:              '运行中',
  awaiting_risk_review: '待风控审核',
  approved:             '风控已通过',
  rejected:             '已驳回',
  published:            '已发布',
  failed:               '失败',
}

export function createTask(companyCode: string) {
  return client.post<{ task_id: string; status: string }>('/research/tasks', {
    company_code: companyCode,
  })
}

export function getTask(taskId: string) {
  return client.get<TaskDetail>(`/research/tasks/${taskId}`)
}

export function listTasks(limit = 20, status?: string) {
  return client.get<{ items: TaskSummary[] }>('/research/tasks', { params: { limit, status } })
}

// ── 研报详情（P5）────────────────────────────────────────────

// 估值假设。source 的取值形如 config:valuation_discount_rate /
// derived:research_data_items —— 界面直接把来源标出来，不翻译：
// 「这条是配置里的经验值」与「这条来自数据」对读者是两件不同的事。
export interface ValuationAssumption {
  name: string
  value: number
  source: string
  excerpt?: string
}

export interface ReportValuation {
  method: 'dcf' | 'comparable' | 'blended' | null
  equity_value_low: number | null      // 亿元
  equity_value_high: number | null
  per_share_low: number | null         // 元/股
  per_share_high: number | null
  currency: string
  assumptions: ValuationAssumption[]
  rationale: string | null
  is_available: boolean
}

export interface ReportDetail {
  task_id: string
  title: string
  content: string
  rating: string | null
  risk_disclosure: string | null
  published_at: string | null
  status: string
  valuation: ReportValuation | null
  references: { has_reference: boolean; comparison_points: string[] }
}

export function getReport(taskId: string) {
  return client.get<ReportDetail>(`/research/tasks/${taskId}/report`)
}

// 估值方法的中文名。null 是「没有方法」（不可用），不是「方法未知」。
export const VALUATION_METHOD_LABELS: Record<string, string> = {
  dcf:        '现金流折现（DCF）',
  comparable: '可比公司法',
  blended:    '双路合成',
}

// ── 风控人工签字（P6）──────────────────────────────────────────

// 预检项。passed 为 null 表示「自动预检判不了，得由签字人自己确认」——
// 不用 false 冒充一个自动结论（后端 risk/nodes.py 的 conflict_declared 就是这样）。
export interface ChecklistItem {
  key: string
  label: string
  passed: boolean | null
  note: string
}

export interface ReviewDraft {
  id: string
  title: string
  content: string
  rating: string | null
  status: 'draft' | 'published' | 'rejected'
  risk_disclosure: string | null
}

export interface ReviewRecord {
  id: string
  decision: 'approve' | 'modify' | 'reject' | null
  comments: string | null
  redo_targets: { stage?: string; dimensions?: string[] } | null
  signed_at: string | null
}

export interface ReviewDetail {
  task_id: string
  status: TaskDetail['status']
  current_stage: string
  redo_count: number
  // 后端算好的「这条现在还能不能签」。前端不自己从 status 推 ——
  // 那种推算一旦和后端的判定漂移，界面会给出一个后端不接受的按钮。
  reviewable: boolean
  report: ReviewDraft | null
  checklist: { items?: ChecklistItem[]; blocking?: string[] }
  review: ReviewRecord | null
  history: { decision: string; comments: string; redo_targets: unknown; signed_at: string }[]
  stages: StageEvent[]
}

export interface RiskDecisionPayload {
  decision: 'approve' | 'modify' | 'reject'
  comments: string
  redo_targets?: { stage: string; dimensions?: string[] }
}

export function getReview(taskId: string) {
  return client.get<ReviewDetail>(`/research/tasks/${taskId}/review`)
}

export function submitRiskDecision(taskId: string, payload: RiskDecisionPayload) {
  return client.post<{ task_id: string; decision: string; status: string }>(
    `/research/tasks/${taskId}/risk-decision`,
    payload,
  )
}

export const DECISION_LABELS: Record<string, string> = {
  approve: '通过',
  modify:  '有条件通过',
  reject:  '驳回重做',
}

// 驳回时可选的重做阶段。与后端 risk/nodes.py 的 REDO_STAGES 一一对应，
// 且必须与研发语言的 STAGE_LABELS 对得上 —— 否则界面上会冒出一个内部标识。
export const REDO_STAGE_OPTIONS = [
  { value: 'analyze',   label: '多维分析' },
  { value: 'valuation', label: '估值测算' },
]

// 维度取值与后端 research_rules.DIMENSIONS 一一对应
export const DIMENSION_LABELS: Record<string, string> = {
  fundamental: '基本面',
  technical:   '技术面',
  sentiment:   '市场情绪',
  industry:    '行业景气',
}

export const MIN_COMMENTS_LEN = 10

export function listCompanies() {
  return client.get<{ items: Company[] }>('/companies')
}
