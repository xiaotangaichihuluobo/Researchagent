<template>
  <div class="qa-page">
    <!-- 左：会话列表 -->
    <aside class="qa-side">
      <div class="qa-side-head">
        <span class="title">会话</span>
        <el-button type="primary" size="small" :icon="Plus" @click="newSession">新会话</el-button>
        <el-button
          v-if="sessions.length"
          size="small"
          :icon="Delete"
          :disabled="streaming"
          @click="clearAll"
        >清空</el-button>
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          class="session-item"
          :class="{ 'session-item--active': s.id === currentId }"
          @click="switchSession(s.id)"
        >
          <span class="session-name">{{ s.name }}</span>
          <el-icon class="session-remove" title="删除" @click.stop="removeSession(s)">
            <Delete />
          </el-icon>
        </div>
        <div v-if="sessions.length === 0" class="session-empty">暂无会话</div>
      </div>
    </aside>

    <!-- 右：对话区 -->
    <section class="qa-main">
      <div ref="scrollBox" class="msg-list">
        <div v-if="messages.length === 0" class="empty-hint">
          <div class="empty-icon">📚</div>
          <p>问已发布的研报，例如「贵州茅台 2025 年营收怎么样」</p>
          <p class="sub">支持多轮追问，如「那五粮液呢」</p>
        </div>

        <div
          v-for="(m, i) in messages"
          :key="i"
          class="msg"
          :class="m.role === 'user' ? 'msg--user' : 'msg--assistant'"
        >
          <div class="bubble user-bubble" v-if="m.role === 'user'">{{ m.content }}</div>
          <div class="bubble" v-else v-html="renderMarkdown(m.content)" />
          <!-- aborted 的 assistant 必然也走上面 49 行渲染内容；这里只在有中断标记时并列追加一条提示 -->
          <div v-if="m.role === 'assistant' && m.aborted" class="interrupted-tag">⚠️ 已中断，继续提问后回答未能完成</div>
          <div v-if="m.sources && m.sources.length" class="sources">
            <span class="sources-label">参考来源</span>
            <a v-for="(src, si) in m.sources" :key="si" :href="src" target="_blank" rel="noopener" class="chip">
              {{ src }}
            </a>
          </div>
        </div>
      </div>

      <!-- 底部输入 -->
      <div class="input-bar">
        <div class="input-row">
          <el-checkbox v-model="enableWeb" size="small" class="web-toggle">低置信联网兜底</el-checkbox>
          <el-checkbox v-model="agentic" size="small" class="web-toggle">Agentic 深度搜索</el-checkbox>
          <el-input
            v-model="draft"
            :placeholder="streaming ? '正在回答…；可继续输入，回车将中断当前回答' : '输入问题，回车发送（Shift+Enter 换行）'"
            type="textarea"
            :rows="2"
            resize="none"
            @keydown.enter.exact.prevent="send"
          />
          <el-button
            type="primary"
            :disabled="!draft.trim()"
            @click="send"
          >
            {{ streaming ? '中断并发送下一问' : '发送' }}
          </el-button>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted } from 'vue'
import { Plus, Delete } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { v4 as uuidv4 } from 'uuid'
import MarkdownIt from 'markdown-it'
import { chatStream, getHistory, listSessions, deleteSession, deleteAllSessions } from '@/api/qa'

// 会话名本地缓存：列表本身以 /qa/sessions（后端）为准，这里只存「显示名」。
// 后端 qa_sessions 没有 name 列，展示靠首轮提问，缓存避免刷新丢名。
const STORAGE_KEY = 'research-agent-qa-names'

interface Msg {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  aborted?: boolean   // 流式中被新提问/切走打断的半截回答：保留内容、渲染时标「已中断」
}
interface Session {
  id: string
  name: string
  turns: number
}

const md = new MarkdownIt({
  html: false,          // 禁用原始 HTML，防注入
  linkify: true,
  breaks: true,
}).enable('link')

function renderMarkdown(s: string): string {
  // markdown-it 已禁原始 HTML（html:false），用户文本走文本插值、回答走此处，
  // 两处都不会把前端脚本渲染成可执行代码。
  return md.render(s)
}

// ── 会话状态 ──
const currentId = ref('')
// 会话列表：从 /qa/sessions（后端）加载，是权威来源。这里只存「显示名」为运行时态。
const sessions = ref<Session[]>([])
const messages = ref<Msg[]>([])
const draft = ref('')
const streaming = ref(false)
const streamText = ref('')
const enableWeb = ref(false)
// Agentic 多轮自搜引擎开关：开启则给后端传 engine_mode=agentic（投研问题走检索+联网）
const agentic = ref(false)
const scrollBox = ref<HTMLElement | null>(null)
// 在途 SSE 流的 abort 句柄；streamOwnerId 记录这轮流属于哪个会话，
// 切换会话时 abort 掉它，且流的收尾（落名/刷新列表）只对 owner 会话生效。
const streamAbort = ref<AbortController | null>(null)
const streamOwnerId = ref('')

// 真中断在途 SSE：只 abort fetch 句柄 + 清空 owner（让被断流的 finally 短路），
// 不改 streaming —— 调用方按需决定。切走/新建会话清 streaming，同会话重发则要让新流接着置位。
function kickStream() {
  if (streamAbort.value) {
    streamAbort.value.abort()
    streamAbort.value = null
  }
  streamOwnerId.value = ''
}

function abortStream() {
  kickStream()
  streaming.value = false
}

// 会话名缓存：{ session_id -> 显示名 }。展示靠首轮提问，刷新后从缓存读回；
// 后端没有 name 列，这是纯前端展示层的东西。
function loadNames(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
  } catch {
    return {}
  }
}
function persistNames(names: Record<string, string>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(names))
}
function nameFor(sessionId: string): string {
  return loadNames()[sessionId] ?? ''
}

async function loadFromBackend() {
  try {
    const list = await listSessions()
    const names = loadNames()
    sessions.value = list.map(s => ({
      id: s.session_id,
      name: names[s.session_id] ?? `会话 ${list.indexOf(s) + 1}`,
      turns: s.turns,
    }))
    // 当前会话若已被后端列表覆盖/删除，重置为空，避免 stale。
    if (currentId.value && !sessions.value.some(x => x.id === currentId.value)) {
      currentId.value = ''
    }
    // 有历史会话时默认选中最近一个，方便回看。
    if (!currentId.value && sessions.value.length) {
      currentId.value = sessions.value[0].id
    }
  } catch {
    // 后端没起/报错时保留现状，不阻塞 UI。
  }
}

function ensureCurrent() {
  if (currentId.value) return
  const id = uuidv4()
  currentId.value = id
  const names = loadNames()
  names[id] = `会话 ${sessions.value.length + 1}`
  persistNames(names)
}

function newSession() {
  abortStream()
  streamOwnerId.value = ''
  const id = uuidv4()
  currentId.value = id
  messages.value = []
  streamText.value = ''
}

// 记录首轮提问作为显示名（一次会话只记一次）。
function rememberName(id: string, firstQuestion: string) {
  const names = loadNames()
  if (!names[id]) names[id] = firstQuestion.slice(0, 12)
  persistNames(names)
}

async function removeSession(s: Session) {
  // 正在为它流式时禁止删（先 abort 也没意义，等这轮结束再说）。兜底：删当前会话前同样切走。
  if (streaming.value) {
    ElMessage.warning('正在回答中，请稍后再删除')
    return
  }
  try {
    await ElMessageBox.confirm(`删除会话「${s.name}」？历史消息将一并清除。`, '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      confirmButtonClass: 'el-button--danger',
    })
  } catch {
    return // 用户取消
  }
  await deleteSession(s.id)
  sessions.value = sessions.value.filter(x => x.id !== s.id)
  // 缓存里的显示名一并清掉，避免删完重进又冒出来。
  const names = loadNames()
  delete names[s.id]
  persistNames(names)
  if (currentId.value === s.id) {
    currentId.value = ''
    messages.value = []
  }
  ElMessage.success('已删除')
}

async function clearAll() {
  if (!sessions.value.length) return
  try {
    await ElMessageBox.confirm(
      `将删除全部 ${sessions.value.length} 个会话及其历史消息，不可恢复。`,
      '清空全部会话',
      { type: 'warning', confirmButtonText: '全部删除', cancelButtonText: '取消', confirmButtonClass: 'el-button--danger' },
    )
  } catch {
    return
  }
  abortStream()
  await deleteAllSessions()
  sessions.value = []
  persistNames({})
  currentId.value = ''
  streamText.value = ''
  messages.value = []
  ElMessage.success('已清空')
}

async function switchSession(id: string) {
  if (id === currentId.value) return
  // 正在为旧会话流式时切走：立刻 abort 那个流，别让它的增量/收尾落到新会话上。
  abortStream()
  currentId.value = id
  streamText.value = ''
  try {
    const h = await getHistory(id)
    messages.value = h.messages.map(m => ({
      role: m.role,
      content: m.content,
      sources: m.role === 'assistant' ? extractSources(m.content) : undefined,
    }))
  } catch {
    messages.value = []
  }
}

function extractSources(content: string): string[] {
  // 后端把来源以「📚 参考来源\n • url」附在回答尾部，这里还原成独立来源条
  const urls: string[] = []
  const re = /^\s*[•-]\s*(https?:[^\s]+)/gm
  for (const m of content.matchAll(re)) urls.push(m[1])
  return urls
}

function scrollToBottom() {
  nextTick(() => {
    if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight
  })
}

// 流式中又发新问题：中止当前这条流，把已打的半截回答固化进消息尾巴并标「已中断」，
// 不丢用户已看到的输入，也不让旧流的增量/收尾污染下面新发的这轮。
// 只中止 fetch + 清 owner，不碰 streaming —— send 接下去会为新流正常置位。
function freezeInterruptedReply() {
  const tail = messages.value[messages.value.length - 1]
  // 固化半截（流式块 renderMarkdown 的 streamText）；空则留一句占位，别出现纯空气泡。
  const half = streamText.value.trim()
  if (half) (tail as Msg).content = half
  else (tail as Msg).content = '（回答已中断）'
  ;(tail as Msg).aborted = true
  kickStream()
}

async function send() {
  const text = draft.value.trim()
  if (!text) return
  // 正在流式又发新问题 → 中断当前流、固化半截，再走正常发送（不再静默忽略）。
  if (streaming.value) freezeInterruptedReply()
  ensureCurrent()
  draft.value = ''

  messages.value.push({ role: "user", content: text })
  // 立即 push 一条空的 assistant 占位，后续往这条里写内容。
  // 不能等流式结束再用「数组最后一个下标」去覆盖 —— 那会覆盖掉刚 push 的 user 消息。
  const pushed: Msg = { role: 'assistant', content: '' }
  messages.value.push(pushed)
  streamText.value = ''
  streaming.value = true

  // 记录当前这轮流的主人会话 + 可 abort 句柄，供切走/新会话时中断在途流。
  const ownerId = currentId.value
  streamOwnerId.value = ownerId
  const controller = new AbortController()
  streamAbort.value = controller
  scrollToBottom()

  try {
    await chatStream(ownerId, text, (evt) => {
      // 切走后旧流的 token 不再渲染（ownerId 已变）。
      if (streamOwnerId.value !== ownerId) return
      const type = evt.type as string
      if (type === 'token') {
        streamText.value += (evt.content as string) ?? ''
        if (streamText.value.length) pushed.content = streamText.value
        scrollToBottom()
      } else if (type === 'meta') {
        // meta 携带最终 answer（可能含参考来源），覆盖增量渲染
        pushed.content = (evt.answer as string) ?? streamText.value
        pushed.sources = (evt.sources as string[]) ?? pushSourcesFrom(streamText.value)
        scrollToBottom()
      } else if (type === 'error') {
        pushed.content = `⚠️ ${evt.message ?? '出错了'}`
      }
    }, enableWeb.value, agentic.value ? 'agentic' : undefined, controller.signal)
  } catch (e) {
    // 主动 abort（切换会话）不是错误：不发红色请求失败横幅。
    if ((e as { name?: string })?.name !== 'AbortError') {
      pushed.content = `⚠️ 请求失败：${(e as Error).message ?? e}`
    }
  } finally {
    // 收尾只对 owner 会话生效：切走了就不去动新会话的列表/messages。
    if (streamOwnerId.value === ownerId) {
      streaming.value = false
      streamAbort.value = null
      // 首轮提问作为会话显示名，并刷新列表（后端这一轮才落了 qa_sessions 行）。
      rememberName(ownerId, text)
      await loadFromBackend()
      scrollToBottom()
    }
  }
}

function pushSourcesFrom(content: string): string[] {
  return extractSources(content)
}

onMounted(async () => {
  // 会话列表以后端 /qa/sessions 为权威：进来先拉一次，旧的 uuidv4 本地假 ID 不再生成。
  await loadFromBackend()
})
</script>

<style scoped>
.qa-page {
  display: flex;
  height: calc(100vh - 56px - 48px);
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}
.qa-side {
  width: 220px;
  border-right: 1px solid #e5e7eb;
  background: #fafbfc;
  display: flex;
  flex-direction: column;
}
.qa-side-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid #e5e7eb;
}
.qa-side-head .title { margin-right: auto; }
.title { font-weight: 600; }
.session-list { flex: 1; overflow-y: auto; padding: 8px; }
.session-item {
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.session-item:hover { background: #f0f4ff; }
.session-item--active { background: #e8f1ff; }
.session-name { font-size: 13px; color: #333; max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-remove {
  display: none;
  font-size: 13px;
  color: #c0392b;
  cursor: pointer;
  flex-shrink: 0;
}
.session-item:hover .session-remove { display: inline-flex; }
.session-remove:hover { color: #e74c3c; }
.session-empty { color: #bbb; font-size: 12px; text-align: center; padding-top: 24px; }

.qa-main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.msg-list { flex: 1; overflow-y: auto; padding: 20px 24px; }
.empty-hint { text-align: center; color: #999; margin-top: 15vh; }
.empty-icon { font-size: 40px; margin-bottom: 8px; }
.sub { font-size: 12px; color: #bbb; }

.msg { display: flex; margin-bottom: 14px; flex-wrap: wrap; }
.msg--user { justify-content: flex-end; }
.msg--assistant { justify-content: flex-start; }
.bubble {
  max-width: 72%;
  padding: 10px 14px;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.65;
  word-break: break-word;
}
.msg--user .bubble { background: #1677ff; color: #fff; }
.msg--assistant .bubble { background: #f4f6fb; color: #1f2329; }
.msg--assistant :deep(pre) {
  background: #1e232a; color: #d7e0ec; padding: 10px; border-radius: 6px;
  overflow-x: auto; font-size: 12px;
}
.cursor { color: #1677ff; animation: blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }
.sources {
  margin-top: 6px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  /* 独占一行（排到气泡下方），避免与气泡横向排布被挤出内容区 */
  width: 100%; min-width: 0; box-sizing: border-box;
}
.sources-label { font-size: 11px; color: #999; flex-shrink: 0; }
.interrupted-tag { margin-top: 6px; font-size: 11px; color: #b26a00; background: #fff3e0; border-radius: 4px; padding: 1px 6px; }
.chip {
  font-size: 11px; color: #1677ff; background: #e8f1ff;
  border-radius: 4px; padding: 1px 6px; text-decoration: none;
  max-width: 100%; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.input-bar { border-top: 1px solid #e5e7eb; padding: 12px; background: #fff; }
.input-row { display: flex; align-items: flex-end; gap: 10px; }
.input-row .el-textarea { flex: 1; }
.web-toggle { margin-bottom: 4px; }
</style>