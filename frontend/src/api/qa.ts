import client from './client'

export interface ChatResult {
  session_id: string
  answer: string
  answer_mode: 'rag' | 'llm_direct' | 'general' | 'web' | 'agentic'
  confidence: number
  sources: string[]
}

export interface HistoryResult {
  session_id: string
  messages: { role: 'user' | 'assistant'; content: string; sources?: string[] }[]
  summary: string | null
  total_turns: number
}

/** 非流式问答：跑完一整轮返回最终答案。engineMode 缺省=听后端配置（默认 pipeline）。 */
export async function chat(
  sessionId: string,
  message: string,
  enableWebSearch = false,
  engineMode?: 'pipeline' | 'agentic',
): Promise<ChatResult> {
  const { data } = await client.post<ChatResult>('/qa/chat', {
    session_id: sessionId,
    message,
    enable_web_search: enableWebSearch,
    engine_mode: engineMode,
  })
  return data
}

/** 流式问答（SSE）。onEvent 收到 {type, ...} 帧：progress / token / meta / done / error。
 *  engineMode 缺省=undefined（听后端配置）；agentic 走多轮自搜 + 命中 enableWebSearch 时联网。 */
export async function chatStream(
  sessionId: string,
  message: string,
  onEvent: (evt: Record<string, unknown>) => void,
  enableWebSearch = false,
  engineMode?: 'pipeline' | 'agentic',
  signal?: AbortSignal,
): Promise<void> {
  const token = localStorage.getItem('research-agent-token')
  const res = await fetch(
    `${(import.meta.env.VITE_API_BASE_URL ?? '')}/api/v1/qa/chat/stream`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        enable_web_search: enableWebSearch,
        engine_mode: engineMode,
      }),
      signal,
    },
  )
  if (!res.ok || !res.body) {
    throw new Error(`流式请求失败：HTTP ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))
  let buffer = ''
  // 相邻两个 token 的最小间隔（毫秒）。若不在 token 帧之间让出宏任务，
  // Vue 会把多次 streamText 更新批合成一帧，页面就「一次性全出」而没有打字机节奏。
  const TOKEN_INTERVAL_MS = 24
  // meta（最终答案）之前的 token 加速，避免读完一个长答案要等太久：
  // meta 一到就以完整 answer 收尾，前面的等待只用来看节奏，不值得慢。
  let sawMeta = false

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    // 先把 CRLF 规整成 LF：sse-starlette 发的是 \r\n\r\n，若按 \n\n 找帧边界，
    // 在 \r\n\r\n 里永远不会出现连续 \n\n，解析器就一帧都切不开，页面空白。
    // 规整后统一按 \n\n 切，同时兼容纯 \n 的分隔。
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    // 逐条解析 SSE 帧：data: {json}\n\n
    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      if (!raw.startsWith('data:')) continue
      const payload = raw.slice(5).trim()
      if (!payload) continue
      let evt: Record<string, unknown>
      try {
        evt = JSON.parse(payload)
      } catch {
        continue
      }
      onEvent(evt)
      if (evt.type === 'meta') sawMeta = true
      // 仅对 text token 且还没到 meta 时做节奏节流；其余帧（progress/meta/done）不减速
      if (evt.type === 'token' && !sawMeta) await sleep(TOKEN_INTERVAL_MS)
    }
  }
}

export interface SessionSummary {
  session_id: string
  summary: string | null
  updated_at: string | null
  turns: number
}

/** 读取会话历史（唯一真相来源 qa_messages）。 */
export async function getHistory(sessionId: string): Promise<HistoryResult> {
  const { data } = await client.get<HistoryResult>(`/qa/sessions/${sessionId}/history`)
  return data
}

/** 列出当前用户的历史会话（会话侧栏导航用）。 */
export async function listSessions(): Promise<SessionSummary[]> {
  const { data } = await client.get<SessionSummary[]>('/qa/sessions')
  return data
}

/** 删单个会话；返回后端确认删除的行数。 */
export async function deleteSession(sessionId: string): Promise<number> {
  const { data } = await client.delete<{ deleted: number }>(`/qa/sessions/${sessionId}`)
  return data.deleted
}

/** 删当前用户全部会话；返回删除的会话行数。 */
export async function deleteAllSessions(): Promise<number> {
  const { data } = await client.delete<{ deleted: number }>('/qa/sessions')
  return data.deleted
}