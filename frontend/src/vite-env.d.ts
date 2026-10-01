/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API 基地址，用于 SSE 直连（不经过 Vite proxy） */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
