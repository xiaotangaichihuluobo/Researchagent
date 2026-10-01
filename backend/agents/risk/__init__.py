# backend/agents/risk/__init__.py
"""⑤ 风控复核 Agent：强制 HitL，未签字不得发布。

去 LangGraph 后闸门改成「落草稿 → 翻待审 → 写 checkpoint + 返回哨兵」：
引擎据此把首跑收口为暂停，resume 时注入签字前进式续跑（见 steps/runner）。"""
