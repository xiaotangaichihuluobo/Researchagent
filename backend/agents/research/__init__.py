# backend/agents/research/__init__.py
"""投研编排域：把五个阶段（采集/分析/检索/估值/风控）组装成一条带风控签字与驳回回边的
研究流水线。去 LangGraph 后由 runner.py（表驱动引擎）+ steps.py（段执行器）+ protocols.py
（handoff 契约）共同取代原父图。"""
