# backend/core/exceptions.py
# ResearchAgent 统一异常体系：所有自定义异常都继承同一个基类，
# 便于「一次捕获全部」以及按「可重试 / 不可重试」分类处理。


class ResearchError(Exception):
    """所有 ResearchAgent 自定义异常的基类（继承 Python 内置的 Exception）。
    比普通异常多带两样上下文：是哪个 Agent 出的错、以及任意细节字典。"""
    def __init__(self, message: str, agent_type: str = "", details: dict = None):
        # message：错误描述文字；agent_type：哪个 Agent（collect/analyze/retrieve/valuation/risk）；details：额外细节
        super().__init__(message)          # 调用父类 Exception 的初始化，把错误消息存好
        self.agent_type = agent_type       # 记录出错的 Agent 类型，方便日志与排查
        self.details = details or {}       # 记录细节字典；传 None 时用空字典 {} 兜底，避免后续访问报错


class LLMAPIError(ResearchError):
    """大模型 API 调用失败（超时 / 限流 / 网络错误）。属于【可重试】异常。"""
    pass                                   # 无需额外逻辑，直接继承基类的能力即可


class AgentExecutionError(ResearchError):
    """Agent 业务逻辑执行失败。"""
    pass


class PipelineError(ResearchError):
    """多 Agent Pipeline（流水线编排）失败。"""
    pass


class IntentRouteError(ResearchError):
    """意图识别路由失败（没判断出该交给哪个 Agent）。"""
    pass


class MilvusConnectionError(ResearchError):
    """Milvus 向量库连接失败。属于【可重试】异常。"""
    pass


class FileParseError(ResearchError):
    """文件解析失败（Word / PDF）。"""
    pass


class InvalidInputError(ResearchError):
    """用户输入不合法。属于【不可重试】异常（重试也不会变合法）。"""
    pass


class AuthenticationError(ResearchError):
    """认证失败。属于【不可重试】异常。"""
    pass


class TaskBudgetExceeded(ResearchError):
    """单研报任务累计 token 超预算。属于【必须阻断】(BLOCKING_ERRORS)。

    由 usage_ctx.enforce_budget 在每次 LLM 调用前查出并抛出：不重试、不降级、
    直接上抛，让流水线走到 _fail_task 把任务置 failed。作用范围只限研报任务
    (带 task_id 的 context)；问答(thread_id)与离线调用不受限。
    """
    pass


class NonDegradableError(ResearchError):
    """抛它的错误【不得被降级】—— 必须原样上抛，让任务停下来。

    与「不可重试」是两件事，虽然 retry.py 对两者的处置恰好相同（都不重试、
    都不降级、直接上抛）。区别在语义与用途：
      - 不可重试（InvalidInputError / AuthenticationError）：重试也不会变好；
      - 不可降级（本类）：重试也许有用，但【即使重试彻底失败也不能降级交付】。

    典型场景：采集四源全败（CollectExhaustedError）。降级会把它变成一份空结果，
    任务带着零条数据继续跑到分析阶段，最后产出一份「因数据不足无法评级」的
    研报 —— 错误归因从「采集失败」变成「分析失败」，排查会绕远路。
    风控失败本该是同一类，但它抛的是通用的 LLMAPIError（可重试），
    判不出来，所以风控在 retry.py 里另按 agent_type 拦截。
    """
    pass
if __name__ == '__main__':
    # 抛出一个带上下文的异常，并用基类捕获
    try:
        raise LLMAPIError("DeepSeek 超时", agent_type="collect", details={"timeout": 30})
    except Exception as e:
        print("捕获:", type(e).__name__, "| msg:", e, "| agent:", e.agent_type, "| details:", e.details)

    print("LLMAPIError 是 ResearchError 子类:", issubclass(LLMAPIError, ResearchError))