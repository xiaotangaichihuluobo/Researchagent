# backend/core/timeutil.py
# 全仓唯一的时间戳时区/取值入口：一律北京时区（Asia/Shanghai, UTC+8）。
#
# 为什么收敛到这一处：早期代码散落 `datetime.now(timezone.utc)` / naive `datetime.now()`，
# 写到 DB 里就带 +00:00，而在本仓的业务语境里（中文研报、数据源均为中国）统一记成
# 北京时间更直观——界面与 DB 展示都少一层"减 8 小时"的心算。
#
# 用法：
#   from backend.core.timeutil import cn_now
#   created_at = cn_now()          # 当前北京时间的 aware datetime（tzinfo=Asia/Shanghai）
#
# 注：data-parse 侧把"无时区的时间戳"按数据源约定归一化（多为 UTC）属于【数据语义】，
# 不走本函数——它表达的是"这条数据官方宣布的时刻"，与"现在几点"是两码事。

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# 全仓唯一时区常量。UTC+8，无夏令时。
CN_TZ = ZoneInfo("Asia/Shanghai")

# UTC 别名，避免各模块再 import timezone。
UTC = timezone.utc


def cn_now() -> datetime:
    """当前时刻的北京时区 aware datetime。

    :return: tzinfo=Asia/Shanghai 的当前时间；写入 TIMESTAMPTZ 列时自带 +08:00 偏移。
    """
    return datetime.now(CN_TZ)