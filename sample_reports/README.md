# sample_reports —— 示例研报语料

按 `scripts/seed_data.py` 里 seed 的公司清单生成,每家一份 .md,覆盖 11 个行业 12 家公司。
文件名即 `报告key`:`<代码>_<公司>_<行业>.md`。供导入研报语料库(切块 → BGE-M3 嵌入 → 写 Milvus → 登记 PG)使用。

## 清单

| 文件 | 公司 | 代码 | 行业 |
|---|---|---|---|
| `600519_贵州茅台_白酒.md` | 贵州茅台 | 600519.SH | 白酒 |
| `300750_宁德时代_动力电池.md` | 宁德时代 | 300750.SZ | 动力电池 |
| `000858_五粮液_白酒.md` | 五粮液 | 000858.SZ | 白酒 |
| `600036_招商银行_银行.md` | 招商银行 | 600036.SH | 银行 |
| `600900_长江电力_电力.md` | 长江电力 | 600900.SH | 电力 |
| `000333_美的集团_家用电器.md` | 美的集团 | 000333.SZ | 家用电器 |
| `600276_恒瑞医药_化学制药.md` | 恒瑞医药 | 600276.SH | 化学制药 |
| `002594_比亚迪_新能源汽车.md` | 比亚迪 | 002594.SZ | 新能源汽车 |
| `600887_伊利股份_食品饮料.md` | 伊利股份 | 600887.SH | 食品饮料 |
| `601899_紫金矿业_有色金属.md` | 紫金矿业 | 601899.SH | 有色金属 |
| `601012_隆基绿能_光伏.md` | 隆基绿能 | 601012.SH | 光伏 |
| `002714_牧原股份_养殖.md` | 牧原股份 | 002714.SZ | 养殖 |

## 导入方式

库函数:`backend/agents/qa/... 同 scripts/build_knowledge_base.build_report_pipeline`。

本地先把单份拷进后端容器,再跑导入(示例):
```bash
docker cp sample_reports/600519_贵州茅台_白酒.md research_agent_backend:/app/report.md
docker compose exec backend python -c "
import asyncio
from scripts.build_knowledge_base import build_report_pipeline
asyncio.run(build_report_pipeline('/app/report.md', company_code='600519.SH', industry='白酒'))
"
```
验证:`SELECT count(*) FROM report_corpus;` 不再为 0。

> 说明:内容为示例性质的虚构研报数据,非真实财务数据,仅供检索/演示流程走通。