#!/usr/bin/env bash
# sample_reports/import_sample_reports.sh
# 一键导入 sample_reports/ 下全部分示例研报(切块→嵌入→写 Milvus→登记 PG)。
#
# 用法:把 sample_reports 整个目录 scp 到服务器 /root/researchagent 后,在服务器上:
#   cd /root/researchagent && bash sample_reports/import_sample_reports.sh
#
# 依赖:backend 容器在跑;sample_reports/*.md 已在宿主机。文件名即来源
# (代码_公司_行业.md),脚本从文件名解析 company_code / industry。

set -e
ENV_FILE="/root/researchagent/.env.prod"
CT="research_agent_backend"
DIR="$PWD/sample_reports"

# 校验环境
echo "==> 校验 backend 容器..."
docker inspect "$CT" >/dev/null 2>&1 || { echo "找不到后端容器 $CT,请先 up -d"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "没有 $ENV_FILE,请确认目录"; exit 1; }

# 收集待导入研报(排除 README)
shopt -s nullglob
files=()
for f in "$DIR"/*.md; do
  [ "$(basename "$f")" != "README.md" ] && files+=("$f")
done
if [ "${#files[@]}" -eq 0 ]; then
  echo "sample_reports/ 下没有 .md(别忘了先把 sample_reports scp 上来)"; exit 1
fi
echo "==> 共 ${#files[@]} 份待导入。先统一拷进容器... 已存在集合将按 report_key 覆盖(幂等)。"
for f in "${files[@]}"; do
  echo "    - $(basename "$f")"
done

for f in "${files[@]}"; do
  base=$(basename "$f" .md)
  code=$(echo "$base" | cut -d_ -f1)
  company=$(echo "$base" | cut -d_ -f2)
  industry=$(echo "$base" | cut -d_ -f3-)

  echo ""
  echo ">>> 导入: $company（$code）[行业=$industry]"
  docker cp "$f" "$CT":/app/_import_report.md
  docker compose --env-file "$ENV_FILE" exec -T backend python -c "
import asyncio
from scripts.build_knowledge_base import build_report_pipeline
asyncio.run(build_report_pipeline(
    '/app/_import_report.md',
    company_code='$code',
    industry='$industry',
))
"
done

echo ""
echo "============================================"
echo " 全部导入完成 ✅  验证登记数:"
echo "============================================"
docker compose --env-file "$ENV_FILE" exec postgres psql \
  -U researchagent_user -d researchagent \
  -c "SELECT count(*) FROM report_corpus;"