# ResearchAgent 前端镜像：node 构建静态产物 → nginx 托管 + 同源反代后端
# 构建：docker build -t researchagent-frontend:latest ./frontend
FROM node:20-alpine AS build
WORKDIR /app

# 构建期注入的 API 基址：同源（经 nginx 反代）留空；如需直连后端再 --build-arg 覆盖
# （Vite 读环境变量优先级：shell 环境 > .env.*，故此处 ENV 会覆盖 frontend/.env.local 的 localhost:8000）
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]