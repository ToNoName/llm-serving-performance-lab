# 验证记录

日期：2026-09-09。历史性能数据与当前代码可运行性分开验收。

## 本次检查

- E1–E3 精简 CSV 的 19 行字段与历史汇总一致；E4 的 5 组延迟、日志请求窗口及重新计算的吞吐一致。
- README 与性能报告中的 158 个表格数值按显示精度核对通过；核心成果中的 Queue 秒数和 TPOT 端点值对应同一 CSV，配置值见证据映射。
- Python 使用 AST 解析，shell 使用 bash -n 检查；这些检查不等于 GPU 实验运行。
- Docker 的 COMPOSE_* 后端变量与直接运行 Gateway 的变量分开；复制示例环境文件后检查配置展开，容器地址保持为服务名。
- 源码阅读版本恢复为原稿的 0.21.0；不宣称逐函数验证了实验版本 0.22.1。

## 实际部署验证

从临时干净文件副本复制 `.env.example`，仅修改模型路径、后端模式、密码和隔离端口；不覆盖 COMPOSE_* 后端地址。实际验证通过：

- Gateway 镜像重新构建，Nginx → Gateway → llama.cpp 非流式与流式请求成功，流式收到 [DONE]。
- llama.cpp 以 LLAMA_GPU_LAYERS=0 加载现有 Llama 2 7B GGUF；仍使用 CUDA 镜像和 GPU 设备预约，因此本测试不是纯 CPU 主机兼容性证明。
- Gateway 后端健康检查、Prometheus Gateway 抓取、Grafana 健康、数据源和仪表盘加载通过。
- 测试结束后专用容器、网络和测试匿名卷已清理；已有项目未修改。
- 24 个本地 Markdown 链接及章节锚点检查通过，6 个 PNG 文件签名有效。

本次 llama.cpp 本地镜像标识：`ghcr.io/ggml-org/llama.cpp@sha256:c371aa937f733c5098ebbb8bb02a2f88bc313d5e8bd6f24a797170b115ce0446`。这是本次环境记录，未冒充历史性能环境。

## 尚未验证及限制

- 新实验脚本未在 AutoDL 4090D 上完整重跑 E1–E4；vLLM 缓存与分块开关已显式设置，有效 token budget 依据历史分析固定为 2048，仍需远端核对最终启动配置。
- Docker 参考编排不等同于 E1–E4 的实验启动 profile；性能实验使用 scripts/start_vllm.sh。
- 本地 llama 测试使用 Llama 2 7B GGUF，并非历史 Qwen AWQ 性能环境；不能据此验证 vLLM 的性能数值。
- llama.cpp full-cuda 为浮动标签，requirements 使用部分版本范围。没有历史完整依赖锁文件，不伪造精确锁定环境。
- 本地 Markdown 目标/锚点检查不等于 GitHub Mermaid 浏览器渲染验收；发布页面需另行确认。
