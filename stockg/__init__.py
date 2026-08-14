"""A 股实时行情 + DeepSeek 大模型分析的 DDD 分层实现。

分层:
- domain        : 领域实体与端口(接口), 不依赖任何外部框架
- application   : 用例编排 (应用服务)
- infrastructure: 对外系统的具体适配器 (akshare/东财, DeepSeek)
- interfaces    : 对外入口 (CLI)
"""
