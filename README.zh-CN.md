# MEF-Rank

**MEF-Rank: Model-Aware Action Memory for Financial Ranking** 论文配套代码。

作者：Yifan Yuan、Yuanjian Zhang、Xiang Xu、Chuanchang Liu。单位：北京邮电大学计算机学院。

[论文原文](main.pdf) · [English](README.md) · [完整运行与配置指南](code/README.md) · [数据说明](docs/DATA.md) · [实现说明](code/IMPLEMENTATION.md)

## 项目概览

本仓库提供十个模型主干、五个市场数据集的统一训练入口，以及共享最终预测器权重的 Off/On 对比：Off 直接预测，On 根据历史经验生成动作并通过同一主干再次预测；支持不足时保留 Off 结果。两种设置均使用原重排 **1 号数据**。

支持的主干为 STR、iTransformer、TimeMixer、ModernTCN、SOFTS、TimeXer、TimeBridge、TimeFilter、SRSNet、ChannelTokenFormer。数据集参数为 `gem`、`main`、`nasdaq100`、`sp500`、`crypto`。

## 开始训练

准备 Linux、Python 3.10/3.11（含 venv）、Git，以及首次安装所需的网络连接。向数据所有者取得加密数据和对应密钥，将数据放入 `data_encrypted/<数据集>/`，密钥保存在仓库外。

在仓库根目录运行：

```bash
bash code/run.sh --dataset gem --model STR --key-file /secure/keys/gem.key --device cuda
```

请替换密钥路径。该命令会准备环境、安装固定版本的上游源码，并启动完整训练与配对评估；默认模式为 `both`。CPU 可使用 `--device cpu`。安装已有环境、单独运行 Off、断点恢复和全部参数见[使用指南](code/README.md)。

## 文件与结果

`code/` 保存模型、配置和启动入口；`docs/` 保存数据说明。结果默认写入 `code/outputs/`，包含运行清单、权重、历史经验、校准文件、预测及指标。目录结构见[英文首页](README.md#repository-layout)。

本地加密数据、密钥、运行结果、环境和交付压缩包不进入 Git。公开代码需要配合获授权的数据使用；密钥单独提供。[数据说明](docs/DATA.md)解释了各视图和保密边界。

## 实现状态

新增完整增强链尚未进行端到端训练验证；本交付不包含已核验的论文数值复现结果。具体工程适配见[实现说明](code/IMPLEMENTATION.md)。原重排数据含后期源观察，相关结果属于诊断设置，不能表述为独立样本外表现；排序指标也不等同于扣费交易回测。

论文方法、实验口径和报告结果见[提交稿](main.pdf)，引用元信息见 [CITATION.cff](CITATION.cff)。当前按已投稿论文标注，未填写尚未提供的录用会议、DOI 或正式出版年份。项目许可证尚待选定，上游组件遵循各自许可证，详见[第三方来源说明](THIRD_PARTY_NOTICES.md)。
