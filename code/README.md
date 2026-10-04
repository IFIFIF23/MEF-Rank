# MEF-Rank：训练与配置指南

[项目首页](../README.md) · [中文概览](../README.zh-CN.md) · [数据说明](../docs/DATA.md)

本包包含 STR、iTransformer、TimeMixer、ModernTCN、SOFTS、TimeXer、TimeBridge、TimeFilter、SRSNet、ChannelTokenFormer。每个主干有 `models/<模型>/off.py` 和 `on.py` 两个入口，模型核心及训练代码共享。

**全部训练使用原重排 1 号数据。Off/On 共用同一份最终预测器权重。** Off 直接预测；On 用历史经验生成七维动作，再通过同一模型完成第二次预测。验证/测试保持原观察视图，不将 0 号当成另一套 Off 训练数据。

本次新增了增强链连接代码。根据委托方要求，没有运行新增完整链的训练验证或五市场正式训练；不能把“代码已交付”写成“整链实验已验收/已有有效性结果”。

## 1. 接收哪些文件

```text
代码文件总结/
  code/                  本目录：代码、配置、使用说明
  data_encrypted/
    gem/                 创业板
    main/                A股主板
    nasdaq100/           纳斯达克100
    sp500/               标普500
    crypto/              Crypto-15
```

每个数据包包含 `1_str`、`1_memory` 和 `evaluation` 三个视图，以及加密的 target/source 日期映射。evaluation 中的 2022 数据只作为历史窗口上下文；正式指标从原 2023 验证区间开始。不要自行更换为顺序 1 号数据。

另向数据所有者单独取得所选市场的二进制 `.key` 文件，保存在交付目录以外，例如 `/secure/keys/gem.key`。不同市场使用不同密钥。不要将密钥上传 Git、放进代码压缩包或写进 Python 文件。

## 2. 一条命令启动

Linux、Python 3.10/3.11、可用的 `python3-venv` 和 Git；首次启动需要联网安装依赖及部分官方源码。

进入本目录后运行：

```bash
./run.sh --dataset gem --model STR --key-file /secure/keys/gem.key --device cuda
```

这条命令会准备本地 `.venv`、补齐固定版本的上游源码，然后执行：

1. 读取并解密 1 号 STR 视图，在训练窗内选 epoch，重新拟合最终预测器。
2. 读取 1 号 Memory 视图，按季度训练五年滚动历史预测器。
3. 在标签成熟后执行嵌套 Direction/Magnitude/Outer Writer，保存合格和拒绝记录。
4. 构建同主干、同市场、同 seed 的 Memory，按 Episode 校准 CAL1。
5. 固定最终预测器、Memory 和 CAL1，输出共享权重的 Off/On 预测及指标。

默认会运行完整链，历史模型和 Writer 的计算量明显大于只训练一个主模型；不会重复训练两份最终模型。

已有自己的环境时，可直接运行：

```bash
python -m pip install -r requirements.txt
python setup_sources.py
python train.py --dataset nasdaq100 --model iTransformer --key-file /secure/keys/nasdaq100.key --device cuda
```

已有 PyTorch/CUDA 环境也可以指定 Python：

```bash
MEF_PYTHON=/path/to/your/python ./run.sh --dataset crypto --model ModernTCN --key-file /secure/keys/crypto.key --device cuda
```

该环境须已安装 requirements 中的依赖。没有 GPU 可用 `--device cpu`；默认 `--device auto` 自动选择。

首次安装若 PyTorch 与本机 GPU 驱动不匹配，请安装与机器适配的 PyTorch 2.6 CUDA 构建，再使用上述已有环境方式。导入源码及运行均不依赖原 `paper/试验区2` 路径。

## 3. 如何选择模型和数据

| 参数 | 可选值 |
| --- | --- |
| `--dataset` | `gem`、`main`、`nasdaq100`、`sp500`、`crypto` |
| `--model` | `STR`、`iTransformer`、`TimeMixer`、`ModernTCN`、`SOFTS`、`TimeXer`、`TimeBridge`、`TimeFilter`、`SRSNet`、`ChannelTokenFormer` |
| `--mode` | `both`（默认）、`on`、`off`；`both/on` 均保存成对 Off/On |
| `--seed` | 默认 `20260921`；原计划另有 `20260922`、`20260923` |
| `--data-dir` | 默认 `../data_encrypted`，可以改成数据盘路径 |
| `--output-dir` | 默认本目录 `outputs` |
| `--epochs` | 可选，覆盖最终及历史预测器的训练上限；改变后属于不同运行配置 |
| `--threads` | 默认 4 个 CPU 线程 |

训练时间/主干参数在 `config/training_v001.json`。默认 H5、60 步、125 维输入、50 epoch 上限、训练窗末 126 个合格日选 epoch、每 5 日一个训练日期。仅标签在对应训练截止前已成熟的样本参与损失。

仅训练普通 Off 路径：

```bash
python train.py --dataset gem --model STR --mode off --key-file /secure/keys/gem.key --device cuda
```

也可使用独立入口：

```bash
python models/STR/off.py --dataset gem --key-file /secure/keys/gem.key --device cuda
python models/STR/on.py --dataset gem --key-file /secure/keys/gem.key --device cuda
```

**做 Off/On 对比优先使用默认 `--mode both`。** 两条独立命令若没有 `--resume` 会创建两个新 run；不要把独立运行的权重当作同一份。

如已完成 Off，可使用同一个 run 路径补齐 On，而不重训最终预测器：

```bash
python train.py --dataset gem --model STR --mode on --key-file /secure/keys/gem.key --device cuda --resume outputs/gem/STR/seed_20260921/实际运行目录
```

恢复时保留原数据、代码、模型、seed、epochs 等训练选项；身份不一致会拒绝覆盖。已完成季度/日度 Writer 会跳过，未完成的训练从保存的 epoch 状态继续。

## 4. 输出与改进位置

每次结果位于 `outputs/<数据>/<模型>/seed_<seed>/<UTC时间_随机后缀>/`：

- `run_manifest.json`：代码、配置、数据身份及设备信息。
- `main/model.pt`：Off/On 共用权重及训练期 scaler。
- `history/`：季度权重、内部训练指标、每日 Writer 记录与拒绝原因。
- `memory.json`、`calibration.json`：历史经验和校准结果。
- `predictions.parquet`：`trade_date, ts_code, arm, y_pred, label_H5__primary_return`；`ts_code` 为化名。
- `metrics.json`、`daily_metrics.csv`：分验证/测试的 RankIC、配对差、NW(10) t、APPLY 数量及回退原因。
- `status.json`：进行中、失败或完成；`artifact_hashes.json`：产物哈希。

数据不足以形成 CAL1 所需的 20 个独立季度时，On 保留 Off 原预测，并明确记录原因。一个季度内可使用很多日度记录，但它们仍然属于同一个 Episode。不会复制样本来充当独立历史。

改进主干适配看 `src/mef_unified/adapters.py`；统一 Writer/Memory/CAL1 看 `src/mef_delivery/`；训练和流水线看 `handoff/`。`vendor/mef_rank/` 是来源可追溯的共享数学原语。旧的错误输出侧增强分支、旧训练结果、旧权重和原始行情未包含在交付中。

## 5. 数据与结果的含义

数据以 Arrow IPC 序列化、压缩并用 AES-256-GCM 加密；训练在内存中解密，不生成明文数据缓存。已去除直接证券标识、名称、证券元数据和原始行情字段，保留模型所需特征、标签、日期、mask 和市场状态；同一市场所有视图使用一致化名。

这属于去标识化和存储/传输加密，**不保证数值不可还原或无法重新识别**。有密钥且能本地训练的人能够提取解密后的数值；训练产物也含预测和标签。密钥单独交付的保护边界已由数据所有者确认。

原重排 1 号预先包含部分后期源观察，因此结果属于“先见后纠正诊断”，不能写成未知未来的独立样本外收益。本包指标是排序预测指标，不是已经验证可成交且扣费的策略回测。

小截面 profile 与原生规则的差异见 [实现说明](IMPLEMENTATION.md)。这些新连接和适配未经过本次完整训练验证；接收方应保留失败记录及回退状态，不把全部回退误写成纠错有效。

## 6. 官方源码

各外部主干的固定 commit 与来源在 `third_party/model_sources.lock.yaml`。Git 仓库不收录下载生成的 `third_party/sources/`，首次运行由 `setup_sources.py` 从各官方仓库获取固定版本；本地准备目录已包含六个允许分发的最小源码子集及其许可证。TimeXer、TimeFilter、ChannelTokenFormer 的来源登记尚未明确再分发许可，单独打包时也应排除这三个上游源码。下载后校验逐文件哈希，不默默升级。详见[第三方来源说明](../THIRD_PARTY_NOTICES.md)。

只复制十个主干核心及必要依赖。SRSNet 直接加载其核心模块，避免上游 benchmark 的包初始化把其他模型全集带进交付。
