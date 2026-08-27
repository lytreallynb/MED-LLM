# GRPO 第二轮: 给 RL 制造真实的学习空间

新会话交接文档, 写于 2026-08-26。读完本文档即可接手, 无需其他上下文。
工作目录: ~/MED-LLM。所有代码在 medllm/ 包内。

## 一、现状: 第一轮实验的完整事实

两阶段 post-training 已完成并有完整测量(全部真实, 全部归档):

| 阶段 | 配置 | 结果文件 |
|---|---|---|
| SFT 1.5B | LoRA r=8, q/v_proj, 2 epochs, 894 样本 | results/behavior_base.json, behavior_tuned.json |
| SFT 7B | QLoRA 4-bit NF4, 同配置 | results/behavior_base_7b.json, (tuned 指标在 CHANGELOG) |
| GRPO 1.5B | 从 SFT merge 后继续, 200 步, G=4, lr 1e-5 | results/behavior_sft_grpo.json |

核心数字(179 条 held-out, 按 document_id 防泄漏切分):
- SFT 1.5B: 拒答 recall 0.71->0.98, gold 引用 0.58->0.99, 误拒 0.12->0.02
- SFT 7B: 误拒 0.32->0.03, gold 引用 0.85->1.00 (116/116)
- GRPO vs SFT: 六项指标到小数点后四位一致, disclaimer 0.9385->0.9497 (噪声)

GRPO 训练遥测的关键证据: frac_reward_zero_std 在 0.6~1.0 之间,
grad_norm 频繁为 0, 采样熵 0.5~1.3(很低)。
注意 results/behavior_grpo_INVALID_missing_sft_stack.json 是一次错误评估
(漏 merge SFT), 永不引用; 教训是二阶段 adapter 评估必须用
eval_behavior.py 的 --base-adapter 参数先 merge SFT 再挂 GRPO adapter。

## 二、null result 的根因: 三个 headroom 杀手

1. **GRPO 用了与 SFT 相同的训练 prompt** (data/finetune/train.jsonl 前 600 条)。
   SFT 刚在上面练过 2 epochs, 采样全对, 组内方差为零。
2. **奖励太粗**: score_completion (medllm/grpo_finetune.py) 是 ±1/+0.25 离散档,
   "都对"的 4 个 completion 拿同分。
3. **探索不足**: num_generations=4, SFT 后 policy 峰化, 4 次采样近乎同一答案。

结论: 不是 RL 不适用, 是这个 setup 没给 RL 留作业。第二轮的任务是造作业。

## 三、失败矿: 难题从哪来(全部现成)

- base-7B 在 116 道正例里误拒 37 道 -> results/behavior_base_7b.json 的
  records 里有逐条输出, 这些 prompt 特征就是"难"的定义
- base-1.5B gold 引用错误率 ~42% -> behavior_base.json records
- 已知会导致合理误拒的脏数据模式: drug_name 是供应商名(如 "Medline")
- SFT 从未见过超过 2 段证据的输入(正例=gold+1干扰, 负例=2干扰)

## 四、第二轮实验设计(建议, 新会话可修订)

### 4.1 更难的数据集 (medllm/build_finetune_dataset.py 加参数)

- --n-distractors 3~4: 每题 4~5 段证据, 分布外, SFT 必出错
- 语义相近干扰: 干扰段优先选同 section 类型的其他药(如都取 warnings 段),
  而不是随机段 -- 引用难度大幅上升
- 难负例: 负例的干扰段与问题药名同前缀/同类, 拒答判断变难
- 生成 data/finetune/train_hard.jsonl + test_hard.jsonl,
  同样按 document_id 切分, 且与第一轮 test 的文档不重叠

### 4.2 更细的奖励 (medllm/grpo_finetune.py 的 score_completion)

- 引用部分分: 引对 gold +1.0, 引了有效但非 gold 的 -0.3(而非只看命中)
- 简洁惩罚: 超长回答 -0.1/百token 之类的轻惩罚, 制造"都对"里的方差
- 声明位置: 结尾处 +0.25, 中间 +0.1
- 保留误拒 -1.0 的不对称设计(防坍缩到全拒答), 这是第一轮验证过的

### 4.3 探索配置

- num_generations 8, temperature 1.0~1.2 (GRPOConfig 支持 temperature 参数,
  注意用 inspect 签名过滤的现有模式, trl 版本间参数漂移)
- max_completion_length 从 140 提到 200 (第一轮遥测显示 clipped_ratio
  常年 0.95~1.0, 完成被截断, 可能压制了 disclaimer 相关方差)

### 4.4 评估协议

- 同时在 test.jsonl(旧,易) 和 test_hard.jsonl(新,难) 上评
- 对比三方: SFT / SFT+GRPO(第一轮旧 adapter) / SFT+GRPO-v2
- 预注册预期: 易集上三者应持平(回归检查), 难集上 GRPO-v2 应超过 SFT;
  若仍持平, 报告为真实发现, 不得改口径

## 五、Kaggle 工作流(踩过的坑全在这, 照做能省半天)

1. 依赖必须钉版本, 否则两个已知炸点:
   pip -q install "transformers==4.57.6" "huggingface_hub<1.0" peft accelerate bitsandbytes trl
   (Kaggle 预装 transformers 5.x, warmup_ratio 会 TypeError;
   trl 用 pip 解析到的兼容版即可, grpo_finetune.py 已有签名过滤防参数漂移)
2. kaggle CLI 已配好: export KAGGLE_API_TOKEN=$(cat ~/.kaggle/token)
   数据集: kaggle datasets 下的 yutonglyu/medllm, 更新用 datasets version;
   zip 上传会被自动解压成同名子目录
3. notebook: yutonglyu/notebookaa713825ac, 用 kaggle kernels push 走无头
   commit(Save & Run All 等价), kernels status 轮询, kernels output 取结果
4. GPU T4 x2, 单会话 12h; GRPO 200 步(G=4)约 42 分钟, G=8 预计翻倍
5. 训练完立刻把 adapter zip 进输出目录再跑评估(断线保险)

## 六、红线(不可协商, 来自 ~/job/CLAUDE.md 和用户死命令)

- 任何数字只有真实跑出来才能进 README/简历; null result 照实写
- 不准把 Claude 加进任何仓库 contributor; commit 不带 Co-Authored-By,
  署名只有 Yutong; Claude 是助手, 认领/提交/署名主体是 Yutong
- 面试准备优先于本实验; 本实验上限一个专注周末
- 简历 v10 已定稿发出(bullet 3 挂 SFT 数字), 本实验成功前不改简历;
  成功后 bullet 3 才能升级为 Base->SFT->GRPO 三段式

## 七、成功标准与停止条件

成功: 难集上 SFT+GRPO-v2 相对 SFT 有可复述的行为提升(方向一致且幅度
超出单次运行噪声), 且易集无回归。
停止: 超过一个周末; 或两次配置迭代后难集仍完全持平(此时把"为什么这个
任务对 RL 免疫"写成分析, 同样是有价值的产出); 或挤占面试准备。
