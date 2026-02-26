# Snake RL Showcase (Miniconda)

目标：做一个可训练、可展示、AI 玩得很强、可直接发布到 GitHub 的贪吃蛇成品版本。

## 1) 创建 conda 环境

```bash
cd /Users/sota/code/script/games/snake_rl
conda env create -f environment.yml
```

或：

```bash
conda create -y -n snake-rl python=3.11
```

## 2) 训练强模型

先训练一轮基础模型：

```bash
conda run -n snake-rl python train_q_learning.py \
  --episodes 12000 \
  --width 12 --height 12 \
  --alpha 0.12 --gamma 0.97 \
  --epsilon-start 1.0 --epsilon-end 0.02 \
  --output checkpoints/q_table.json
```

继续增量训练（推荐再跑 20000+）：

```bash
conda run -n snake-rl python train_q_learning.py \
  --episodes 20000 \
  --width 12 --height 12 \
  --alpha 0.08 --gamma 0.97 \
  --epsilon-start 0.2 --epsilon-end 0.01 \
  --resume \
  --output checkpoints/q_table.json
```

## 3) 评估（无界面）

```bash
conda run -n snake-rl python eval.py --model checkpoints/q_table.json --episodes 300 --policy hybrid
```

- `--policy hybrid`：RL + 安全规划（展示推荐）
- `--policy rl`：纯 RL 策略（对比用）

## 4) 成品展示版（GUI）

```bash
conda run -n snake-rl python play.py --model checkpoints/q_table.json --policy hybrid --speed-ms 55
```

控制键：
- `Space`：暂停/继续
- `R`：重开
- `M`：切换 `HYBRID / RL`
- `+ / -`：调节速度

## 5) 目录说明

- `snake_env.py`：贪吃蛇环境与奖励函数
- `train_q_learning.py`：Q-Learning 训练器
- `agent.py`：混合智能体（RL + 安全规划）
- `eval.py`：命令行评估
- `play.py`：展示版自动游玩
- `manual_play.py`：手动游玩

## 6) 当前策略设计

- 训练：Tabular Q-Learning（11 维状态，3 个相对动作）
- 展示默认策略：Hybrid
  - 利用 Q 值偏好
  - 同时做安全空间评估与路径可达性判断
  - 显著降低“贴边自杀/困死”概率
