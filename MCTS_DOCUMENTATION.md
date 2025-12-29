# MCTS + PUCT 算法实现文档

## 概述

本实现提供了一个基于 **PUCT（多项式置信上界）** 的蒙特卡洛树搜索（MCTS）Agent，用于五子棋游戏决策。

## 核心算法

### 1. MCTS 四个阶段

#### Selection（选择）
- 从根节点开始，使用 PUCT 公式递归选择最优子节点
- PUCT 公式平衡了**利用（exploitation）**和**探索（exploration）**

```
PUCT(s,a) = Q(s,a)/N(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
           ↑                   ↑
      利用项（胜率）      探索项（先验+访问）
```

其中：
- `Q(s,a)/N(s,a)`：该动作的平均价值（胜率）
- `P(s,a)`：该动作的先验概率
- `N(s)`：父节点的访问次数
- `N(s,a)`：该动作的访问次数
- `c_puct`：探索系数（通常 1.0-2.0）

#### Expansion（扩展）
- 当到达未完全扩展的节点时，展开所有子节点
- 使用 `recommand_positions()` 生成候选落子点
- 为了效率，限制候选数量（最多10个）

#### Simulation（模拟）
- 从新扩展的节点开始，使用启发式策略进行快速游戏模拟
- 不使用完全随机走棋，而是使用：
  - 优先防守对手的活4、活3
  - 优先创建自己的活4、活3
  - 快速评估函数 `_quick_evaluate()`

#### Backup（反向传播）
- 将模拟结果反向传播到路径上的所有节点
- **重要**：价值在反向传播时会交替改变符号（因为是轮流行动）

### 2. PUCT 公式详解

与标准 UCB1 不同，PUCT 加入了**先验概率 P(s,a)**：

```
利用项：当前胜率
├─ 高胜率的动作会被更多地选择

探索项：未充分探索的动作
├─ 具有高先验概率的动作（启发好）会被早期探索
├─ 访问次数少的动作会被再次尝试
└─ 系数 sqrt(N(s)) 确保随着搜索深入，探索减少
```

### 3. 先验概率计算

在没有神经网络的情况下，我们使用启发式评分作为先验：

```python
先验概率 = sigmoid(加权评分 / 10)

其中加权评分 = 
  10 * 对手融合4 +        # 最危险，必须防守
   8 * 对手活4 +          # 防守优先级高
   4 * 对手活3 +
   1 * 对手活2 +
   9 * 自己融合4 +        # 进攻优先级高
   7 * 自己活4 +
   3 * 自己活3 +
  0.5 * 自己活2
```

## 类设计

### MCTSNode 类

```python
class MCTSNode:
    game_state        # 该节点对应的游戏状态
    parent            # 父节点
    children          # 子节点字典
    visit_count       # 访问次数
    value_sum         # 累计价值（用于计算胜率）
    children_expanded # 是否已展开所有可能的子节点
```

**关键方法：**
- `ucb1_score()`：计算 PUCT 分数
- `select_child()`：根据 PUCT 选择最优子节点
- `expand()`：展开所有子节点
- `backup()`：反向传播价值

### MCTSAgent 类

```python
class MCTSAgent:
    game_core         # GomokuCore 实例
    num_simulations   # 每次搜索的模拟次数
    c_puct            # PUCT 探索系数
    root              # 搜索树的根节点
```

**关键方法：**
- `get_best_move()`：找到最优落子点
- `get_move_probabilities()`：获取落子点的概率分布
- `_simulate_once()`：执行一次 MCTS 迭代
- `_simulate_playout()`：启发式快速模拟游戏

## 使用示例

### 基础使用

```python
from game_for_rl import GomokuCore
from mcts import MCTSAgent

# 初始化游戏和 Agent
game = GomokuCore(board_size=15)
agent = MCTSAgent(game, num_simulations=1000, c_puct=1.0)

# 获取最优落子
best_move = agent.get_best_move(game)
print(f"Best move: {best_move}")

# 执行落子
game.place_stone(best_move[0], best_move[1])
```

### 获取落子概率分布

```python
# 用于强化学习训练
probabilities = agent.get_move_probabilities()
# 返回 {(row, col): probability, ...}
```

### 获取落子价值评估

```python
move_values = agent.get_move_values()
# 返回 {(row, col): win_rate, ...}
# win_rate 范围: [-1, 1]
#   > 0: 该落子对当前玩家有利
#   < 0: 该落子对当前玩家不利
```

### 打印搜索树信息

```python
agent.print_tree_info()
# 显示访问次数最多的前10个落子点及其统计数据
```

## 性能优化

### 1. 候选点限制
- 使用 `recommand_positions()` 返回的候选点排序，优先考虑棋局重心附近的点
- 展开时最多保留 10 个候选
- 模拟时最多考虑 5 个候选

### 2. 启发式模拟
- 不使用完全随机走棋（慢且质量差）
- 使用快速评分函数选择"看起来更好"的落子
- 限制最大步数（100步）

### 3. 深拷贝优化
- 使用 `copy.deepcopy()` 创建游戏状态副本
- 虽然不如传统 MCTS 中的"快速翻转"那样高效，但便于与现有的 GomokuCore 集成

## 调整参数

### num_simulations（模拟次数）
- **更多**：更强的棋力，更长的思考时间
- **200-500**：快速演示
- **1000+**：较强棋力
- **5000+**：很强棋力（耗时）

### c_puct（探索系数）
- **0.5-1.0**：更倾向于利用已知的好手
- **1.0-2.0**：平衡利用和探索
- **>2.0**：更多探索新手段

### 候选点限制
在 `expand()` 和 `_simulate_playout()` 中修改 `max_candidates`：
- **更多**：考虑更广泛的可能性，但速度慢
- **更少**：速度快，但可能错过好手

## 与神经网络集成

当集成预训练的神经网络时，修改：

```python
def _get_prior_probability(self):
    # 替换启发式评分，使用神经网络输出
    move = self.move
    network_prior = neural_network.predict(game_state, move)
    return network_prior
```

和模拟阶段：

```python
def _simulate_playout(self, game_state):
    # 使用神经网络评估而不是随机模拟
    while not game_state.game_over:
        candidates = game_state.recommand_positions()
        values = neural_network.evaluate_batch(game_state, candidates)
        best_move = candidates[np.argmax(values)]
        game_state.place_stone(best_move[0], best_move[1])
    
    # 使用神经网络的最终评估代替胜负判定
    value = neural_network.evaluate_final(game_state)
    return value
```

## 常见问题

### Q: 为什么价值在 backup 时要翻转符号？
**A**: 因为五子棋是零和博弈。如果节点 A 对黑棋是胜局，那么对白棋就是败局。父节点的价值应该是对自己有利的概率。

### Q: 为什么使用 visit_count 作为最终选择标准而不是 PUCT？
**A**: 搜索树完成后，PUCT 已经完成了其探索-利用的任务。此时，**访问次数最多的动作是蒙特卡洛采样最确定的**（方差最小）。

### Q: 可以调整模拟策略吗？
**A**: 可以。修改 `_simulate_playout()` 方法：
- 实现更复杂的启发式评分
- 集成神经网络
- 使用更快的评估函数
- 限制搜索深度

## 扩展建议

1. **Alpha-Zero 风格集成**：加入神经网络先验和价值估计
2. **并行化**：实现虚拟损失或批处理以支持多线程搜索
3. **对称性**：利用棋盘对称性减少搜索空间
4. **开局库**：对开局位置使用预计算的评估
5. **残局求解**：对简单局面进行精确求解
