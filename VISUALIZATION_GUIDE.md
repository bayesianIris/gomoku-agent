# MCTS 可视化示例使用指南

## 概述

已更新的 `mcts_example.py` 现在支持完整的可视化展示，能够实时显示 MCTS Agent 的决策过程和棋盘状态。

## 核心特性

### 1. **完整游戏可视化**
```python
play_game_with_mcts(
    board_size=15,           # 棋盘大小
    num_simulations=500,     # 每步搜索模拟次数
    verbose=True,            # 打印详细信息
    visualize=True           # 启用 pygame 可视化
)
```

**显示内容：**
- ✅ 实时棋盘状态（黑白棋子位置）
- ✅ 上一步落子位置（红色圆圈标记）
- ✅ 实时统计信息（L2、L3、L4、RL4 数量）
- ✅ 游戏进度和搜索时间
- ✅ 游戏结果和最终消息

### 2. **局面分析可视化**
```python
demonstrate_mcts_analysis(
    board_size=15,      # 棋盘大小
    visualize=True      # 启用可视化
)
```

**显示流程：**
1. 显示初始局面（3 个落子的起始状态）
2. 执行 1000 次 MCTS 模拟
3. 显示分析结果的统计数据
4. 显示预测的最优落子点（临时预览）
5. 关闭窗口

### 3. **候选点评估可视化**
```python
compare_positions(
    game_state=game,         # 当前游戏状态
    num_simulations=500,     # 搜索次数
    visualize=True           # 启用可视化
)
```

**功能：**
- 分析所有候选落子点
- 按价值排序显示前 15 个落子点
- 可视化显示评分最高的落子

## 使用示例

### 快速开始（运行所有示例）

```bash
python mcts_example.py
```

这会依次执行：
1. **示例 1**：两个 MCTS Agent 对弈（每步 200 次模拟）
   - 预期时长：3-5 分钟（取决于棋盘和硬件）
   - 自动显示每一步的棋盘和统计信息

2. **示例 2**：分析特定局面（1000 次模拟）
   - 显示初始的 3 步局面
   - 分析该局面的最优应对
   - 显示候选点的价值排序

### 自定义使用

```python
from game_for_rl import GomokuCore
from mcts import MCTSAgent
from mcts_example import play_game_with_mcts, demonstrate_mcts_analysis

# 示例 1：更强的 Agent（更多模拟）
winner = play_game_with_mcts(
    board_size=15,
    num_simulations=1000,  # 每步更多模拟 = 更强的棋力
    verbose=True,
    visualize=True
)
print(f"Winner: {winner}")

# 示例 2：关闭可视化（仅输出日志）
winner = play_game_with_mcts(
    board_size=15,
    num_simulations=200,
    verbose=True,
    visualize=False  # 关闭 UI，仅保留控制台输出
)

# 示例 3：更小的棋盘（快速演示）
winner = play_game_with_mcts(
    board_size=9,  # 9x9 棋盘更快
    num_simulations=100,
    verbose=True,
    visualize=True
)
```

## 界面说明

### 棋盘显示

```
┌─────────────────────────────────────┐
│  Black: L2=0 | L3=0 | L4=0 | RL4=0  │  ← 黑棋统计信息
│                                     │
│    ○  ○  ○  ○  ○  ○  ○  ○  ○      │  ← 棋盘网格线
│    ○  ○  ●  ○  ●  ○  ○  ○  ○      │  ← ● 黑棋  ○ 空位
│    ○  ○  ○  ●  ○  ○  ○  ○  ○      │  ← ◎ 最后一步
│    ○  ○  ◎  ○  ○  ○  ○  ○  ○      │
│    ○  ○  ○  ○  ○  ○  ○  ○  ○      │
│    ○  ○  ○  ○  ○  ○  ○  ○  ○      │
│    ○  ○  ○  ○  ○  ○  ○  ○  ○      │
│    ○  ○  ○  ○  ○  ○  ○  ○  ○      │
│    ○  ○  ○  ○  ○  ○  ○  ○  ○      │
│                                     │
│ White: L2=0 | L3=0 | L4=0 | RL4=0  │  ← 白棋统计信息
└─────────────────────────────────────┘
```

### 控制台输出

```
======================================================================
Game Start: 15x15 board
MCTS Simulations per move: 200
Visualization: ON
======================================================================

Move 1: Black (1)
  Move: (7, 7)
  Search time: 2.45s
  L2 count - Black: 0, White: 0
  L3 count - Black: 0, White: 0
  L4 count - Black: 0, White: 0
  RL4 count - Black: 0, White: 0

Move 2: White (2)
  Move: (7, 8)
  Search time: 2.31s
  L2 count - Black: 0, White: 0
  L3 count - Black: 0, White: 0
  L4 count - Black: 0, White: 0
  RL4 count - Black: 0, White: 0
  ...
```

## 性能参数调整

### 加快游戏速度

```python
# 减少模拟次数
winner = play_game_with_mcts(num_simulations=100, visualize=True)

# 减小棋盘
winner = play_game_with_mcts(board_size=9, num_simulations=200, visualize=True)

# 关闭可视化（明显加快，但看不到棋盘）
winner = play_game_with_mcts(num_simulations=200, visualize=False)
```

### 提升 AI 棋力

```python
# 增加模拟次数（最有效）
winner = play_game_with_mcts(num_simulations=2000, visualize=True)

# 提高 c_puct（探索系数）
agent = MCTSAgent(game, num_simulations=1000, c_puct=2.0)

# 两者结合
winner = play_game_with_mcts(num_simulations=5000, visualize=True)
```

## 常见问题

### Q: 游戏太慢了怎么办？
**A:** 
1. 减少 `num_simulations`（100-300）
2. 关闭 `visualize=False`（纯控制台运行）
3. 使用更小的棋盘 `board_size=9`

### Q: 窗口卡住了怎么办？
**A:** 直接关闭窗口，程序会自动清理并退出。

### Q: 可以修改棋盘大小吗？
**A:** 可以，传入 `board_size` 参数（如 9, 11, 15, 19）。注意：较大的棋盘会显著增加搜索时间。

### Q: 支持人机对弈吗？
**A:** 当前版本只支持 AI vs AI。可以通过修改代码添加人机交互。

## 进阶用法

### 自定义初始局面

```python
from game_for_rl import GomokuCore
from mcts_example import compare_positions

# 创建游戏
game = GomokuCore(board_size=15)

# 设置初始落子
game.place_stone(7, 7)   # 黑
game.place_stone(7, 8)   # 白
game.place_stone(8, 7)   # 黑

# 分析当前局面
compare_positions(game, num_simulations=500, visualize=True)
```

### 保存游戏记录

```python
from mcts_example import play_game_with_mcts

moves_history = []

def play_and_record():
    game = GomokuCore(board_size=15)
    agent = MCTSAgent(game, num_simulations=200)
    
    while not game.game_over:
        move = agent.get_best_move(game)
        moves_history.append((game.current_player, move))
        game.place_stone(move[0], move[1])
    
    print("Game history:", moves_history)
```

## 文件依赖

- `game_for_rl.py` - 游戏逻辑核心
- `mcts.py` - MCTS 算法实现
- `display.py` - pygame 可视化界面
- `mcts_example.py` - 示例和演示脚本
- `pygame` - 图形库（需要安装）

```bash
pip install pygame numpy
```

## 总结

通过 `mcts_example.py` 的可视化功能，你可以：
- 👀 **观看** MCTS Agent 的对弈过程
- 📊 **分析** 特定局面的评估结果
- 🔍 **检验** 五子棋的评分函数是否合理
- 🎓 **学习** PUCT 算法的实际应用

享受五子棋 AI 的可视化之旅！
