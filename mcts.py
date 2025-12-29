import numpy as np
import copy
from math import sqrt, log
from typing import List, Tuple, Optional


class MCTSNode:
    """MCTS 树的节点"""
    
    def __init__(self, game_state, parent=None, move=None, is_root=False):
        """
        初始化 MCTS 节点
        :param game_state: GomokuCore 游戏状态的深拷贝
        :param parent: 父节点
        :param move: 导致该节点的移动 (row, col)
        :param is_root: 是否为根节点
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move  # 该节点对应的落子点
        self.children = {}  # 子节点字典 {move: MCTSNode}
        self.is_root = is_root
        
        # 访问统计
        self.visit_count = 0
        self.value_sum = 0.0  # 累计价值
        self.children_expanded = False  # 是否已展开所有子节点
        
    def ucb1_score(self, c_puct=1.0):
        """
        计算 PUCT 值（上置信界）
        PUCT = Q(s,a)/N(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        由于我们没有神经网络提供先验概率 P(s,a)，我们使用：
        - 棋局评分启发值作为先验
        - Q 值为 win_rate
        
        :param c_puct: 探索系数（通常 1.0-2.0）
        :return: PUCT 分数
        """
        if self.visit_count == 0:
            return float('inf')
        
        # 胜率
        win_rate = self.value_sum / self.visit_count
        
        # 先验概率（使用启发式评分）
        prior = self._get_prior_probability()
        
        # PUCT 公式
        exploitation = win_rate
        exploration = c_puct * prior * sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def _get_prior_probability(self):
        """
        获取节点的先验概率（启发式）
        基于该落子点的棋局评分
        :return: [0, 1] 范围内的概率值
        """
        if self.move is None:
            return 0.5
        
        target_player = self.game_state.current_player
        
        # 使用详细的局势评估函数
        score = self._evaluate_position(target_player)
        
        # 归一化到 [0.01, 1.0]（避免为0）
        # 使用 sigmoid 函数确保在合理范围
        # 除以 100 是为了让评分在合理的 sigmoid 输入范围内
        prior = 1.0 / (1.0 + np.exp(-score / 100))
        
        # 确保返回有效的概率值
        return float(np.clip(prior, 0.01, 1.0))
    
    def _evaluate_position(self, my_player):
        """
        局势评估函数（细化版）
        基于活二、活三、冲四、必杀四的数量计算评分
        :param my_player: 当前玩家 (1 或 2)
        :return: 评分（正数表示对 my_player 有利）
        """
        opponent = 3 - my_player
        score = 0.0
        
        # 获取各类型的计数
        my_l2 = self.game_state.l2_count.get(my_player, 0)
        opp_l2 = self.game_state.l2_count.get(opponent, 0)
        
        my_l3 = self.game_state.l3_count.get(my_player, 0)
        opp_l3 = self.game_state.l3_count.get(opponent, 0)
        
        my_l4 = self.game_state.l4_count.get(my_player, 0)
        opp_l4 = self.game_state.l4_count.get(opponent, 0)
        
        my_rl4 = self.game_state.rl4_count.get(my_player, 0)
        opp_rl4 = self.game_state.rl4_count.get(opponent, 0)
        
        # =================== 基础评分 ===================
        # l2 得1分，双l2得额外3分
        score += my_l2 * 1
        if my_l2 >= 2:
            score += 3
        score -= opp_l2 * 1
        if opp_l2 >= 2:
            score -= 3
        
        # l3 得5分
        score += my_l3 * 5
        score -= opp_l3 * 5
        
        # l4（冲四）得20分
        score += my_l4 * 20
        score -= opp_l4 * 20
        
        # rl4（必杀四/连五）得1000分
        score += my_rl4 * 1000
        score -= opp_rl4 * 1000
        
        # =================== 杀棋判定 ===================
        # 我方杀棋判定
        # 条件1：对方无l4和rl4，我方(l3+l4)>1且l4>1 -> +500分
        if opp_l4 == 0 and opp_rl4 == 0:
            if (my_l3 + my_l4) > 1 and my_l4 > 1:
                score += 500
        
        # 条件2：对方无l4、l3和rl4，我方有两个l3 -> +100分
        if opp_l4 == 0 and opp_l3 == 0 and opp_rl4 == 0:
            if my_l3 >= 2:
                score += 100
        
        # 对方杀棋判定（镜像）
        # 条件1：我方无l4和rl4，对手(l3+l4)>1且l4>1 -> -500分
        if my_l4 == 0 and my_rl4 == 0:
            if (opp_l3 + opp_l4) > 1 and opp_l4 > 1:
                score -= 500
        
        # 条件2：我方无l4、l3和rl4，对手有两个l3 -> -100分
        if my_l4 == 0 and my_l3 == 0 and my_rl4 == 0:
            if opp_l3 >= 2:
                score -= 100
        
        return score
    
    def select_child(self, c_puct=1.0):
        """
        根据 PUCT 公式选择最优的子节点
        :param c_puct: 探索系数
        :return: 最优子节点
        """
        if not self.children:
            return None
        
        best_child = None
        best_score = -float('inf')
        
        for child in self.children.values():
            score = child.ucb1_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self):
        """
        扩展该节点的所有可能的子节点
        使用推荐位置快速生成候选落子点
        """
        if self.children_expanded:
            return
        
        if self.game_state.game_over:
            self.children_expanded = True
            return
        
        candidates = self.game_state.recommand_positions(padding=1)
        
        # 限制候选数量以加快搜索
        max_candidates = 10
        if len(candidates) > max_candidates:
            # 优先考虑最近的候选
            candidates = candidates[:max_candidates]
        
        for move in candidates:
            # 为每个候选落子点创建子节点
            child_game_state = copy.deepcopy(self.game_state)
            child_game_state.place_stone(move[0], move[1])
            child_node = MCTSNode(child_game_state, parent=self, move=move)
            self.children[move] = child_node
        
        self.children_expanded = True
    
    def backup(self, value):
        """
        反向传播价值
        :param value: 该节点的游戏价值 (1: 当前玩家赢, 0: 平手, -1: 当前玩家输)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            # 反转价值视角：父节点看该节点的价值是相反的
            node.value_sum += value
            value = -value  # 切换视角
            node = node.parent


class MCTSAgent:
    """蒙特卡洛树搜索 Agent"""
    
    def __init__(self, game_core, num_simulations=1000, c_puct=1.0):
        """
        初始化 MCTS Agent
        :param game_core: GomokuCore 实例
        :param num_simulations: 每次搜索的模拟次数
        :param c_puct: PUCT 的探索系数
        """
        self.game_core = game_core
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
    
    def reset_tree(self):
        """重置搜索树"""
        self.root = None
    
    def get_best_move(self, game_state, return_value=False):
        """
        使用 MCTS 找到最优落子点
        :param game_state: GomokuCore 当前游戏状态
        :param return_value: 是否返回该步的评估值
        :return: (row, col) 或 ((row, col), value)
        """
        # 创建根节点
        root_state = copy.deepcopy(game_state)
        self.root = MCTSNode(root_state, is_root=True)
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            self._simulate_once()
        
        # 选择最优子节点（访问次数最多）
        best_child = max(self.root.children.values(), 
                        key=lambda child: child.visit_count)
        best_move = best_child.move
        best_value = best_child.value_sum / best_child.visit_count if best_child.visit_count > 0 else 0
        
        if return_value:
            return best_move, best_value
        return best_move
    
    def get_move_probabilities(self):
        """
        获取所有落子点的概率分布（基于访问次数）
        用于强化学习训练
        :return: {move: probability}
        """
        if self.root is None:
            return {}
        
        total_visits = sum(child.visit_count for child in self.root.children.values())
        if total_visits == 0:
            return {}
        
        probabilities = {}
        for move, child in self.root.children.items():
            probabilities[move] = child.visit_count / total_visits
        
        return probabilities
    
    def _simulate_once(self):
        """
        执行一次 MCTS 模拟：Selection -> Expansion -> Simulation -> Backup
        """
        node = self.root
        
        # Selection & Expansion
        while not node.game_state.game_over and node.children_expanded:
            # 从现有子节点中选择最优的
            child = node.select_child(self.c_puct)
            if child is None:
                break
            node = child
        
        # 如果游戏未结束且节点未展开，则展开
        if not node.game_state.game_over and not node.children_expanded:
            node.expand()
            
            # 如果有子节点，随机选择一个进行模拟
            if node.children:
                node = list(node.children.values())[
                    np.random.randint(0, len(node.children))
                ]
        
        # Simulation：使用快速启发式模拟游戏
        value = self._simulate_playout(node.game_state)
        
        # Backup
        node.backup(value)
    
    def _simulate_playout(self, game_state):
        """
        快速启发式模拟：使用游戏内已有的评分系统
        :param game_state: 当前游戏状态
        :return: 游戏结果 (1: 当前玩家赢, 0: 平手, -1: 当前玩家输)
        """
        sim_game = copy.deepcopy(game_state)
        original_player = game_state.current_player
        max_moves = 100  # 防止无限循环
        
        for _ in range(max_moves):
            if sim_game.game_over:
                if sim_game.winner == original_player:
                    return 1  # 原始玩家赢
                else:
                    return -1  # 原始玩家输
            
            # 使用评分启发式选择落子点
            candidates = sim_game.recommand_positions(padding=1)
            
            if not candidates:
                return 0  # 棋盘满了，平手
            
            # 限制候选数量
            candidates = candidates[:5]
            
            # 使用 simu_check 评估落子后的局面
            best_move = None
            best_score = -float('inf')
            
            for move in candidates:
                # 使用 simu_check 获取该落子后的评估
                # 这会利用游戏内已有的 _check_win 和 _my_search_side 计算
                win, res_l2, res_l3, res_l4, res_rl4 = sim_game.simu_check(move[0], move[1])
                
                if win:
                    # 能直接赢，选择这个落子
                    best_move = move
                    break
                
                # 基于落子后的棋局评估评分
                score = self._evaluate_move(res_l2, res_l3, res_l4, res_rl4, sim_game.current_player)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move is None:
                best_move = candidates[0]
            
            sim_game.place_stone(best_move[0], best_move[1])
        
        # 如果超过最大步数，返回平手
        return 0
    
    def _evaluate_move(self, l2, l3, l4, rl4, my_player):
        """
        基于棋局计数评估一个落子（用于 simu_check 结果）
        :param l2, l3, l4, rl4: 落子后的各类型计数字典
        :param my_player: 当前玩家
        :return: 评分（正数表示对 my_player 有利）
        """
        opponent = 3 - my_player
        score = 0.0
        
        # 获取各类型的计数
        my_l2 = l2.get(my_player, 0)
        opp_l2 = l2.get(opponent, 0)
        
        my_l3 = l3.get(my_player, 0)
        opp_l3 = l3.get(opponent, 0)
        
        my_l4 = l4.get(my_player, 0)
        opp_l4 = l4.get(opponent, 0)
        
        my_rl4 = rl4.get(my_player, 0)
        opp_rl4 = rl4.get(opponent, 0)
        
        # =================== 基础评分 ===================
        # l2 得1分，双l2得额外3分
        score += my_l2 * 1
        if my_l2 >= 2:
            score += 3
        score -= opp_l2 * 1
        if opp_l2 >= 2:
            score -= 3
        
        # l3 得5分
        score += my_l3 * 5
        score -= opp_l3 * 5
        
        # l4（冲四）得20分
        score += my_l4 * 20
        score -= opp_l4 * 20
        
        # rl4（必杀四/连五）得1000分
        score += my_rl4 * 1000
        score -= opp_rl4 * 1000
        
        # =================== 杀棋判定 ===================
        # 我方杀棋判定
        # 条件1：对方无l4和rl4，我方(l3+l4)>1且l4>1 -> +500分
        if opp_l4 == 0 and opp_rl4 == 0:
            if (my_l3 + my_l4) > 1 and my_l4 > 1:
                score += 500
        
        # 条件2：对方无l4、l3和rl4，我方有两个l3 -> +100分
        if opp_l4 == 0 and opp_l3 == 0 and opp_rl4 == 0:
            if my_l3 >= 2:
                score += 100
        
        # 对方杀棋判定（镜像）
        # 条件1：我方无l4和rl4，对手(l3+l4)>1且l4>1 -> -500分
        if my_l4 == 0 and my_rl4 == 0:
            if (opp_l3 + opp_l4) > 1 and opp_l4 > 1:
                score -= 500
        
        # 条件2：我方无l4、l3和rl4，对手有两个l3 -> -100分
        if my_l4 == 0 and my_l3 == 0 and my_rl4 == 0:
            if opp_l3 >= 2:
                score -= 100
        
        return score
    
    
    def get_move_values(self):
        """
        获取所有候选落子点的价值
        :return: {move: win_rate}
        """
        if self.root is None:
            return {}
        
        move_values = {}
        for move, child in self.root.children.items():
            if child.visit_count > 0:
                win_rate = child.value_sum / child.visit_count
                move_values[move] = win_rate
            else:
                move_values[move] = 0
        
        return move_values
    
    def print_tree_info(self):
        """打印树的统计信息"""
        if self.root is None:
            print("No tree available")
            return
        
        print(f"\n=== MCTS Tree Info ===")
        print(f"Root visit count: {self.root.visit_count}")
        print(f"Number of children: {len(self.root.children)}")
        print(f"\nTop 10 moves by visit count:")
        print(f"{'Move':<10} {'Visits':<10} {'Value':<10} {'Win Rate':<10}")
        print("-" * 40)
        
        sorted_children = sorted(
            self.root.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True
        )
        
        for i, (move, child) in enumerate(sorted_children[:10]):
            win_rate = child.value_sum / child.visit_count if child.visit_count > 0 else 0
            print(f"{str(move):<10} {child.visit_count:<10} {child.value_sum:<10.2f} {win_rate:<10.2f}")
