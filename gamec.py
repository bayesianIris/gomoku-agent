import numpy as np
from numba import njit, int32, float64

# ==========================================
#  Numba JIT 加速的核心计算函数 (独立于类)
# ==========================================

@njit(cache=True)
def jit_search_side(board, target, row, col, dr, dc, board_size):
    """
    原 _my_search_side 的纯逻辑版
    返回: (bool 是否连五, int 活3增量, int 活4增量)
    """
    count = 1
    count_middle_blank = 0
    count_edge_blocked_blank = 0
    side = 0
    zero_flag = 0
    count_hard = 1
    count_this_dir_middle_blank = 0
    
    # 辅助：获取对手棋子颜色
    opponent = 3 - target

    # 正向检查
    for i in range(1, 7):
        r, c = row + dr * i, col + dc * i
        if not (0 <= r < board_size and 0 <= c < board_size):
            if zero_flag == 1:
                count_edge_blocked_blank += 1
                zero_flag = 0
            break
        
        if zero_flag == 1:
            zero_flag = 0
            if board[r, c] == target:
                count_middle_blank += 1
                count_this_dir_middle_blank += 1
            elif board[r, c] == opponent:
                count_edge_blocked_blank += 1
                break
            else: # 空
                side += 1
                break
        
        if board[r, c] == target:
            count += 1
            if count_this_dir_middle_blank == 0:
                count_hard += 1
        elif board[r, c] == 0:
            zero_flag = 1
        else:
            break

    # 反向检查
    count_this_dir_middle_blank = 0
    for i in range(1, 7):
        r, c = row - dr * i, col - dc * i
        if not (0 <= r < board_size and 0 <= c < board_size):
            if zero_flag == 1:
                count_edge_blocked_blank += 1
                zero_flag = 0
            break
        
        if zero_flag == 1:
            zero_flag = 0
            if board[r, c] == target:
                count_middle_blank += 1
                count_this_dir_middle_blank += 1
            elif board[r, c] == opponent:
                count_edge_blocked_blank += 1
                break
            else:
                side += 1
                break
        
        if board[r, c] == target:
            count += 1
            if count_this_dir_middle_blank == 0:
                count_hard += 1
        elif board[r, c] == 0:
            zero_flag = 1
        else:
            break

    l3_inc = 0
    l4_inc = 0
    
    if count_hard >= 5:
        return True, 0, 0
    
    if count == 3 and count_middle_blank <= 1:
        if side == 2:
            l3_inc += 1
        elif side == 1 and count_edge_blocked_blank == 1:
            l3_inc += 1
        elif side == 0 and count_edge_blocked_blank == 2 and count_middle_blank == 1:
            l3_inc += 1
    elif count == 4 and count_middle_blank <= 1:
        if side + count_edge_blocked_blank >= 1:
            l4_inc += 1
            
    return False, l3_inc, l4_inc

@njit(cache=True)
def jit_check_logic(board, current_player, row, col, board_size):
    """
    原 _check_win 的完整逻辑封装。
    包含两部分：
    1. 检查当前玩家是否胜利，并计算当前玩家的活三/活四增量
    2. 模拟检查对手棋型的变化（用于更新对手的活三/活四计数）
    
    返回: 
    (bool is_win, 
     int current_l3_delta, int current_l4_delta, 
     int opponent_l3_delta, int opponent_l4_delta)
    """
    
    # --- Part 1: 检查自身 (Win check & L3/L4 update) ---
    target = current_player
    is_win = False
    
    cur_l3_delta = 0
    cur_l4_delta = 0
    
    # 四个方向
    # directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    # 为了Numba性能，手动展开或者用数组，这里用简单的循环
    dr_list = np.array([0, 1, 1, 1])
    dc_list = np.array([1, 0, 1, -1])
    
    for i in range(4):
        w, l3, l4 = jit_search_side(board, target, row, col, dr_list[i], dc_list[i], board_size)
        if w:
            is_win = True
            # 赢了就不必继续算heuristics了，不过原逻辑似乎会继续，这里如果赢了直接返回
            return True, 0, 0, 0, 0 
        cur_l3_delta += l3
        cur_l4_delta += l4
        
    # --- Part 2: 计算对手变化 (Opponent simulation) ---
    # 原逻辑：活三： 2011100 ... 再算对手的
    
    opp_target = 3 - current_player
    opp_l3_delta = 0
    opp_l4_delta = 0
    
    # 8个方向的初始化检查点
    # inits = [(0,1),(1,0),(1,1),(1,-1),(0,-1),(-1,0),(-1,-1),(-1,1)]
    # 拆分成 dr_arr, dc_arr
    opp_dr_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    opp_dc_arr = np.array([1, 0, 1, -1, -1, 0, -1, 1])
    
    for i in range(8):
        dr, dc = opp_dr_arr[i], opp_dc_arr[i]
        
        # 对手棋起点
        _r, _c = row + dr, col + dc
        
        if not (0 <= _r < board_size and 0 <= _c < board_size):
            continue
            
        cell_val = board[_r, _c]
        
        # 逻辑分支1: 旁边是空位，但再往后是对手棋子 (对应原代码 if self.board[_r, _c] != 0 check)
        # 原代码逻辑：if self.board[_r, _c] != 0 ... if board[next] != target
        # 其实原代码意思大约是：如果紧邻不是0(非空)，检查它；如果是0，不检查。
        # 但原代码紧接着写了 if != 0 ... inside: if board[next] != target continue
        # 这里为了保持逻辑一致性，仔细还原：
        
        if cell_val != 0:
            # 只有当它是对手棋子时才可能有影响，或者它阻挡了对手？
            # 原代码逻辑比较复杂，这里严格还原原代码的行为
            
            # 原代码 case 1: 紧邻有子
            next_r, next_c = _r + dr, _c + dc
            if not (0 <= next_r < board_size and 0 <= next_c < board_size):
                continue
            
            if board[next_r, next_c] != opp_target:
                continue
            
            # 模拟：如果没有当前这个子 (board[row,col]=0)，对手的情况 vs 有这个子的情况
            # Step 1: 假设这里为空 (Simulate Remove)
            board[row, col] = 0 
            _, l3_before, l4_before = jit_search_side(board, opp_target, next_r, next_c, dr, dc, board_size)
            
            # 优化：如果拿掉也没棋，就不必放回去再算了 (原代码: if rec2==(0,0): continue)
            if l3_before == 0 and l4_before == 0:
                board[row, col] = current_player # Restore
                continue
                
            # Step 2: 放回当前子 (Simulate Place/Restore)
            board[row, col] = current_player
            _, l3_after, l4_after = jit_search_side(board, opp_target, next_r, next_c, dr, dc, board_size)
            
            opp_l3_delta += (l3_after - l3_before)
            opp_l4_delta += (l4_after - l4_before)
            
        elif cell_val == opp_target:
            # 原代码 case 2: 紧邻就是对手棋子
            # Step 1: Remove
            board[row, col] = 0
            _, l3_before, l4_before = jit_search_side(board, opp_target, _r, _c, dr, dc, board_size)
            
            if l3_before == 0 and l4_before == 0:
                board[row, col] = current_player # Restore
                continue
                
            # Step 2: Restore
            board[row, col] = current_player
            _, l3_after, l4_after = jit_search_side(board, opp_target, _r, _c, dr, dc, board_size)
            
            opp_l3_delta += (l3_after - l3_before)
            opp_l4_delta += (l4_after - l4_before)

    return False, cur_l3_delta, cur_l4_delta, opp_l3_delta, opp_l4_delta

@njit(cache=True)
def jit_get_candidates_sort(board, row_min, row_max, col_min, col_max, center_r, center_c):
    """
    加速 recommand_positions 中的距离排序计算
    """
    # 获取子区域
    sub_board = board[row_min:row_max, col_min:col_max]
    
    # 找到空位 (局部坐标)
    # np.argwhere 在 Numba 中支持，但返回形式需要注意
    # 这里手动遍历可能更快，或者直接用 numpy 语法（Numba支持部分）
    rows, cols = np.where(sub_board == 0)
    
    n = len(rows)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    
    # 转换为全局坐标并计算距离
    candidates = np.empty((n, 2), dtype=np.int64)
    dists = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        g_r = rows[i] + row_min
        g_c = cols[i] + col_min
        candidates[i, 0] = g_r
        candidates[i, 1] = g_c
        # 计算距离平方
        dists[i] = (g_r - center_r)**2 + (g_c - center_c)**2
        
    # 排序
    sorted_indices = np.argsort(dists)
    
    # 重排
    sorted_candidates = candidates[sorted_indices]
    return sorted_candidates

# ==========================================
#  修改后的类
# ==========================================

class GomokuCore:
    def __init__(self, board_size=15):
        self.board_size = board_size
        # 0: 空, 1: 黑棋, 2: 白棋
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 黑棋先行
        self.winner = None
        self.game_over = False
        self.last_move = None
        # 活3活4解算存
        self.l3_count = {1: 0, 2: 0}
        self.l4_count = {1: 0, 2: 0}

    def reset(self):
        """重置游戏"""
        self.board.fill(0)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.l3_count = {1: 0, 2: 0}
        self.l4_count = {1: 0, 2: 0}

    def place_stone(self, row, col):
        """
        尝试落子
        :return: (bool) 是否落子成功
        """
        if self.game_over:
            return False
        
        # 检查边界和是否已有子
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        if self.board[row, col] != 0:
            return False

        # 执行落子
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # 检查胜负并更新计数 (原有逻辑都在 _check_win 中)
        # 注意：这里 _check_win 现在是一个 wrappers，它会调用 JIT 函数并更新 self.l3_count
        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        else:
            # 切换棋手 (1 -> 2, 2 -> 1)
            self.current_player = 3 - self.current_player
        return True

    def _check_win(self, row, col):
        """
        基于最后落子的位置，判断是否获胜，并更新活三活四计数
        """
        # 调用 JIT 编译的核心逻辑
        # jit_check_logic 会做两件事：
        # 1. 检查 current_player 是否赢了
        # 2. 计算 current_player 增加的活3/4
        # 3. 模拟计算 opponent 改变的活3/4
        
        is_win, cur_l3, cur_l4, opp_l3, opp_l4 = jit_check_logic(
            self.board, self.current_player, row, col, self.board_size
        )

        if is_win:
            return True

        # 更新计数 (将 JIT 返回的 delta 应用到 python 字典)
        opponent = 3 - self.current_player
        
        self.l3_count[self.current_player] += cur_l3
        self.l4_count[self.current_player] += cur_l4
        
        self.l3_count[opponent] += opp_l3
        self.l4_count[opponent] += opp_l4
        
        return False

    def simu_check(self, row, col):
        """
        模拟检查：不改变当前棋盘状态，计算落子后的结果
        """
        # 这一步在原代码中非常昂贵，因为它涉及深拷贝字典和回滚
        # 使用 JIT 函数后，我们可以直接计算出 delta，而不必真正修改 dict
        
        # 1. 临时落子
        if self.board[row, col] != 0:
            # 理论上不应该发生，但防御性编程
            return False, self.l3_count.copy(), self.l4_count.copy()
            
        self.board[row, col] = self.current_player
        
        # 2. 计算 JIT 结果
        is_win, cur_l3, cur_l4, opp_l3, opp_l4 = jit_check_logic(
            self.board, self.current_player, row, col, self.board_size
        )
        
        # 3. 回滚落子
        self.board[row, col] = 0
        
        # 4. 构造返回结果 (不修改 self 的属性，只返回计算出的新状态)
        res_l3 = self.l3_count.copy()
        res_l4 = self.l4_count.copy()
        
        opponent = 3 - self.current_player
        
        if not is_win:
            res_l3[self.current_player] += cur_l3
            res_l4[self.current_player] += cur_l4
            res_l3[opponent] += opp_l3
            res_l4[opponent] += opp_l4
            
        return is_win, res_l3, res_l4

    def get_search_range(self, padding=2):
        """
        计算有子区域的 Bounding Box (边界框)，用于减少搜索范围
        :return: (row_min, row_max, col_min, col_max)
        """
        row_has_stone = self.board.any(axis=1)
        col_has_stone = self.board.any(axis=0)
        
        if not np.any(row_has_stone):
            center = self.board_size // 2
            return center-1, center+2, center-1, center+2
        
        row_indices = np.where(row_has_stone)[0]
        col_indices = np.where(col_has_stone)[0]

        row_min = np.clip(row_indices[0] - padding, 0, self.board_size)
        row_max = np.clip(row_indices[-1] + padding + 1, 0, self.board_size)
        
        col_min = np.clip(col_indices[0] - padding, 0, self.board_size)
        col_max = np.clip(col_indices[-1] + padding + 1, 0, self.board_size)

        return row_min, row_max, col_min, col_max

    def recommand_positions(self, padding=2):
        """
        返回候选落子点，按照离当前棋局重心距离从近到远排序
        """
        row_min, row_max, col_min, col_max = self.get_search_range(padding)
        
        # 1. 计算重心
        stone_indices = np.argwhere(self.board != 0)
        if len(stone_indices) > 0:
            center_r, center_c = np.mean(stone_indices, axis=0)
        else:
            center_r, center_c = float(self.board_size // 2), float(self.board_size // 2)

        # 2. 使用 JIT 加速的排序函数
        # 注意：这里需要传入 float 类型的 center，并在 jit 内部处理
        sorted_candidates = jit_get_candidates_sort(
            self.board, row_min, row_max, col_min, col_max, center_r, center_c
        )

        if len(sorted_candidates) == 0:
            return []

        # 转换为 list of tuples
        return [tuple(pos) for pos in sorted_candidates]

    def get_board(self):
        """返回只读的numpy数组供显示层使用"""
        return self.board