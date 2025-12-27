import numpy as np
from numba import njit
from numba.typed import List
from numba.types import Tuple, int32

# ==================== JIT 编译的核心计算函数 ====================

@njit(cache=True)
def jit_search_side(board, target, row, col, dr, dc, board_size):
    """
    JIT 优化版本的 _my_search_side。
    返回: (bool_win, record_array)
    record 格式: [l3_count, l4_count, l2_count, rl4_count]
    """
    count = 1
    count_middle_blank = 0
    count_edge_blocked_blank = 0
    side = 0
    zero_flag = 0
    count_hard = 1
    count_this_dir_middle_blank = 0
    
    record = np.zeros(4, dtype=np.int32)

    # 正向检查
    for i in range(1, 7):
        r, c = row + dr * i, col + dc * i
        if not (0 <= r < board_size and 0 <= c < board_size):
            if zero_flag == 1:
                count_edge_blocked_blank += 1
                zero_flag = 0
            break
        
        val = board[r, c]
        if zero_flag == 1:
            zero_flag = 0
            if val == target:
                count_middle_blank += 1
                count_this_dir_middle_blank += 1
            elif val == 3 - target:
                count_edge_blocked_blank += 1
                break
            else:
                side += 1
                break
        
        if val == target:
            count += 1
            if count_this_dir_middle_blank == 0:
                count_hard += 1
        elif val == 0:
            zero_flag = 1
        else:
            break

    # 反向检查
    count_this_dir_middle_blank = 0
    zero_flag = 0
    
    for i in range(1, 7):
        r, c = row - dr * i, col - dc * i
        if not (0 <= r < board_size and 0 <= c < board_size):
            if zero_flag == 1:
                count_edge_blocked_blank += 1
                zero_flag = 0
            break
            
        val = board[r, c]
        if zero_flag == 1:
            zero_flag = 0
            if val == target:
                count_middle_blank += 1
                count_this_dir_middle_blank += 1
            elif val == 3 - target:
                count_edge_blocked_blank += 1
                break
            else:
                side += 1
                break
        
        if val == target:
            count += 1
            if count_this_dir_middle_blank == 0:
                count_hard += 1
        elif val == 0:
            zero_flag = 1
        else:
            break

    # 判定逻辑
    if count_hard >= 5:
        return True, record
    
    if count == 3 and count_middle_blank <= 1:
        if side == 2:
            record[0] += 1  # l3
        elif side == 1 and count_edge_blocked_blank == 1:
            record[0] += 1
        elif side == 0 and count_edge_blocked_blank == 2 and count_middle_blank == 1:
            record[0] += 1
    elif count == 4 and count_middle_blank <= 1:
        if side + count_edge_blocked_blank >= 1:
            record[1] += 1  # l4
    elif count == 4 and count_middle_blank == 0:
        if side + count_edge_blocked_blank >= 1:
            record[3] += 1  # rl4
    elif count == 2 and count_middle_blank <= 1:
        if side + count_edge_blocked_blank == 2:
            record[2] += 1  # l2

    return False, record


@njit(cache=True)
def jit_check_logic(board, row, col, current_player, board_size):
    """
    整合了 _check_win 的完整逻辑。
    返回: (is_win, l3_delta, l4_delta, l2_delta, rl4_delta)
    delta 是针对 opponent (3-current_player) 的增量
    """
    target = current_player
    
    # 检查四个方向是否获胜
    directions = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)
    
    for i in range(4):
        dr, dc = directions[i, 0], directions[i, 1]
        win, _ = jit_search_side(board, target, row, col, dr, dc, board_size)
        if win:
            return True, 0, 0, 0, 0

    # 统计活三/活四/活二/rl4
    opponent = 3 - current_player
    
    total_l3 = 0
    total_l4 = 0
    total_l2 = 0
    total_rl4 = 0
    
    # 8个方向
    check_dirs = np.array([
        [0, 1], [1, 0], [1, 1], [1, -1],
        [0, -1], [-1, 0], [-1, -1], [-1, 1]
    ], dtype=np.int32)

    for i in range(8):
        dr, dc = check_dirs[i, 0], check_dirs[i, 1]
        _r, _c = row + dr, col + dc
        
        if not (0 <= _r < board_size and 0 <= _c < board_size):
            continue
            
        cell_val = board[_r, _c]
        
        # 判断是否需要模拟
        need_sim = False
        sim_r, sim_c = -1, -1
        
        if cell_val != 0:
            # 邻居有棋子
            next_r, next_c = _r + dr, _c + dc
            if (0 <= next_r < board_size and 0 <= next_c < board_size):
                if board[next_r, next_c] == opponent:
                    need_sim = True
                    sim_r, sim_c = next_r, next_c
        elif cell_val == opponent:
            need_sim = True
            sim_r, sim_c = _r, _c
        
        if need_sim:
            # 模拟计算
            board[row, col] = 0
            rec2 = np.zeros(4, dtype=np.int32)
            for k in range(4):
                dr2, dc2 = directions[k, 0], directions[k, 1]
                _, r_tmp = jit_search_side(board, opponent, sim_r, sim_c, dr2, dc2, board_size)
                rec2 += r_tmp
            
            if np.sum(rec2) == 0:
                board[row, col] = current_player
                continue
                
            board[row, col] = current_player
            rec1 = np.zeros(4, dtype=np.int32)
            for k in range(4):
                dr2, dc2 = directions[k, 0], directions[k, 1]
                _, r_tmp = jit_search_side(board, opponent, sim_r, sim_c, dr2, dc2, board_size)
                rec1 += r_tmp
            
            total_l3 += (rec1[0] - rec2[0])
            total_l4 += (rec1[1] - rec2[1])
            total_l2 += (rec1[2] - rec2[2])
            total_rl4 += (rec1[3] - rec2[3])

    return False, total_l3, total_l4, total_l2, total_rl4


@njit(cache=True)
def jit_get_candidates_sort(board, board_size, padding):
    """
    获取候选落子点并按距离重心排序
    返回: 二维数组 shape=(n, 2)，每行是 (row, col)
    """
    # 1. 找边界
    row_min, row_max = board_size, -1
    col_min, col_max = board_size, -1
    
    has_stone = False
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] != 0:
                has_stone = True
                if r < row_min:
                    row_min = r
                if r > row_max:
                    row_max = r
                if c < col_min:
                    col_min = c
                if c > col_max:
                    col_max = c
    
    if not has_stone:
        # 空棋盘，返回中心区域
        center = board_size // 2
        count = 0
        result = np.empty((9, 2), dtype=np.int32)  # 最多3x3=9个点
        for r in range(max(0, center-1), min(board_size, center+2)):
            for c in range(max(0, center-1), min(board_size, center+2)):
                if count < 9:
                    result[count, 0] = r
                    result[count, 1] = c
                    count += 1
        return result[:count]
    
    # 扩展边界
    row_min = max(0, row_min - padding)
    row_max = min(board_size - 1, row_max + padding)
    col_min = max(0, col_min - padding)
    col_max = min(board_size - 1, col_max + padding)
    
    # 2. 计算重心
    sum_r, sum_c, count = 0.0, 0.0, 0
    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] != 0:
                sum_r += r
                sum_c += c
                count += 1
    
    center_r = sum_r / count
    center_c = sum_c / count
    
    # 3. 收集空位
    max_size = (row_max - row_min + 1) * (col_max - col_min + 1)
    candidates_r = np.empty(max_size, dtype=np.int32)
    candidates_c = np.empty(max_size, dtype=np.int32)
    distances = np.empty(max_size, dtype=np.float64)
    
    idx = 0
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            if board[r, c] == 0:
                candidates_r[idx] = r
                candidates_c[idx] = c
                distances[idx] = (r - center_r) ** 2 + (c - center_c) ** 2
                idx += 1
    
    # 4. 排序
    if idx == 0:
        # 返回空数组，但类型一致
        return np.empty((0, 2), dtype=np.int32)
    
    sorted_indices = np.argsort(distances[:idx])
    
    # 5. 构建结果数组
    result = np.empty((idx, 2), dtype=np.int32)
    for i in range(idx):
        pos = sorted_indices[i]
        result[i, 0] = candidates_r[pos]
        result[i, 1] = candidates_c[pos]
    
    return result


# ==================== 封装的类 (保持接口不变) ====================

class GomokuCore:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.l3_count = {1: 0, 2: 0}
        self.l4_count = {1: 0, 2: 0}
        self.l2_count = {1: 0, 2: 0}
        self.rl4_count = {1: 0, 2: 0}

    def reset(self):
        """重置游戏"""
        self.board.fill(0)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.l3_count = {1: 0, 2: 0}
        self.l4_count = {1: 0, 2: 0}
        self.l2_count = {1: 0, 2: 0}
        self.rl4_count = {1: 0, 2: 0}

    def place_stone(self, row, col):
        """
        尝试落子
        :return: (bool) 是否落子成功
        """
        if self.game_over:
            return False
        
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        if self.board[row, col] != 0:
            return False

        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # 调用 JIT 函数
        is_win, dl3, dl4, dl2, drl4 = jit_check_logic(
            self.board, row, col, self.current_player, self.board_size
        )
        
        if is_win:
            self.winner = self.current_player
            self.game_over = True
        else:
            target = 3 - self.current_player
            self.l3_count[target] += dl3
            self.l4_count[target] += dl4
            self.l2_count[target] += dl2
            self.rl4_count[target] += drl4
            self.current_player = 3 - self.current_player
            
        return True

    def _my_search_side(self, target, row, col, dr, dc, simu=False):
        """保持接口兼容"""
        win, record = jit_search_side(self.board, target, row, col, dr, dc, self.board_size)
        if not simu:
            self.l3_count[target] += record[0]
            self.l4_count[target] += record[1]
            self.l2_count[target] += record[2]
            self.rl4_count[target] += record[3]
        return win, tuple(record)

    def _check_win(self, row, col):
        """保持接口兼容"""
        is_win, dl3, dl4, dl2, drl4 = jit_check_logic(
            self.board, row, col, self.current_player, self.board_size
        )
        if not is_win:
            target = 3 - self.current_player
            self.l3_count[target] += dl3
            self.l4_count[target] += dl4
            self.l2_count[target] += dl2
            self.rl4_count[target] += drl4
        return is_win

    def simu_check(self, row, col):
        """
        模拟落子检查，不修改状态
        """
        if self.board[row, col] != 0:
            return False, self.l3_count, self.l4_count, self.l2_count, self.rl4_count
            
        original_val = self.board[row, col]
        self.board[row, col] = self.current_player
        
        try:
            is_win, dl3, dl4, dl2, drl4 = jit_check_logic(
                self.board, row, col, self.current_player, self.board_size
            )
            
            res_l3 = self.l3_count.copy()
            res_l4 = self.l4_count.copy()
            res_l2 = self.l2_count.copy()
            res_rl4 = self.rl4_count.copy()
            
            if not is_win:
                target = 3 - self.current_player
                res_l3[target] += dl3
                res_l4[target] += dl4
                res_l2[target] += dl2
                res_rl4[target] += drl4
                
            return is_win, res_l3, res_l4, res_l2, res_rl4
            
        finally:
            self.board[row, col] = original_val

    def get_search_range(self, padding=2):
        """计算搜索范围"""
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
        使用 JIT 加速的推荐点计算
        返回 list of tuples
        """
        result_array = jit_get_candidates_sort(self.board, self.board_size, padding)
        # 将 numpy 数组转换为 list of tuples
        return [(int(result_array[i, 0]), int(result_array[i, 1])) for i in range(len(result_array))]

    def get_board(self):
        """返回只读的numpy数组供显示层使用"""
        return self.board