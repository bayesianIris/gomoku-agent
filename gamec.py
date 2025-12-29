import numpy as np
from numba import jit, types
from numba.core import cgutils
import numba

# ===================== Numba优化的核心函数 =====================

@jit(nopython=True, cache=True)
def jit_search_side(board, board_size, target, row, col, dr, dc):
    """
    使用Numba JIT编译的棋型检查函数
    返回: (is_win, l3_count, l4_count, l2_count, rl4_count)
    """
    count = 1
    count_middle_blank = 0
    count_edge_blocked_blank = 0
    side = 0
    zero_flag = 0
    count_hard = 1
    count_this_dir_middle_blank = 0
    
    # 正向检查
    for i in range(1, 7):
        r = row + dr * i
        c = col + dc * i
        
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
            elif board[r, c] == 3 - target:
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
    
    # 反向检查
    count_this_dir_middle_blank = 0
    for i in range(1, 7):
        r = row - dr * i
        c = col - dc * i
        
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
            elif board[r, c] == 3 - target:
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
    
    # 判断棋型
    l3 = 0
    l4 = 0
    l2 = 0
    rl4 = 0
    is_win = False
    
    if count_hard >= 5:
        is_win = True
    elif count == 3 and count_middle_blank <= 1:
        if side == 2:
            l3 = 1
        elif side == 1 and count_edge_blocked_blank == 1:
            l3 = 1
        elif side == 0 and count_edge_blocked_blank == 2 and count_middle_blank == 1:
            l3 = 1
    elif count == 4 and count_middle_blank <= 1:
        if side + count_edge_blocked_blank >= 1:
            l4 = 1
    elif count == 4 and count_middle_blank == 0:
        if side + count_edge_blocked_blank >= 1:
            rl4 = 1
    elif count == 2 and count_middle_blank <= 1:
        if side + count_edge_blocked_blank == 2:
            l2 = 1
    
    return is_win, (l3, l4, l2, rl4)


@jit(nopython=True, cache=True)
def jit_get_search_range(board, board_size, padding):
    """
    使用Numba JIT优化的搜索范围计算
    """
    # 检查是否有棋子
    has_stone = False
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] != 0:
                has_stone = True
                break
        if has_stone:
            break
    
    if not has_stone:
        center = board_size // 2
        return center - 1, center + 2, center - 1, center + 2
    
    # 找出最小和最大的行列索引
    row_min = board_size
    row_max = -1
    col_min = board_size
    col_max = -1
    
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] != 0:
                if i < row_min:
                    row_min = i
                if i > row_max:
                    row_max = i
                if j < col_min:
                    col_min = j
                if j > col_max:
                    col_max = j
    
    # 应用padding
    row_min = max(0, row_min - padding)
    row_max = min(board_size, row_max + padding + 1)
    col_min = max(0, col_min - padding)
    col_max = min(board_size, col_max + padding + 1)
    
    return row_min, row_max, col_min, col_max


@jit(nopython=True, cache=True)
def jit_get_candidates_sort(board, board_size, row_min, row_max, col_min, col_max):
    """
    使用Numba JIT优化的候选点获取与排序
    返回排序后的候选点列表
    """
    # 第一步：找出所有空位
    candidates = []
    for i in range(row_min, row_max):
        for j in range(col_min, col_max):
            if board[i, j] == 0:
                candidates.append((i, j))
    
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    # 第二步：计算棋局重心
    stone_count = 0
    center_r = 0.0
    center_c = 0.0
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] != 0:
                stone_count += 1
                center_r += i
                center_c += j
    
    if stone_count > 0:
        center_r /= stone_count
        center_c /= stone_count
    else:
        center_r = board_size / 2.0
        center_c = board_size / 2.0
    
    # 第三步：计算距离并排序
    n = len(candidates)
    dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        r, c = candidates[i]
        dr = r - center_r
        dc = c - center_c
        dists[i] = dr * dr + dc * dc
    
    # 使用argsort进行排序
    sorted_indices = np.argsort(dists)
    
    # 构建结果数组
    result = np.empty((n, 2), dtype=np.int32)
    for i in range(n):
        idx = sorted_indices[i]
        result[i, 0] = candidates[idx][0]
        result[i, 1] = candidates[idx][1]
    
    return result


# ===================== GomokuCore 类 =====================

class GomokuCore:
    def __init__(self, board_size=15):
        self.board_size = board_size
        # 0: 空, 1: 黑棋, 2: 白棋
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.current_player = 1  # 黑棋先行
        self.winner = None
        self.game_over = False
        self.last_move = None
        # 活3活4解算存
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
        
        # 检查胜负
        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        else:
            # 切换棋手 (1 -> 2, 2 -> 1)
            self.current_player = 3 - self.current_player
        return True

    def _my_search_side(self, target, row, col, dr, dc, simu=False):
        """
        使用Numba优化版本的搜索函数
        """
        win, record = jit_search_side(self.board, self.board_size, target, row, col, dr, dc)
        
        if not simu:
            self.l3_count[target] += record[0]
            self.l4_count[target] += record[1]
            self.l2_count[target] += record[2]
            self.rl4_count[target] += record[3]
        
        return win, record

    def _check_win(self, row, col):
        """
        基于最后落子的位置，判断是否获胜
        只需检查四个方向：横、竖、主对角、副对角
        """
        target = self.current_player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            win, _ = self._my_search_side(target, row, col, dr, dc)
            if win:
                return True
        
        # 活三、活四、活二、rl4 统计
        inits = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        target = 3 - self.current_player
        for dr, dc in inits:
            _r, _c = row + dr, col + dc
            if not (0 <= _r < self.board_size and 0 <= _c < self.board_size):
                continue
            if self.board[_r, _c] != 0:
                next_r, next_c = _r + dr, _c + dc
                if not (0 <= next_r < self.board_size and 0 <= next_c < self.board_size):
                    continue
                if self.board[next_r, next_c] != target:
                    continue
                try:
                    rec = rec2 = (0, 0, 0, 0)
                    self.board[row, col] = 0
                    _, rec2 = self._my_search_side(target, next_r, next_c, dr, dc, simu=True)
                    if rec2 == (0, 0, 0, 0):
                        continue
                    self.board[row, col] = self.current_player
                    _, rec = self._my_search_side(target, next_r, next_c, dr, dc, simu=True)
                finally:
                    self.board[row, col] = self.current_player
                    self.l3_count[target] += rec[0] - rec2[0]
                    self.l4_count[target] += rec[1] - rec2[1]
                    self.l2_count[target] += rec[2] - rec2[2]
                    self.rl4_count[target] += rec[3] - rec2[3]
            elif self.board[_r, _c] == target:
                try:
                    rec = rec2 = (0, 0, 0, 0)
                    self.board[row, col] = 0
                    _, rec2 = self._my_search_side(target, _r, _c, dr, dc, simu=True)
                    if rec2 == (0, 0, 0, 0):
                        continue
                    self.board[row, col] = self.current_player
                    _, rec = self._my_search_side(target, _r, _c, dr, dc, simu=True)
                finally:
                    self.board[row, col] = self.current_player
                    self.l3_count[target] += rec[0] - rec2[0]
                    self.l4_count[target] += rec[1] - rec2[1]
                    self.l2_count[target] += rec[2] - rec2[2]
                    self.rl4_count[target] += rec[3] - rec2[3]
        return False

    def simu_check(self, row, col):
        bkup_l3 = self.l3_count.copy()
        bkup_l4 = self.l4_count.copy()
        bkup_l2 = self.l2_count.copy()
        bkup_rl4 = self.rl4_count.copy()
        try:
            win = self._check_win(row, col)
            res_l3 = self.l3_count.copy()
            res_l4 = self.l4_count.copy()
            res_l2 = self.l2_count.copy()
            res_rl4 = self.rl4_count.copy()
            return win, res_l3, res_l4, res_l2, res_rl4
        finally:
            self.l3_count = bkup_l3
            self.l4_count = bkup_l4
            self.l2_count = bkup_l2
            self.rl4_count = bkup_rl4

    def get_search_range(self, padding=2):
        """
        计算有子区域的 Bounding Box (边界框)，用于减少搜索范围
        使用Numba优化版本
        """
        return jit_get_search_range(self.board, self.board_size, padding)

    def recommand_positions(self, padding=2):
        """
        返回候选落子点，按照离当前棋局重心距离从近到远排序
        使用Numba优化版本
        """
        row_min, row_max, col_min, col_max = self.get_search_range(padding)
        sorted_candidates = jit_get_candidates_sort(self.board, self.board_size, row_min, row_max, col_min, col_max)
        
        # 转换为 Python 原生 list (tuple格式)
        return [tuple(pos) for pos in sorted_candidates]

    def get_board(self):
        """返回只读的numpy数组供显示层使用"""
        return self.board