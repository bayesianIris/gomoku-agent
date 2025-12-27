import numpy as np

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
    def _my_search_side(self,target, row, col, dr, dc,simu=False):
        count = 1
        count_middle_blank = 0 # 记录中间空格数,中空零
        count_edge_blocked_blank = 0 # 边堵零
        side = 0  # 边不堵零
        zero_flag = 0
        count_hard = 1 # 硬连接
        count_this_dir_middle_blank = 0
        # 正向检查
        for i in range(1, 7):
            r, c = row + dr * i, col + dc * i
            # 超边界截断
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                if zero_flag == 1:
                    count_edge_blocked_blank += 1
                    zero_flag = 0
                break
            if zero_flag == 1:
                zero_flag = 0
                if self.board[r, c] == target:
                    count_middle_blank += 1
                    count_this_dir_middle_blank += 1
                elif self.board[r, c] == 3-target:
                    count_edge_blocked_blank += 1
                    break
                else:
                    side += 1
                    break
            # 检查是否是己方棋子
            if self.board[r, c] == target:
                count += 1
                if count_this_dir_middle_blank == 0:
                    count_hard += 1
            # 检查活还是死
            elif self.board[r, c] == 0:
                zero_flag = 1
            # 碰到对方棋子肯定是死
            else: break
        # 反向检查
        count_this_dir_middle_blank = 0 # 单方向中空零重置
        for i in range(1, 7):
            r, c = row - dr * i, col - dc * i
            # 超边界截断
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                if zero_flag == 1:
                    count_edge_blocked_blank += 1
                    zero_flag = 0
                break
            if zero_flag == 1:
                zero_flag = 0
                if self.board[r, c] == target:
                    count_middle_blank += 1
                    count_this_dir_middle_blank += 1
                elif self.board[r, c] == 3-target:
                    count_edge_blocked_blank += 1
                    break
                else:
                    side += 1
                    break
            # 检查是否是己方棋子
            if self.board[r, c] == target:
                count += 1
                if count_this_dir_middle_blank == 0:
                    count_hard += 1
            # 检查活还是死
            elif self.board[r, c] == 0:
                zero_flag = 1
            # 碰到对方棋子肯定是死
            else: break
        record=[0,0,0,0]
        # 对于三四的判断
        if count_hard >= 5:
            return True, record
        if count == 3 and count_middle_blank <= 1:
            if side == 2:
                if not simu: self.l3_count[target] += 1
                record[0]+=1
            elif side == 1 and count_edge_blocked_blank == 1:
                if not simu: self.l3_count[target] += 1
                record[0]+=1
            elif side == 0 and count_edge_blocked_blank == 2 and count_middle_blank == 1:
                if not simu: self.l3_count[target] += 1
                record[0]+=1
        elif count == 4 and count_middle_blank <= 1:
            if side + count_edge_blocked_blank >= 1:
                if not simu: self.l4_count[target] += 1
                record[1]+=1
        elif count == 4 and count_middle_blank == 0:
            if side + count_edge_blocked_blank >= 1:
                if not simu: self.rl4_count[target] += 1
                record[3]+=1
        elif count == 2 and count_middle_blank <= 1:
            if side + count_edge_blocked_blank == 2:
                if not simu: self.l2_count[target] += 1
                record[2]+=1
        return False, tuple(record)
    
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
        inits = [(0,1),(1,0),(1,1),(1,-1),(0,-1),(-1,0),(-1,-1),(-1,1)]
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
        :param padding: 向外扩充的格子数 (通常为了检测周围的落子点)
        :return: (row_min, row_max, col_min, col_max) 注意：max是切片用的，包含padding
        """
        row_has_stone = self.board.any(axis=1)
        col_has_stone = self.board.any(axis=0)
        # 如果棋盘是空的 (没有任何落子)，返回天元附近的范围
        if not np.any(row_has_stone):
            center = self.board_size // 2
            return center-1, center+2, center-1, center+2
        # 2. 获取有子行列的索引列表
        row_indices = np.where(row_has_stone)[0]
        col_indices = np.where(col_has_stone)[0]

        # 3. 计算边界 (利用 numpy 数组的第一个和最后一个元素)
        # 这里的 max 需要 +1 是因为 Python 切片是左闭右开区间 [start, end)
        # 使用 clip 防止越界
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
        # 1. 获取搜索范围内的所有空位坐标 (局部坐标)
        # 注意：这里切片是视图操作，非常快
        sub_board = self.board[row_min:row_max, col_min:col_max]
        empty_indices = np.argwhere(sub_board == 0)
        
        # 如果没有空位（极罕见情况），直接返回空列表
        if len(empty_indices) == 0:
            return []

        # 2. 将局部坐标转换为全局坐标
        # empty_indices 是 [[r1, c1], [r2, c2]...]
        # 我们直接加上偏移量 (利用numpy广播机制)
        candidates = empty_indices + [row_min, col_min]

        # 3. 计算棋局的“重心” (Center of Gravity)
        # 如果棋盘是空的，重心就是天元
        # 这一步是为了让搜索更有启发性：优先搜棋子密集区域的中心
        stone_indices = np.argwhere(self.board != 0)
        if len(stone_indices) > 0:
            # 计算所有棋子的平均坐标 (float)，这就是重心
            center_r, center_c = np.mean(stone_indices, axis=0)
        else:
            center_r, center_c = self.board_size // 2, self.board_size // 2

        # 4. 计算每个候选点到重心的距离 (使用距离的平方避免开根号，速度更快)
        # candidates[:, 0] 是所有行坐标， candidates[:, 1] 是所有列坐标
        # dist = (r - cr)^2 + (c - cc)^2
        dists = (candidates[:, 0] - center_r)**2 + (candidates[:, 1] - center_c)**2

        # 5. 根据距离排序
        # argsort 返回的是排序后的索引
        sorted_indices = np.argsort(dists)
        
        # 6. 利用排序索引重新排列 candidates，并转回 list of tuples
        # 这一步是 NumPy 的高级索引，速度很快
        sorted_candidates = candidates[sorted_indices]
        
        # 转换为 Python 原生 list (tuple格式)，方便后续遍历
        return [tuple(pos) for pos in sorted_candidates]
    def get_board(self):
        """返回只读的numpy数组供显示层使用"""
        return self.board