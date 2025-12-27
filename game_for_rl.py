import numpy as np

class GomokuCore:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None
        self.l3_count = {1: 0, 2: 0}
        self.l4_count = {1: 0, 2: 0}

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.last_move = None

    def place_stone(self, row, col):
        if self.game_over or not (0 <= row < self.board_size and 0 <= col < self.board_size) or self.board[row, col] != 0:
            return False
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        else:
            self.current_player = 3 - self.current_player
        return True

    def _my_search_side(self, target, row, col, dr, dc, simu=False):
        count = count_hard = 1
        count_middle_blank = count_edge_blocked_blank = side = zero_flag = 0
        for i in range(1, 7):
            r, c = row + dr * i, col + dc * i
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                if zero_flag: count_edge_blocked_blank += 1; zero_flag = 0
                break
            if zero_flag:
                zero_flag = 0
                if self.board[r, c] == target:
                    count_middle_blank += 1
                elif self.board[r, c] == 3 - target:
                    count_edge_blocked_blank += 1
                    break
                else:
                    side += 1
                    break
            if self.board[r, c] == target:
                count += 1
                if count_middle_blank == 0: count_hard += 1
            elif self.board[r, c] == 0:
                zero_flag = 1
            else:
                break
        for i in range(1, 7):
            r, c = row - dr * i, col - dc * i
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                if zero_flag: count_edge_blocked_blank += 1; zero_flag = 0
                break
            if zero_flag:
                zero_flag = 0
                if self.board[r, c] == target:
                    count_middle_blank += 1
                elif self.board[r, c] == 3 - target:
                    count_edge_blocked_blank += 1
                    break
                else:
                    side += 1
                    break
            if self.board[r, c] == target:
                count += 1
                if count_middle_blank == 0: count_hard += 1
            elif self.board[r, c] == 0:
                zero_flag = 1
            else:
                break
        record = [0, 0]
        if count_hard >= 5: return True, record
        if count == 3 and count_middle_blank <= 1:
            if side == 2 or (side == 1 and count_edge_blocked_blank == 1) or (side == 0 and count_edge_blocked_blank == 2 and count_middle_blank == 1):
                if not simu: self.l3_count[target] += 1
                record[0] += 1
        elif count == 4 and count_middle_blank <= 1:
            if side + count_edge_blocked_blank >= 1:
                if not simu: self.l4_count[target] += 1
                record[1] += 1
        return False, tuple(record)

    def _check_win(self, row, col):
        target = self.current_player
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            win, _ = self._my_search_side(target, row, col, dr, dc)
            if win: return True
        inits = [(0,1),(1,0),(1,1),(1,-1),(0,-1),(-1,0),(-1,-1),(-1,1)]
        target = 3 - self.current_player
        for dr, dc in inits:
            _r, _c = row + dr, col + dc
            if not (0 <= _r < self.board_size and 0 <= _c < self.board_size): continue
            if self.board[_r, _c] != 0:
                next_r, next_c = _r + dr, _c + dc
                if not (0 <= next_r < self.board_size and 0 <= next_c < self.board_size): continue
                if self.board[next_r, next_c] != target: continue
                try:
                    self.board[row, col] = 0
                    _, rec2 = self._my_search_side(target, next_r, next_c, dr, dc, simu=True)
                    if rec2 == (0, 0): continue
                    self.board[row, col] = self.current_player
                    _, rec = self._my_search_side(target, next_r, next_c, dr, dc, simu=True)
                finally:
                    self.board[row, col] = self.current_player
                    self.l3_count[target] += rec[0] - rec2[0]
                    self.l4_count[target] += rec[1] - rec2[1]
            elif self.board[_r, _c] == target:
                try:
                    self.board[row, col] = 0
                    _, rec2 = self._my_search_side(target, _r, _c, dr, dc, simu=True)
                    if rec2 == (0, 0): continue
                    self.board[row, col] = self.current_player
                    _, rec = self._my_search_side(target, _r, _c, dr, dc, simu=True)
                finally:
                    self.board[row, col] = self.current_player
                    self.l3_count[target] += rec[0] - rec2[0]
                    self.l4_count[target] += rec[1] - rec2[1]
        return False

    def simu_check(self, row, col):
        bkup_l3 = self.l3_count.copy()
        bkup_l4 = self.l4_count.copy()
        try:
            win = self._check_win(row, col)
            res_l3 = self.l3_count.copy()
            res_l4 = self.l4_count.copy()
            return win, res_l3, res_l4
        finally:
            self.l3_count = bkup_l3
            self.l4_count = bkup_l4

    def get_search_range(self, padding=2):
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
        row_min, row_max, col_min, col_max = self.get_search_range(padding)
        sub_board = self.board[row_min:row_max, col_min:col_max]
        empty_indices = np.argwhere(sub_board == 0)
        if len(empty_indices) == 0: return []
        candidates = empty_indices + [row_min, col_min]
        stone_indices = np.argwhere(self.board != 0)
        if len(stone_indices) > 0:
            center_r, center_c = np.mean(stone_indices, axis=0)
        else:
            center_r, center_c = self.board_size // 2, self.board_size // 2
        dists = (candidates[:, 0] - center_r)**2 + (candidates[:, 1] - center_c)**2
        sorted_indices = np.argsort(dists)
        sorted_candidates = candidates[sorted_indices]
        return [tuple(pos) for pos in sorted_candidates]

    def get_board(self):
        return self.board