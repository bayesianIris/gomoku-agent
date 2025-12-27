import pygame
import numpy as np

class GomokuUI:
    def __init__(self, board_size=15, cell_size=40):
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = cell_size  # 棋盘边缘留白
        
        # 计算窗口大小
        self.width = self.cell_size * (self.board_size - 1) + 2 * self.margin
        self.height = self.width
        
        # 颜色定义
        self.COLOR_BG = (220, 179, 92)  # 木头色
        self.COLOR_LINE = (0, 0, 0)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HINT = (200, 0, 0)   # 鼠标悬停提示色

        pygame.init()
        pygame.display.set_caption("Numpy Gomoku")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont('Arial', 24)

    def draw(self, board_array, current_player, game_over, winner, l2_count=None, l3_count=None, l4_count=None, rl4_count=None, last_move=None):
        """
        渲染函数：支持 l2_count 和 rl4_count
        """
        self.screen.fill(self.COLOR_BG)
        
        # 1. 画网格线
        for i in range(self.board_size):
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.width - self.margin, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_LINE, start_pos, end_pos, 1)
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.height - self.margin)
            pygame.draw.line(self.screen, self.COLOR_LINE, start_pos, end_pos, 1)

        # 2. 画星位
        center = self.board_size // 2
        star_points = [(center, center), (3, 3), (3, self.board_size-4), 
                       (self.board_size-4, 3), (self.board_size-4, self.board_size-4)]
        for r, c in star_points:
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            pygame.draw.circle(self.screen, self.COLOR_LINE, (x, y), 5)

        # 3. 画棋子
        black_stones = np.argwhere(board_array == 1)
        white_stones = np.argwhere(board_array == 2)
        for r, c in black_stones:
            self._draw_stone(r, c, self.COLOR_BLACK)
        for r, c in white_stones:
            self._draw_stone(r, c, self.COLOR_WHITE)

        # 上一步高亮
        if last_move is not None:
            r, c = last_move
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 4)

        # 4. 绘制文字信息（增加l2和rl4显示）
        l2_count = l2_count or {1: 0, 2: 0}
        l3_count = l3_count or {1: 0, 2: 0}
        l4_count = l4_count or {1: 0, 2: 0}
        rl4_count = rl4_count or {1: 0, 2: 0}
        p1_info = f"Black: L2={l2_count.get(1,0)} | L3={l3_count.get(1,0)} | L4={l4_count.get(1,0)} | RL4={rl4_count.get(1,0)}"
        p2_info = f"White: L2={l2_count.get(2,0)} | L3={l3_count.get(2,0)} | L4={l4_count.get(2,0)} | RL4={rl4_count.get(2,0)}"
        surf_p1 = self.font.render(p1_info, True, (0, 0, 0))
        surf_p2 = self.font.render(p2_info, True, (255, 255, 255))
        self.screen.blit(surf_p1, (10, 5))
        self.screen.blit(surf_p2, (10, self.height - 30))

        # 5. 游戏结束文字
        if game_over:
            text = f"Player {winner} Wins! (Click to Reset)"
            text_surface = self.font.render(text, True, (255, 0, 0))
            text_rect = text_surface.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def _draw_stone(self, row, col, color):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2
        pygame.draw.circle(self.screen, color, (x, y), radius)

    def convert_mouse_to_grid(self, mouse_pos):
        x, y = mouse_pos
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        return row, col