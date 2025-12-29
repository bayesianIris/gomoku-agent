# %%
from game import *
from display import GomokuUI
from copy import deepcopy
import pygame
import numpy as np
import numpy as np
import copy
import time

# 定义无穷大
INF = 999999999

def advantage_f(l2, l3, l4, rl4, my_player):
    """
    局势评估函数（细化版）
    l2: 活二计数 dict
    l3: 活三计数 dict
    l4: 冲四计数 dict
    rl4: 必杀四/连五计数 dict
    """
    opponent = 3 - my_player
    score = 0

    # l2得1分，双l2得3分
    my_l2 = l2.get(my_player, 0)
    opp_l2 = l2.get(opponent, 0)
    score += (my_l2 * 1 + (2 if my_l2 >= 2 else 0))
    score -= (opp_l2 * 1 + (2 if opp_l2 >= 2 else 0))

    # l3得5分
    my_l3 = l3.get(my_player, 0)
    opp_l3 = l3.get(opponent, 0)
    score += my_l3 * 5
    score -= opp_l3 * 5

    # l4（冲四）得20分
    my_l4 = l4.get(my_player, 0)
    opp_l4 = l4.get(opponent, 0)
    score += my_l4 * 20
    score -= opp_l4 * 20

    # rl4（必杀四/连五）得1000分
    my_rl4 = rl4.get(my_player, 0)
    opp_rl4 = rl4.get(opponent, 0)
    score += my_rl4 * 1000
    score -= opp_rl4 * 1000

    # 杀棋判定（我方）
    # 对方l4、rl4等于0，自己l3+l4>1且l4>1，+500分
    if opp_l4 == 0 and opp_rl4 == 0:
        if (my_l3 + my_l4) > 1 and my_l4 > 1:
            score += 500
    # 对方l4、l3、rl4均为0，自己有两个l3，+100分
    if opp_l4 == 0 and opp_l3 == 0 and opp_rl4 == 0:
        if my_l3 >= 2:
            score += 100
    
    # 杀棋判定（对手镜像）
    # 我方l4、rl4等于0，对手l3+l4>1且l4>1，-500分
    if my_l4 == 0 and my_rl4 == 0:
        if (opp_l3 + opp_l4) > 1 and opp_l4 > 1:
            score -= 500
    # 我方l4、l3、rl4均为0，对手有两个l3，-100分
    if my_l4 == 0 and my_l3 == 0 and my_rl4 == 0:
        if opp_l3 >= 2:
            score -= 100

    return score

def advantage_vector(l3, l4, my_player):
    opponent = 3 - my_player
    vec = np.array([
        l4.get(my_player, 0),l4.get(opponent, 0),
        l3.get(my_player, 0),l3.get(opponent, 0)
    ])
    # 其中，在maximizing中，应该关注我方的L4 L3，即vec[0], vec[2]
    # 在minimizing中，应该关注我方的L4 L3 以及对方的L4，即vec[0], vec[2], vec[1]
    return vec

def basic_ai_move(game:GomokuCore, my_player, depth=3):
    score, move = minimax(game, depth, -INF, INF, True, my_player)
    print(f"AI selects move {move} with score {score}")
    return move

def kill(game:GomokuCore, depth, alpha, beta, is_maximizing, my_player):
    """
    带排序剪枝(Beam Search)的 Minimax
    """
    # 1. 检查游戏结束
    if game.game_over:
        if game.winner == my_player:
            return INF, depth
        elif game.winner == (3 - my_player):
            return -INF, depth
        else:
            return 0, depth
    adv_saved = advantage_vector(game.l3_count, game.l4_count, my_player)
    # 2. 达到深度限制 (Leaf Node)
    if depth == 0:
        return advantage_f(game.l3_count, game.l4_count, my_player), depth
    # 3. 生成并评估所有候选状态 (这是本层的核心开销)
    candidates = game.recommand_positions()
    scored_moves = [] # 格式: (score, next_game_instance)


    # score += (l4.get(my_player, 0) - l4.get(opponent, 0)) * 20
    # score += (l3.get(my_player, 0) - l3.get(opponent, 0)) * 1



    for r, c in candidates:
        # 这里的 deepcopy 是必须的，为了计算该状态的得分
        next_game = copy.deepcopy(game)
        success = next_game.place_stone(r, c)
        
        if success:
            # 如果这一步直接导致游戏结束，赋予极值，确保它会被排在第一位
            if next_game.game_over:
                current_score = INF if next_game.winner == my_player else -INF
            else:
                # 计算启发式分数 (Heuristic Score)
                current_score = advantage_f(next_game.l3_count, next_game.l4_count, my_player)
            current_vec = advantage_vector(next_game.l3_count, next_game.l4_count, my_player)
            chaa = current_vec - adv_saved # chaa-差
            judgement1 = (chaa[0] !=0 or chaa[2] !=0 ) and is_maximizing
            judgement2 = (chaa[2]!=0 or chaa[1] !=0 or chaa[0] !=0 )and (not is_maximizing)
            if judgement1 or judgement2 or next_game.game_over:
                scored_moves.append((current_score, next_game))
    print(f"len:{len(scored_moves)}, depth:{depth}")           
    # 4. 排序与截断 (Sorting & Pruning)
    # 如果没有合法走法 (比如平局填满)，直接返回平局分
    if not scored_moves and is_maximizing:
        return 0, depth
    elif not scored_moves and (not is_maximizing):
        scored_moves.append((current_score, next_game))
    max_eval = -INF
    dpt_best = -1
    if is_maximizing:
        # Max层：希望分数越高越好，所以降序排列 (Reverse=True)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for _, next_game_state in scored_moves:
            # 递归时不需要再 deepcopy 了，因为 scored_moves 里存的已经是独立的副本
            eval_score, dpt = kill(next_game_state, depth - 1, alpha, beta, False, my_player)
            dpt_best = max(dpt_best, dpt)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, dpt_best
    else:
        # Min层：对手希望分数越低越好(对我越不利)，所以升序排列 (Reverse=False)
        scored_moves.sort(key=lambda x: x[0], reverse=False)
        min_eval = INF
        for _, next_game_state in scored_moves:
            eval_score, dpt = kill(next_game_state, depth - 1, alpha, beta, True, my_player)
            dpt_best = max(dpt_best, dpt)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, dpt_best


# %%
def minimax(game:GomokuCore, depth, alpha, beta, is_maximizing, my_player, top_k=30):
    """
    带排序剪枝(Beam Search)的 Minimax
    """
    # 1. 检查游戏结束
    if game.game_over:
        if game.winner == my_player:
            return INF, (0,0)
        elif game.winner == (3 - my_player):
            return -INF, (0,0)
        else:
            return 0, (0,0)
    # 2. 达到深度限制 (Leaf Node)
    if depth == 0:
        return advantage_f(game.l2_count, game.l3_count, game.l4_count, game.rl4_count, my_player), (0,0)

    # 3. 生成并评估所有候选状态 (这是本层的核心开销)
    candidates = game.recommand_positions()
    scored_moves = [] # 格式: (score, next_game_instance, (r, c))

    for r, c in candidates:
        next_game = copy.deepcopy(game)
        success = next_game.place_stone(r, c)
        if success:
            if next_game.game_over:
                current_score = INF if next_game.winner == my_player else -INF
            else:
                current_score = advantage_f(
                    next_game.l2_count,
                    next_game.l3_count,
                    next_game.l4_count,
                    next_game.rl4_count,
                    my_player
                )
            scored_moves.append((current_score, next_game, (r, c)))

    if not scored_moves:
        return 0, (0,0)
    final_move = scored_moves[0][2]

    if is_maximizing:
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        best_moves = scored_moves[:top_k] if depth != 4 else scored_moves
        max_eval = -INF
        for _, next_game_state, move in best_moves:
            eval_score, _ = minimax(next_game_state, depth - 1, alpha, beta, False, my_player, top_k)
            if eval_score > max_eval:
                max_eval = eval_score
                alpha = max(alpha, eval_score)
                final_move = move
            # max_eval = max(max_eval, eval_score)
            # alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, final_move
    else:
        scored_moves.sort(key=lambda x: x[0], reverse=False)
        best_moves = scored_moves[:top_k]
        min_eval = INF
        for _, next_game_state, move in best_moves:
            eval_score, _ = minimax(next_game_state, depth - 1, alpha, beta, True, my_player, top_k)
            if eval_score < min_eval:
                min_eval = eval_score
                beta = min(beta, eval_score)
                final_move = move
            # min_eval = min(min_eval, eval_score)
            # beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, final_move

# %%

def main():
    # 1. 实例化逻辑层和显示层
    game = GomokuCore(board_size=15)
    ui = GomokuUI(board_size=15, cell_size=40)
    player = 3- game.current_player
    clock = pygame.time.Clock()
    running = True

    while running:
        # --- 事件处理 (Input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN and game.current_player == player:
                if event.button == 1: # 左键点击
                    if game.game_over:
                        game.reset()
                    else:
                        # UI 负责将点击转换成坐标
                        row, col = ui.convert_mouse_to_grid(event.pos)
                        # Logic 负责判断是否合法并更新数据
                        game.place_stone(row, col)
        
        # --- 渲染循环 (Output) ---
        # 显示层只需要当前的数据状态
        ui.draw(
            board_array=game.get_board(), 
            current_player=game.current_player,
            game_over=game.game_over,
            winner=game.winner,
            l2_count=game.l2_count,      
            l3_count=game.l3_count,
            l4_count=game.l4_count,
            rl4_count=game.rl4_count,    
            last_move=game.last_move
        )
        if player != game.current_player:
            row,col = basic_ai_move(game, my_player=game.current_player, depth=4)
            print("AI落子:",row,col)
            game.place_stone(row, col)
        clock.tick(30) # 限制30帧，节省资源

    pygame.quit()

if __name__ == "__main__":
    main()


