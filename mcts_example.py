"""
MCTS Agent 使用示例（带可视化）
展示如何使用蒙特卡洛树搜索 Agent 来玩五子棋，并实时显示棋盘状态
"""

import numpy as np
import copy
from game import GomokuCore
from mcts import MCTSAgent
from display import GomokuUI
import time
import pygame


def play_game_with_mcts(board_size=15, num_simulations=500, verbose=True, visualize=True):
    """
    使用 MCTS Agent 进行单次游戏演示（支持可视化）
    :param board_size: 棋盘大小
    :param num_simulations: 每步的 MCTS 模拟次数
    :param verbose: 是否打印详细信息
    :param visualize: 是否显示棋盘 UI
    :return: 游戏结果
    """
    # 初始化游戏
    game = GomokuCore(board_size=board_size)
    
    # 初始化 UI（如果启用可视化）
    ui = None
    if visualize:
        ui = GomokuUI(board_size=board_size, cell_size=40)
        clock = pygame.time.Clock()
    
    # 创建两个 MCTS Agent
    agent_black = MCTSAgent(game, num_simulations=num_simulations, c_puct=1.0)
    agent_white = MCTSAgent(game, num_simulations=num_simulations, c_puct=1.0)
    
    agents = {1: agent_black, 2: agent_white}
    move_count = 0
    
    print(f"\n{'='*60}")
    print(f"Game Start: {board_size}x{board_size} board")
    print(f"MCTS Simulations per move: {num_simulations}")
    print(f"Visualization: {'ON' if visualize else 'OFF'}")
    print(f"{'='*60}\n")
    
    while not game.game_over and move_count < board_size * board_size:
        # 检查关闭窗口事件
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
        
        move_count += 1
        player = game.current_player
        player_name = "Black (1)" if player == 1 else "White (2)"
        
        if verbose:
            print(f"Move {move_count}: {player_name}")
        
        # 获取 Agent
        agent = agents[player]
        
        # 搜索最优落子
        start_time = time.time()
        best_move = agent.get_best_move(game, return_value=False)
        search_time = time.time() - start_time
        
        if best_move is None:
            if verbose:
                print(f"{player_name} has no valid moves! Game ends.")
            break
        
        # 执行落子
        game.place_stone(best_move[0], best_move[1])
        
        if verbose:
            print(f"  Move: {best_move}")
            print(f"  Search time: {search_time:.2f}s")
            print(f"  L2 count - Black: {game.l2_count[1]}, White: {game.l2_count[2]}")
            print(f"  L3 count - Black: {game.l3_count[1]}, White: {game.l3_count[2]}")
            print(f"  L4 count - Black: {game.l4_count[1]}, White: {game.l4_count[2]}")
            print(f"  RL4 count - Black: {game.rl4_count[1]}, White: {game.rl4_count[2]}")
            print()
        
        # 渲染棋盘
        if visualize:
            ui.draw(
                board_array=game.get_board(),
                current_player=game.current_player,
                game_over=game.game_over,
                winner=game.winner,
                l2_count=game.l2_count,
                l3_count=game.l3_count,
                l4_count=game.l4_count,
                rl4_count=game.rl4_count,
                last_move=best_move
            )
            clock.tick(2)  # 控制刷新速度，便于观看
        
        # 如果游戏结束
        if game.game_over:
            winner = "Black" if game.winner == 1 else "White"
            print(f"\n{'='*60}")
            print(f"Game Over! {winner} (Player {game.winner}) wins!")
            print(f"Total moves: {move_count}")
            print(f"{'='*60}\n")
            
            if visualize:
                # 显示最终结果 3 秒
                for _ in range(3):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return game.winner
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
                    clock.tick(1)
                pygame.quit()
            
            return game.winner
    
    if visualize:
        pygame.quit()
    
    print(f"\nGame ended after {move_count} moves (board full or no valid moves)")
    return None


def demonstrate_mcts_analysis(board_size=15, visualize=True):
    """
    演示 MCTS 对特定局面的分析（支持可视化）
    """
    game = GomokuCore(board_size=board_size)
    
    # 初始化 UI（如果启用可视化）
    ui = None
    if visualize:
        ui = GomokuUI(board_size=board_size, cell_size=40)
        clock = pygame.time.Clock()
    
    # 设置一个简单的初始局面
    game.place_stone(7, 7)
    print(f"Initial position: (7, 7) - Black")
    
    game.place_stone(7, 8)
    print(f"Move 2: (7, 8) - White")
    
    game.place_stone(8, 7)
    print(f"Move 3: (8, 7) - Black\n")
    
    # 显示初始局面
    if visualize:
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
        clock.tick(1)
    
    # 进行 MCTS 搜索
    agent = MCTSAgent(game, num_simulations=1000, c_puct=1.0)
    print("Performing MCTS search with 1000 simulations...\n")
    
    start_time = time.time()
    best_move = agent.get_best_move(game, return_value=True)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.2f}s")
    print(f"Best move: {best_move[0]}")
    print(f"Evaluation: {best_move[1]:.2f} (positive = good for Black)\n")
    
    # 打印树的信息
    agent.print_tree_info()
    
    # 获取所有候选的概率分布
    move_probs = agent.get_move_probabilities()
    move_values = agent.get_move_values()
    
    print("\n=== Top 10 candidate moves ===")
    print(f"{'Move':<10} {'Probability':<15} {'Win Rate':<10}")
    print("-" * 35)
    
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (move, prob) in enumerate(sorted_moves[:10]):
        win_rate = move_values.get(move, 0)
        print(f"{str(move):<10} {prob:<15.4f} {win_rate:<10.2f}")
    
    # 可视化显示最佳落子点
    if visualize:
        # 临时落子最佳点进行预览
        game_preview = copy.deepcopy(game)
        game_preview.place_stone(best_move[0][0], best_move[0][1])
        
        ui.draw(
            board_array=game_preview.get_board(),
            current_player=game_preview.current_player,
            game_over=game_preview.game_over,
            winner=game_preview.winner,
            l2_count=game_preview.l2_count,
            l3_count=game_preview.l3_count,
            l4_count=game_preview.l4_count,
            rl4_count=game_preview.rl4_count,
            last_move=best_move[0]
        )
        print(f"\n显示预测最佳落子点 {best_move[0]} 的局面...")
        
        # 显示 3 秒
        for _ in range(3):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            clock.tick(1)
        
        pygame.quit()


def compare_positions(game_state, num_simulations=500, visualize=True):
    """
    对当前局面的所有候选落子点进行评估对比（支持可视化）
    :param game_state: GomokuCore 游戏状态
    :param num_simulations: 每个位置的 MCTS 模拟次数
    :param visualize: 是否显示棋盘
    """
    agent = MCTSAgent(game_state, num_simulations=num_simulations, c_puct=1.0)
    
    # 初始化 UI（如果启用可视化）
    ui = None
    if visualize:
        ui = GomokuUI(board_size=game_state.board_size, cell_size=40)
        clock = pygame.time.Clock()
        
        # 显示当前局面
        ui.draw(
            board_array=game_state.get_board(),
            current_player=game_state.current_player,
            game_over=game_state.game_over,
            winner=game_state.winner,
            l2_count=game_state.l2_count,
            l3_count=game_state.l3_count,
            l4_count=game_state.l4_count,
            rl4_count=game_state.rl4_count,
            last_move=game_state.last_move
        )
        print("开始分析所有候选落子点...")
        clock.tick(1)
    
    print("Analyzing all candidate moves...")
    agent.get_best_move(game_state)
    
    move_values = agent.get_move_values()
    move_probs = agent.get_move_probabilities()
    
    print(f"\n=== Position Analysis ===")
    print(f"Total candidates: {len(move_values)}")
    print(f"{'Rank':<5} {'Move':<10} {'Win Rate':<12} {'Visits':<10} {'Probability':<12}")
    print("-" * 50)
    
    sorted_moves = sorted(
        move_values.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for rank, (move, win_rate) in enumerate(sorted_moves[:15], 1):
        visits = agent.root.children[move].visit_count
        prob = move_probs.get(move, 0)
        print(f"{rank:<5} {str(move):<10} {win_rate:<12.4f} {visits:<10} {prob:<12.4f}")
    
    # 显示最优落子
    if visualize and sorted_moves:
        best_move = sorted_moves[0][0]
        game_preview = copy.deepcopy(game_state)
        game_preview.place_stone(best_move[0], best_move[1])
        
        ui.draw(
            board_array=game_preview.get_board(),
            current_player=game_preview.current_player,
            game_over=game_preview.game_over,
            winner=game_preview.winner,
            l2_count=game_preview.l2_count,
            l3_count=game_preview.l3_count,
            l4_count=game_preview.l4_count,
            rl4_count=game_preview.rl4_count,
            last_move=best_move
        )
        print(f"\n显示评分最高的落子点 {best_move}...")
        
        for _ in range(2):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            clock.tick(1)
        
        pygame.quit()


if __name__ == "__main__":
    # 示例 1：完整的游戏（有可视化）
    print("\n" + "="*70)
    print("Example 1: Play a complete game with MCTS (with visualization)")
    print("="*70)
    while 1:
        winner = play_game_with_mcts(board_size=15, num_simulations=200, verbose=True, visualize=True)
    
