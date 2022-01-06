import mcts
import chess

root = mcts.Node(chess.Board())
root.win_count = 15
root.visit_count = 17
root.is_expanded = True

node1 = mcts.Node(chess.Board())
node1.parent = root
node1.win_count = 1
node1.visit_count = 2
node1.is_expanded = True
root.children.append(node1)

node2 = mcts.Node(chess.Board())
node2.parent = root
node2.win_count = 5
node2.is_expanded = True
node2.visit_count = 5
root.children.append(node2)

node3 = mcts.Node(chess.Board())
node3.parent = root
node3.win_count = 9
node3.is_expanded = True
node3.visit_count = 10
root.children.append(node3)

node4 = mcts.Node(chess.Board())
node4.win_count = 40
root.children.append(node4)

node5 = mcts.Node(chess.Board())
node5.win_count = 50
root.children.append(node5)

algo = mcts.Vanilla_MCTS()

print(algo.traverse(root).win_count)
