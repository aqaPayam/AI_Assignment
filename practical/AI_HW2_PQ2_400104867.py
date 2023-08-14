"""


MY CODE
Payam Taebi
400104867

piece faghat bayad ya 1 bashe ya 2 va code file Game meghdar dige ii support nemikone !

"""


class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return [random.choice(BoardUtility.get_valid_locations(board)), random.choice([1, 2, 3, 4]),
                random.choice(["skip", "clockwise", "anticlockwise"])]


class HumanPlayer(Player):
    def play(self, board):
        move = input("row, col, region, rotation\n")
        move = move.split()
        print(move)
        return [[int(move[0]), int(move[1])], int(move[2]), move[3]]


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=3):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board):
        tree = Tree(piece=self.piece, max_depth=self.depth, board=board, random_prob=0)
        answer = tree.get_best_action()
        row = answer['row']
        col = answer['col']
        region = answer['region']
        rotation = answer['rotation']
        return [[row, col], region, rotation]


class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=3, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board):
        tree = Tree(piece=self.piece, max_depth=self.depth, board=board, random_prob=self.prob_stochastic)
        answer = tree.get_prob_best_action()
        row = answer['row']
        col = answer['col']
        region = answer['region']
        rotation = answer['rotation']
        return [[row, col], region, rotation]


"""


MY CODE
Payam Taebi
400104867

"""
import math
import random
import numpy as np

"""
tarjihan ba omgh = 3 bazi run beshe ham sari ham khoob
"""
from Board import BoardUtility

main_board = None
main_max_depth = 0

""""
ev1 miad mishmore be har tool chand ta darim va namei jam mizane va roo kaghez ev behtarie ehtemalan
"""


def evaluate(piece):
    ROWS = 6
    COLS = 6
    result = 0
    board = main_board
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == piece:
                horizontal = 0
                vertical = 0
                diagonal = 0
                while r + vertical < ROWS and board[r + vertical][c] == piece:
                    vertical += 1
                while c + horizontal < COLS and board[r][c + horizontal] == piece:
                    horizontal += 1
                while r + diagonal < ROWS and c + diagonal < COLS and board[r + diagonal][c + diagonal] == piece:
                    diagonal += 1
                change = 2 ** diagonal + 2 ** vertical + 2 ** horizontal
                #                if (r == 2 and c == 2) or (r == 2 and c == 3) or (r == 3 and c == 2) or (r == 3 and c == 3):
                #                    change *= 8
                result += change

            if board[r][c] != piece and board[r][c] != 0:
                horizontal = 0
                vertical = 0
                diagonal = 0
                while r + vertical < ROWS and board[r + vertical][c] != piece and board[r + vertical][c] != 0:
                    vertical += 1
                while c + horizontal < COLS and board[r][c + horizontal] != piece and board[r][c + horizontal] != 0:
                    horizontal += 1
                while r + diagonal < ROWS and c + diagonal < COLS and board[r + diagonal][c + diagonal] != piece and \
                        board[r + diagonal][c + diagonal] != 0:
                    diagonal += 1
                change = 2 ** diagonal + 2 ** vertical + 2 ** horizontal
                result -= change

    return result


"""
ev 2 ke estefade kardam miad max tool momken ro mishmore faghat albate be 4 khoone vasati ham ahamiat vixhe i mide
"""


def evaluate2(piece):
    ROWS = 6
    COLS = 6
    result = 0
    board = main_board
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == piece:
                horizontal = 0
                vertical = 0
                diagonal = 0
                while r + vertical < ROWS and board[r + vertical][c] == piece:
                    vertical += 1
                while c + horizontal < COLS and board[r][c + horizontal] == piece:
                    horizontal += 1
                while r + diagonal < ROWS and c + diagonal < COLS and board[r + diagonal][c + diagonal] == piece:
                    diagonal += 1
                result = max(diagonal, horizontal, vertical, result)
    if result >= 5:
        return math.inf
    if board[2][2] == piece:
        result += 1
    if board[2][3] == piece:
        result += 1
    if board[3][2] == piece:
        result += 1
    if board[3][3] == piece:
        result += 1
    max_result = result

    result = 0
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != piece and board[r][c] != 0:
                horizontal = 0
                vertical = 0
                diagonal = 0
                while r + vertical < ROWS and board[r + vertical][c] != piece and board[r + vertical][c] != 0:
                    vertical += 1
                while c + horizontal < COLS and board[r][c + horizontal] != piece and board[r][c + horizontal] != 0:
                    horizontal += 1
                while r + diagonal < ROWS and c + diagonal < COLS and board[r + diagonal][c + diagonal] != piece and \
                        board[r + diagonal][c + diagonal] != 0:
                    diagonal += 1
                result = max(diagonal, horizontal, vertical, result)
    if result >= 5:
        return - 10 ** result

    return 10 ** max_result


def generate_valid_actions(piece):
    result = []
    for target in BoardUtility.get_valid_locations(main_board):
        action = {'row': target[0], 'col': target[1], 'region': 1, 'rotation': "skip",
                  'piece': piece}
        result.append(action)

    for target in BoardUtility.get_valid_locations(main_board):
        for region in range(1, 5):
            for rotation in ["clockwise", "anticlockwise"]:
                action = {'row': target[0], 'col': target[1], 'region': region, 'rotation': rotation,
                          'piece': piece}
                result.append(action)

    return result


"""
az hamsaye haye momken ye tedadi ro ba tavajoh be omgh shuffle bar midare ke code sari tar beshe
"""


def select_best_actions(valid_actions, piece, prob=False):
    number_of_sample = 200 // main_max_depth
    if len(valid_actions) > number_of_sample:
        valid_actions = random.sample(valid_actions, number_of_sample)
    if prob:
        return sort_actions(actions=valid_actions, piece=piece)
    else:
        return valid_actions


def sort_actions(actions, piece):
    values = []
    for act in actions:
        do_act(act)
        values.append(evaluate2(piece))
        undo_act(act)
    values = np.array(values)
    sort = np.argsort(values)
    np_actions = np.array(actions)
    np_actions = np_actions[sort]
    actions = list(np_actions)
    return actions


class Node:
    def __init__(self, action):
        self.children = []
        self.height = 0
        self.action = action
        self.value = 0  # optima_value
        self.expected_value = 0

    def add_child(self, child):
        child.height = self.height + 1
        self.children.append(child)

    def is_max(self):
        if self.height % 2 == 0:
            return True
        return False

    def is_min(self):
        return not self.is_max()

    def is_leaf(self):
        if self.height >= main_max_depth or BoardUtility.is_terminal_state(main_board):
            return True
        return False

    def add_children(self, piece, prob=False):
        valid_actions = generate_valid_actions(piece)
        valid_actions = select_best_actions(valid_actions, piece, prob)
        if prob:
            if self.is_max():
                for act in reversed(valid_actions):
                    self.add_child(Node(act))
            if self.is_min():
                for act in valid_actions:
                    self.add_child(Node(act))
        else:
            for act in valid_actions:
                self.add_child(Node(act))

    def get_best_action(self):
        for child in self.children:
            if child.value == self.value:
                return child.action

    def get_best_action_prob(self, prob):
        coin = np.random.binomial(1, prob, 1)
        if coin[0] == 1 and self.expected_value != math.inf:
            dif = math.inf
            result = None
            for child in self.children:
                if abs(child.value - self.expected_value) <= dif:
                    dif = abs(child.value - self.expected_value)
                    result = child
            return result.action
        else:
            for child in self.children:
                if child.value == self.value:
                    return child.action


def do_act(act):
    if act is None:
        return
    global main_board
    board = main_board
    BoardUtility.make_move(board, act['row'], act['col'], act['region'], act['rotation'], act['piece'])


def undo_act(act):
    if act is None:
        return
    global main_board
    board = main_board
    if act['rotation'] == "clockwise":
        BoardUtility.rotate_region(board, act['region'], "anticlockwise")
    elif act['rotation'] == "anticlockwise":
        BoardUtility.rotate_region(board, act['region'], "clockwise")
    board[act['row']][act['col']] = 0


class Tree:
    def __init__(self, piece, max_depth, board, random_prob):
        self.random_prob = random_prob
        self.piece = piece
        self.max_depth = max_depth
        self.board = board
        self.node_opened = 0

    def get_best_action(self):
        global main_board
        global main_max_depth
        main_board = self.board
        main_max_depth = self.max_depth

        root = self.create_tree()
        print("node opened:", self.node_opened)
        return root.get_best_action()

    def get_prob_best_action(self):
        global main_board
        global main_max_depth
        main_board = self.board
        main_max_depth = self.max_depth

        root = self.create_tree_prob()
        print("node opened:", self.node_opened)
        return root.get_best_action_prob(self.random_prob)

    def create_tree_prob(self):
        root = Node(action=None)
        self.minimax_prob(root, 0, -math.inf, math.inf)
        return root

    def create_tree(self):
        root = Node(action=None)
        self.minimax(root, 0, -math.inf, math.inf)
        return root

    def minimax(self, node, depth, alpha, beta):
        do_act(act=node.action)
        self.node_opened += 1

        if node.is_leaf():
            value = evaluate2(piece=self.piece)
            undo_act(node.action)
            node.value = value
            return value

        if node.is_max():
            bestVal = - math.inf
            node.add_children(piece=self.piece, prob=False)
            for child in node.children:
                value = self.minimax(child, depth + 1, alpha, beta)
                if value > bestVal:
                    bestVal = value
                alpha = max(alpha, bestVal)
                if beta <= alpha:
                    break

            undo_act(node.action)
            node.value = bestVal
            return bestVal

        else:
            bestVal = +math.inf
            node.add_children(piece=3 - self.piece, prob=False)
            for child in node.children:
                value = self.minimax(child, depth + 1, alpha, beta)
                if value < bestVal:
                    bestVal = value
                beta = min(beta, bestVal)
                if beta <= alpha:
                    break

            undo_act(node.action)
            node.value = bestVal
            return bestVal

    def minimax_prob(self, node, depth, alpha, beta):
        do_act(act=node.action)
        self.node_opened += 1

        if node.is_leaf():
            value = evaluate2(piece=self.piece)
            undo_act(node.action)
            node.value = value
            return value

        if node.is_max():
            visited_value = []
            bestVal = - math.inf
            node.add_children(piece=self.piece, prob=True)
            for child in node.children:
                value = self.minimax(child, depth + 1, alpha, beta)
                visited_value.append(value)
                if value > bestVal:
                    bestVal = value
                alpha = max(alpha, bestVal)
                if beta <= alpha:
                    break

            undo_act(node.action)

            node.value = bestVal
            node.expected_value = np.mean(visited_value)
            return self.random_prob * node.expected_value + (1 - self.random_prob) * bestVal

        else:
            bestVal = +math.inf
            node.add_children(piece=3 - self.piece, prob=True)
            for child in node.children:
                value = self.minimax(child, depth + 1, alpha, beta)
                if value < bestVal:
                    bestVal = value
                beta = min(beta, bestVal)
                if beta <= alpha:
                    break

            undo_act(node.action)
            node.value = bestVal
            return bestVal
