#!/usr/bin/env python3

""" River Crossing Puzzle with BFS, DFS, iterative DFS, and A* algorithm implementation.
    By: Kyle Huang
"""

import sys
from math import sqrt
import heapq as hq


class Node:
    state = []

    def __init__(self, p=None, s=None):
        self.parent = p         # Pointer to previous node
        self.state = s          # Current state

        # A* variables
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.state == other


class PriorityQueue:
    """ Implementation of priority queue using heapq.
        Used for A* algorithm
    """
    def __init__(self):
        self._data = []
        self._i = 0

    def pop(self):
        """ Pop from queue
        :return:
        """
        return hq.heappop(self._data)[-1]

    def push(self, node, p):
        """ Push onto queue
        :param node: node to insert into queue
        :param p: priority
        :return:
        """
        hq.heappush(self._data, (p, self._i, node))
        self._i += 1

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if len(self._data) > self._idx:
            res = self._data[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration


class Puzzle:
    # Constants
    BANK_SIZE = 3
    MOVE = [
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [0, 2, 1],
        [2, 0, 1]
    ]
    end = [0, 0, 0]

    def __init__(self, in_file):
        self._read_file(in_file)
        self.visited = {}           # Visited nodes
        self.count = 0              # Number of nodes expanded

    def _read_file(self, file):
        """ Read start and end points from file

        :param file:
        :return:
        """
        with open(file, mode='r') as f:
            try:
                l1 = f.readline()
                l2 = [int(float(i.strip())) for i in f.readline().split(',')]

            except ValueError as e:
                print_err("Error: ", e)
                exit(1)
            self.start = [l2[0], l2[1], l2[2]]
            self.c = l2[0]
            self.w = l2[1]

    def to_file(self, out_file, path, mode=0):
        """ Output to file

        :param out_file: file name
        :param path: solution
        :param mode: write to stdout or file
                    0: stdout, 1: file
        :return: file output
        """
        magenta = "\u001b[35;1m"
        yellow = "\u001b[33;1m"
        red = "\u001b[31;1m"
        green = "\u001b[32;1m"
        reset = "\u001b[0m"

        if mode:
            sys.stdout = open(out_file, mode='w')
            magenta = yellow = red = green = reset = ''

        prev = path[0]
        print("Number of moves: %d"% (len(path)-1))
        print("[%sChicken%s, %sWolf%s, Start:%s1%s/End:%s0%s]"% (green,reset,red,reset,magenta,reset,yellow,reset))

        for curr in path:
            self.to_string(curr, prev)
            c = magenta + str(curr[2]) if curr[2] == 1 else yellow + str(curr[2])
            print("[",green,curr[0],reset,", ",red,curr[1],reset,", ", c,reset,"]")
            prev = curr
        sys.stdout.close()

    def get(self, mode):
        """ Get method from mode name

        :param mode: Algorithm selection
        :return: Method containing requested algorithm
        """
        def not_found(_, __):
            print_err("Invalid method")
            exit(1)
        mode = 'itr_dfs' if mode == 'iddfs' else mode
        func = getattr(self, mode, not_found)
        return func

    @staticmethod
    def to_string(*args):
        """ Print output

        :param args: Current node and next node
        :return: Output string
        """
        next_state = args[1]
        curr = args[0]
        if args[0][2] == 0:
            next_state = args[0]
            curr = args[1]
        if next_state[0] == curr[0] - 1 and curr[1] == next_state[1]:
            # One chicken in the boat
            print("Put one chicken in the boat")
        elif next_state[0] == curr[0] - 2 and curr[1] == next_state[1]:
            # Two chicken in the boat
            print("Put two chickens in the boat")
        elif next_state[1] == curr[1] - 1 and next_state[0] == curr[0]:
            # one wolf in the boat
            print("Put one wolf in the boat")
        elif next_state[1] == curr[1] - 2 and next_state[0] == curr[0]:
            # two wolves in boat
            print("Put two wolves in the boat")
        elif next_state[1] == curr[1] - 1 and next_state[0] == curr[0] - 1:
            # one chicken and one wolf
            print("Put one wolf and one chicken in the boat")

    def validate_move(self, node):
        """ Check if a given node is a legal move
            Check that given move is between 0 <= move <= max animal(c, w).
        :param node: bank to check
        :return: bool: True if its a valid move
        """
        return True if 0 <= node[0] <= self.c and 0 <= node[1] <= self.w else False
    
    @staticmethod
    def check_banks(node):
        """ Check both to see if it is valid
            Check that bank has more chicken than wolf.
        :param node: state of bank to check
        :return: bool: true if bank is valid
        """
        return False if node[0] != 0 and node[0] < node[1] else True

    def __next_node(self, node, idx):
        """ Get list of next nodes

        :param node: Current node
        :param idx: Index in MOVE for MOVE
        :return: List of all next possible nodes
        """
        # Boat is at end
        if node[2] == 0:
            return [node[i] + self.MOVE[idx][i] for i in range(self.BANK_SIZE)]
        else:
            return [node[i] - self.MOVE[idx][i] for i in range(self.BANK_SIZE)]
            
    def check_node(self, node):
        """ Checks given node if it is valid

        :param node: Node to check
        :return: Bool, True if node is valid
        """
        if self.check_banks(node) and self.check_banks([self.start[i] - node[i] for i in range(self.BANK_SIZE)]):
            if self.validate_move(node):
                return True
        return False
    
    def _successor(self, node):
        """ Next possible nodes

        :param node: Node to check
        :return: List of all possible nodes from given node
        """
        stack = []

        for idx, move in enumerate(self.MOVE):
            neighbor = self.__next_node(node, idx)

            if self.check_node(neighbor):
                stack.append(neighbor)
        return stack

    def dfs(self, node, path, count_r=0):
        """ Depth First Search

        :param node: list of nodes
        :param path: Solution to puzzle, empty initially
        :param count_r: Recursion counter
        :return: (list): list of nodes
        """
        # Base case
        if node == self.end:
            path.append(node)
            return
        # End goal reached
        if self.end in path:
            return
        if tuple(node) not in self.visited and self.check_node(node):
            self.visited[tuple(node)] = True
            path.append(node)

        self.count += 1
        for n in self._successor(node):
            if tuple(n) not in self.visited:
                count_r += 1
                self.dfs(n, path, count_r)
        return count_r

    def bfs(self, node, path):
        """ Breadth First Search

        :param node: Starting node
        :param path: path from start to end
        :return: path
        """
        queue = [node]
        self.visited[tuple(node)] = True

        while queue:
            node = queue.pop(0)
            if node == self.end:
                path.append(node)
                return
            # Check if not in path or current bank location is not previous
            if not path or (node[2] != path[-1][2] and self.check_node(node)):
                path.append(node)
            self.count += 1
            for n in self._successor(node):
                if tuple(n) not in self.visited:
                    self.visited[tuple(n)] = True
                    queue.append(n)

    def itr_dfs(self, node, path):
        """ Iterative Depth First Search

        :param node: Starting node
        :param path: path from start to end
        :return: path
        """
        stk = [node]

        while stk:
            top = stk[-1]
            stk.pop()

            if top == self.end:
                path.append(top)
                return
            if tuple(top) not in self.visited and self.check_node(top):
                self.visited[tuple(top)] = True
                path.append(top)

            for n in self._successor(top):
                self.count += 1
                if tuple(n) not in self.visited:
                    stk.append(n)

    @staticmethod
    def backtrace(curr_node, path):
        """ Backtrace up node pointers

        :param curr_node: node to trace
        :param path: empty path to return
        :return: path
        """
        path_ = []
        while curr_node is not None:
            path_.append(curr_node.state)
            curr_node = curr_node.parent
        for i in path_[::-1]:
            path.append(i)

    def heuristic(self, node, cost):
        """ Calculate Euclidean distance

        :param node: Node for distance calculation
        :param cost: Count from starting node
        :return:
        """
        dx = self.start[0] - node[0]
        dy = self.start[1] - node[1]
        return cost * sqrt(dx * dx + dy * dy)

    def astar(self, node, path):
        """ A* algorithm.
            Uses dfs to get different path costs.
        :param node: Starting node
        :param path: Path to return
        :return: Path
        """
        # Initialize priority queue
        queue = PriorityQueue()
        queue.push(Node(s=node), 0)

        while queue:
            curr_node = queue.pop()
            self.visited[tuple(curr_node.state)] = True

            # Stop when goal state is popped from queue
            if curr_node == self.end:
                return self.backtrace(curr_node, path)

            self.count += 1
            # Get all possible nodes from current
            for n in self._successor(curr_node.state):
                if not self.check_node(n):
                    continue
                loc_node = Node(curr_node, n)

                if tuple(n) in self.visited:
                    continue

                # Calculate f(n) = g(n) + h(n)
                loc_node.g = curr_node.g + 1             # Movement cost of 1
                loc_node.h = self.heuristic(n, loc_node.g)
                loc_node.f = loc_node.g + loc_node.h

                # If child.g > parent.g, then it's a worse path
                # Cutoff for "cost-limited" search
                if loc_node.g > curr_node.g and loc_node in queue:
                    continue
                queue.push(loc_node, loc_node.h)


def print_err(*args, **kwargs):
    """ Print to stderr
    :param args: arguments
    :param kwargs: separator
    """
    print(*args, file=sys.stderr, **kwargs)


def main(argc, argv):
    """
    :param argc: arg count
    :param argv: <initial state file> <goal state file> <mode> <output file>
    """
    if argc > 5 or argc < 4:
        print_err("Wrong number of arguments!!")
        print_err("Usage:")
        print_err("\tpython3 main.py <initial state file> <goal state file> <mode> <output file> <Debug:[y]>")
        exit(1)
    else:
        debug = True if argc == 5 else False
        if not debug:
            modes = ['bfs',
                     'dfs',
                     'iddfs',
                     'astar']

            # Mode selection
            if argv[2] in modes:
                q = []
                p1 = Puzzle(argv[0])
                # Get correct method
                mode = p1.get(argv[2])
                mode(p1.start, q)
                p1.to_file(argv[3], q)
                p1.to_file(argv[3], q, 1)
            else:
                exit(1)
        else:
            # Debug mode
            q = []
            qq = []
            qqq = []
            qqqq = []
            file = argv[0]
            p1 = Puzzle(file)
            p2 = Puzzle(file)
            p3 = Puzzle(file)
            p4 = Puzzle(file)

            p1.bfs(p1.start, q)

            count = p2.dfs(p2.start, qq)
            p3.itr_dfs(p3.start, qqq)
            p4.astar(p4.start, qqqq)

            print("+--------------------------------+")
            print("BFS moves: %d"% len(q))
            print("BFS nodes expanded: %d"% p1.count)
            print("+--------------------------------+")
            print("DFS moves: %d" % len(qq))
            print("DFS nodes expanded: %d" % p2.count)
            print("DFS recursion depth: %d" % count)
            print("+--------------------------------+")
            print("ITR-DFS moves: %d" % len(qqq))
            print("ITR-DFS nodes expanded: %d" % p3.count)
            print("+--------------------------------+")
            print("A-STAR moves: %d" % len(qqqq))
            print("A-STAR nodes expanded: %d" % p4.count)
            print("+--------------------------------+")


if __name__ == "__main__":
    main(len(sys.argv[1:]), sys.argv[1:])
