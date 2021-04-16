import sys
from math import sqrt


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

    def _read_file(self, file):
        """ Read start and end points from file

        :param file:
        :return:
        """
        with open(file, mode='r') as f:
            l1 = f.readline()
            l2 = [int(i.strip()) for i in f.readline().split(',')]
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
        if mode:
            sys.stdout = open(out_file, mode='w')
        prev = path[0]
        print("Number of moves: %d"% (len(path)-1))
        for curr in path:
            self.to_string(curr, prev)
            print(curr)
            prev = curr
        sys.stdout.close()

    def get(self, mode):
        """ Get method from mode name

        :param mode: Algorithm selection
        :return: Method containing requested algorithm
        """
        def not_found():
            print_err("Invalid method")
            exit(1)
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
        if self.check_banks(node) and self.check_banks([self.start[i]-node[i] for i in range(self.BANK_SIZE)]):
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

    def dfs(self, node, path):
        """ Depth First Search

        :param node: list of nodes
        :param path: Solution to puzzle, empty initially
        :return: (list): list of nodes
        """
        # End goal reached
        if self.end in path:
            return path
        if node == self.end:
            path.append(node)
            return path
        if tuple(node) not in self.visited:
            self.visited[tuple(node)] = True
            path.append(node)

        for n in self._successor(node):
            if tuple(n) not in self.visited:
                self.dfs(n, path)

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
            if not path or node[2] != path[-1][2]:
                path.append(node)
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

        while len(stk):
            top = stk[-1]
            stk.pop()

            if top == self.end:
                path.append(top)
                return
            if tuple(top) not in self.visited:
                self.visited[tuple(top)] = True
                path.append(top)
            for n in self._successor(top):
                if tuple(n) not in self.visited:
                    stk.append(n)

    @staticmethod
    def backtrace(curr_node, path):
        """ Backtrace up node pointers

        :param curr_node: node to trace
        :param path: empty path to return
        :return: path
        """

        while curr_node is not None:
            path.append(curr_node.state)
            curr_node = curr_node.parent
        return path[::-1]

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
        loc_visited = {}
        queue = [Node(s=node)]

        # Max number of iterations before stopping
        max_itr = (node[0] + node[1]) ** 5
        idx = 0

        while queue and idx < max_itr:
            curr_node = queue.pop(0)
            loc_visited[tuple(curr_node.state)] = True
            idx += 1

            if curr_node == self.end:
                return self.backtrace(curr_node, path)

            # Get all possible nodes from current
            for n in self._successor(curr_node.state):
                loc_node = Node(curr_node, n)

                if tuple(n) in loc_visited:
                    continue

                # Calculate f(n) = g(n) + h(n)
                loc_node.g = curr_node.g + 1             # Movement cost of 1
                loc_node.h = self.heuristic(n, loc_node.g)
                loc_node.f = loc_node.g + loc_node.h

                if loc_node.g > curr_node.g and loc_node in queue:
                    continue
                queue.append(loc_node)

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
    if argc != 4:
        print_err("Wrong number of arguments!!")
        print_err("Usage:")
        print_err("\tpython3 main.py <initial state file> <goal state file> <mode> <output file>")
        exit(1)
    else:
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


if __name__ == "__main__":
    main(len(sys.argv[1:]), sys.argv[1:])
