import sys
import itertools as itr


class Puzzle:
    start = ()
    end = {}
    nodes = {}         # Tree of all possible nodes
    visited = {}       # Hash table of visited nodes
    raft = {}

    def __init__(self, file_name):
        # 'r': is raft at start
        self.raft = {"w": 0, "c": 0}
        self.chicken = 0
        self.wolf = 0
        self.__read_file(file_name)
        self.nodes[self.start] = {}
        self.visited[self.start] = False
        self.visited[(0, 0)] = False        # Starting point empty init to false

    def __read_file(self, file):
        """ Read start and end points from file

        :param file:
        :return:
        """
        with open(file, mode='r') as f:
            l1 = f.readline()
            l2 = [int(i.strip()) for i in f.readline().split(',')]
            self.start = (l2[0], l2[1])
            self.chicken = l2[0]
            self.wolf = l2[1]

    def check_banks(self):
        """ Check if the number of wolf > chicken on both banks.

        :return:
            Bool
        """
        if self.start[0] >= self.start[1] and self.end[0] >= self.end[1]:
            if self.start[0] >= 0 and self.end[1] >= 0:
                return True
        else:
            return False

    def check_boat(self):
        """ Check if the number of wolf > chicken on boat.

        :return:
            Bool
        """
        if self.start[0] > 0 or self.end[1] > 0 and self.check_banks():
            return True
        else:
            return False

    def is_goal(self):
        """ Check if current state is goal.

        :return:
            Bool
        """
        if self.start[0] == 0 and self.start[1] == 0:
            return True
        else:
            return False

    def gen_all_nodes(self, chicken, wolf, tree):
        while True:
            if chicken == 0 and wolf == 0:
                break
            nodes = self.__next_nodes(chicken, wolf)
            # Use current chicken, wolf as key and all possible nodes as value
            #tree[(chicken, wolf)] = nodes
            keys = tree.keys()
            for key in keys:
                # if key in self.visited:
                #     if self.visited[key]:
                #         break
                #     else:
                #         self.visited[key] = True
                # else:
                self.visited[key] = False
                chicken = key[0]
                wolf = key[1]
                nodes = self.__next_nodes(chicken, wolf)
                tree[key] = nodes
                self.gen_all_nodes(chicken, wolf, tree[key])


        return tree

    def __next_nodes(self, chn, wolf):
        """ Generate all possible states from current node
        :param:
            chn (int): number of chickens
            wolf (int): number of wolf
        :return:
            nodes (dict): map of all possible nodes
        """
        nodes = {}
        if chn > 0 and chn - 1 >= wolf:
            #if not self.visited[(chn - 1, wolf)]:
            # Take one chicken to end
            nodes[(chn - 1, wolf)] = {}
        if chn > 1 and chn - 2 >= wolf:
            #if not self.visited[(chn - 2, wolf)]:
            # Take 2 chicken to end
            nodes[(chn - 2, wolf)] = {}
        if wolf > 0:
            #if not self.visited[(chn, wolf - 1)]:
            # Take 1 wolf to end
            nodes[(chn, wolf - 1)] = {}
        if wolf > 1:
            #if not self.visited[(chn, wolf - 2)]:
            # Take 2 wolf to end
            nodes[(chn, wolf - 2)] = {}
        if chn > 0 and chn - 1 >= wolf - 1 and wolf > 0:
            #if not self.visited[(chn - 1, wolf - 1)]:
            # Take one chicken and one wolf to end
            nodes[(chn - 1, wolf - 1)] = {}
        return nodes


def pretty_print(_dict, i=0):
    for key, val in _dict.items():
        print('\t' * i + str(key))
        if isinstance(val, dict):
            pretty_print(val, i+1)
        else:
            print('\t' * (i + 1) + str(val))

def main():
    import json
    p1 = Puzzle("start1.txt")
    #print(p1.nodes)
    n = p1.gen_all_nodes(p1.chicken, p1.wolf, p1.nodes)
    #pretty_print(n)
    import pprint
    p = pprint.PrettyPrinter()
    p.pprint(n)
    print("-------------------")
    print(n.keys())

if __name__ == "__main__":
    main()
