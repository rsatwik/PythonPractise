import sys
import math
from collections import defaultdict
import random

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

# n: the total number of nodes in the level, including the gateways
# l: the number of links
# e: the number of exit gateways
g = Graph()
exits = []
n, l, e = [int(i) for i in input().split()]

for i in range(l):
    # n1: N1 and N2 defines a link between these nodes
    n1, n2 = [int(j) for j in input().split()]
    g.addEdge(n1,n2)
    g.addEdge(n2,n1)

for i in range(e):
    ei = int(input())  # the index of a gateway node
    exits.append(ei)

# game loop
while True:
    si = int(input())  # The index of the node on which the Bobnet agent is positioned this turn
    next_links = g.graph[si]
    found = False
    for link in next_links:
        if link in exits: # If the node is connected to exit, cut it
            print(str(si) + " " + str(link))
            found = True
    if not found:# Else cut some random connected node
            temp = random.choice(next_links)
            print(str(si) + " " + str(temp))
        
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
    # Example: 0 1 are the indices of the nodes you wish to sever the link between
    # print("0 1")

