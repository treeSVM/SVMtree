import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class BinaryTree():
    def __init__(self,rootid):
      self.left = None
      self.right = None
      self.root = rootid

    def getLeftChild(self):
        return self.left


    def getRightChild(self):
        return self.right


    def setNodeValue(self,value):
        self.root = value


    def getNodeValue(self):
        return self.root


    def insert(self, newNodes):
        self.insertLeft(newNodes[0])
        self.insertRight(newNodes[1])


    def insertLeft(self,newNode):
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.left = self.left
            self.left = tree


    def insertRight(self,newNode):
        if self.right == None:
            self.right = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.right = self.right
            self.right = tree


    def insertChild(self, newNode):
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            self.right = BinaryTree(newNode)


    def getChild(self):
        if self.left == None:
            return self.left
        else:
            return self.right



def printTree(tree):
    G = nx.Graph()
    preOrden(tree, G, 1)
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()


def preOrden(tree, G, i):
    if i==1:
        G.add_node(i)
        print(i)
        print(tree.getNodeValue())
        print('\n')
    if(tree.getLeftChild()):
        m = sorted(list(G.nodes), reverse=True)[0]
        print(m+1)
        print(tree.getLeftChild().getNodeValue())
        print('\n')
        G.add_node(m+1)
        G.add_edge(i, m+1)
        preOrden(tree.getLeftChild(), G, i + 1)
    if(tree.getRightChild()):
        m = sorted(list(G.nodes), reverse=True)[0]
        print(m+1)
        print(tree.getRightChild().getNodeValue())
        print('\n')
        G.add_node(m + 1)
        G.add_edge(i, m + 1)
        preOrden(tree.getRightChild(), G, i + 1)
    return