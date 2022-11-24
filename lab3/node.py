"""
CSCI-630: Foundation of AI
node.py
Autor: Prakhar Gupta
Username: pg9349

implement node

"""
class Node():
    def __init__(self, findex=None, th=None, left=None,
                 right=None, gain=None, value=None,type="D",counts={}):
        ''' constructor '''

        self.type=type
        self.left = left
        self.right = right
        self.th = th
        self.gain = gain
        self.findex = findex
        self.value = value
        self.counts=counts


