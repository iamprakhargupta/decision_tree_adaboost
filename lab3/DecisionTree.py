"""
CSCI-630: Foundation of AI
DecisionTree.py
Autor: Prakhar Gupta
Username: pg9349

Decision tree implementation

"""

from node import Node
import numpy as np
import pandas as pd





class DecisionTree():
    def __init__(self,root=None,level=2):
        self.root=root
        self.level=level

    # Referred to following resources -
    # https://www.youtube.com/watch?v=jVh5NA9ERDA


    def create_tree(self,df,labels,level):
        # data=df[:,:-1]
        # labels=df[:,-1]

        if level<=self.level:
            left,right,labell,labelr,params=self.split_column(df,labels)
            if left is not None and right is not None:
                left_tree=self.create_tree(left,labell,level+1)
                right_tree=self.create_tree(right,labelr,level+1)
                #work here
                return Node(findex=params["feature_number"], th=params["breakpoint"], left=left_tree,right=right_tree, gain=params["info_gain"], value=None,type="D")

        d={}
        for i in labels:
            if i in d:
                d[i]+=1
            else:
                d[i]=1
        max_v = max(d, key=d.get)
        return Node(value=max_v,type="L",counts=d)


    def split_column(self,data,labels):
        max_gain=-9999999999
        params={}
        data_left=None
        data_right=None
        labels_left, labels_right=None,None
        #labels=list(labels)
        n, f = data.shape
        for i in range(f):
            slice=data[:,i]
            nuniq=np.unique(slice)
            for j in nuniq:
                left=[]
                right=[]
                l=[]
                r=[]


                for row in range(len(data)):
                    if data[row][i]<=j:

                        left.append(data[row])
                        l.append(labels[row])
                for row in range(len(data)):
                    if data[row][i]>j:
                        right.append(data[row])
                        r.append(labels[row])
                left=np.array(left)
                right=np.array(right)
                l=np.array(l)
                r=np.array(r)

                if len(left)==0 or len(right)==0:
                    continue;

                gain=self.get_gain(l,r,labels)
                if gain>max_gain:
                    labels_left=l
                    labels_right=r
                    data_left=left
                    data_right=right
                    params["info_gain"]=gain
                    params["feature_number"]=i
                    params["breakpoint"]=j
                    max_gain=gain

        return data_left,data_right,labels_left,labels_right,params

    def get_gain(self,left,right,labels):
        """
        calculate imformation gain
        Gain(A)=B(p/p+n)−Remainder(A).
        """
        left=np.array(left)
        right = np.array(right)
        l=len(left)
        r=len(right)
        n=len(labels)
        left_ratio=l/n
        right_ratio = r / n
        return self.H_entropy(labels)-(left_ratio*self.H_entropy(left) + right_ratio*self.H_entropy(right))

    def train(self,data,label):
        root=self.create_tree(data,label,1)
        self.root=root
        return root

    def H_entropy(self,labels):
        """
        Entropy calulation
        H(V)= −∑k P(vk)log2P(vk).
        :param labels:
        :return:
        """

        n=np.unique(labels)
        l=len(labels)
        entropy=0
        labels=labels.astype(list)
        for i in n:
            c_entropy = 0
            counter=0
            for j in labels:
                if j==i:
                    counter+=1
            c_entropy=-(counter/l*(np.log2(counter/l)))
            entropy+=c_entropy

        return entropy

    def pred(self,data):
        test=[]
        for i in data:
            test.append(self._pred(i,self.root))

        return test


    def _pred(self,x,root):
        if root.value!=None:
            return root.value
        feature_val = x[root.findex]
        if feature_val<=root.th:
            return self._pred(x, root.left)
        else:
            return self._pred(x, root.right)

    def travese_tree(self, tree=None, indent="--",level=1):


        if tree is None:
            tree = self.root

        if tree.value is not None:
            print(" Predicted= "+tree.value+ "|| Actual number of labels in this leaf= "+  str(tree.counts))

        else:
            print("Level ",level," Split on Column No " + str(tree.findex+1), " || left split <=", tree.th, " || right split > ", tree.th,"|| Information GAIN= ",
                  tree.gain)

            print("%s> Left node:" % (indent), end="")
            self.travese_tree(tree.left, indent + indent,level+1)

            print("%s> Right node:" % (indent), end="")
            self.travese_tree(tree.right, indent + indent,level+1)

