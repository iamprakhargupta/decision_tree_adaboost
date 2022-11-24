"""
CSCI-630: Foundation of AI
helper.py
Autor: Prakhar Gupta
Username: pg9349

accuracy function implementation

"""
def accuracy(x1,x2):
    accu=0
    counter=0
    if len(x1)!=len(x2):
        print(f" Check length --> {len(x1)} of X1 and {len(x2)} of X2"  )
    else:
        for i in range(len(x1)):
            if x1[i]==x2[i]:
                counter+=1

        accu=counter/len(x1)
        return accu
