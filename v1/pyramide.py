t_ = [[1],[2,3],[3,4,5],[8,7,6,9]]


def next_lev(t1,t2):
    n = []
    
    for j in t1:
        for i in t2:
            n.append(j+i)
    return n

def somme_tree(tree):
    n = tree[0]
    for t1 in tree[1:]:
        n = next_lev(n,t1)
    return n


print(somme_tree(t_))