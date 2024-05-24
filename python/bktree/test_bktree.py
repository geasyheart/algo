from bktree import BurkhardKellerTree

tree = BurkhardKellerTree()
tree.add('中华人民共和国')
tree.add('中华人民共和国成立了')
tree.add('中国人民')
tree.add('中华人民')
print(tree.search_similar_word('中国人'))
