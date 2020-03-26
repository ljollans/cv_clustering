import numpy as np

from utils import co_clustering_match

assignment1=np.random.randint(0,3,10)
assignment2=np.random.randint(0,3,10)

contingency, best_match_1, best_match_2=co_clustering_match(assignment1,assignment2)

print(assignment1)
print(assignment2)
print(contingency)
print(best_match_1)
print(best_match_2)