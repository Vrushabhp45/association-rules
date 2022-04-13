# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
# Import Dataset
movie=pd.read_csv('C:/Users/HP/PycharmProjects/Excelrdatascience/my_movies.csv')
movie.shape
movie.info()
movie2=movie.iloc[:,5:]
# with 30% support
itemsets=apriori(movie2,min_support=0.3,use_colnames=True)
# 50% confidence
rules=association_rules(itemsets,metric='lift',min_threshold=0.5)
# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]
# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# 2. Association rules with 50% Support and 50% confidence
# with 50% support
itemsets2=apriori(movie2,min_support=0.5,use_colnames=True)
# 90% confidence
rules2=association_rules(itemsets2,metric='lift',min_threshold=0.5)
# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()
