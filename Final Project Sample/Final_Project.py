# import required libraries

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


#load the data
basket = pd.read_csv('BreadBasket_DMS.csv')

# need to clean the data
## when i import the csv file, i need to import only november 2016 through end of march 2017
basket = basket.drop(basket[basket.Date < '2016-11-01'].index)
basket = basket.drop(basket[basket.Date >= '2017-04-01'].index)

# 19717 rows as final result of 'basket' dataframe

# what kind of items were sold at the baker?
print('list of items sold at the bakery')
for i in basket.Item.unique():
    print(i)
item_bakery_num = len(basket.Item.unique())
print(f'There are {item_bakery_num} kinds of items at the bakery')

# based on the result. we have 'NONE' as item. remove this datapoints from the 'basket' dataset

basket = basket[basket.Item != 'NONE']

#18981 rows after cleaning

# I need to know the frequency of each item to before i choose item for 'interesting' rule
basket['Item'].value_counts().head(20)
## add a visaulization for this data
### Pie Chart
plt.figure(1, figsize= (4,4))
basket['Item'].value_counts().head(20).plot.pie(autopct ='%1.2f%%')
plt.show()

### Bar Chart for the easier comparison
itemNames = basket['Item'].value_counts().index
itemCounts = basket['Item'].value_counts().values
plt.figure(figsize=(10,4))
plt.ylabel('Values', fontsize = 5)
plt.xlabel('Items', fontsize = 5)
plt.title('Top 20 items sold between Nov16 to March17')
plt.bar(itemNames[:20], itemCounts[:20], width=0.8, color='red', linewidth=.2)
plt.xticks(fontsize=10, rotation=90)
plt.show()

# Date & Time
## change the date time column into one.

basket['datetime'] = pd.to_datetime(basket['Date']+" "+basket['Time'])
basket['Week'] = basket['datetime'].dt.week
basket['Month'] = basket['datetime'].dt.month
basket['Weekday'] = basket['datetime'].dt.weekday
basket['Hours'] = basket['datetime'].dt.hour

## need to remove outlier
basket = basket[basket.Hours != 1] # there are only one transaction
basket = basket[basket.Hours != 23]  # there are only two transaction on a specific day and seems to be an error


## Monthly transactions
plt.figure(figsize=(5,5))
plt.bar(basket['Month'].value_counts().index, basket['Month'].value_counts().values, width=0.8, align='center',linewidth=.2)
plt.title('Number of Transacation per month between Nov to March')
plt.show()
Month = {11:'November',12:'December', 1:'January',2:'February',3:'March'}
Monthmax = Month[basket['Month'].value_counts().index[0]]
Monthmaxtran = basket['Month'].value_counts().values[0]
Monthmin = Month[basket['Month'].value_counts().index[4]]
Monthmintran = basket['Month'].value_counts().values[4]

print(f'On {Monthmax}, there are the most amount of transaction, which is {Monthmaxtran}, at the bakery. \nOn {Monthmin}, there are the least amount transaction, which is {Monthmintran}, at the bakery.')



## Weekly transactions
plt.figure(figsize=(3,3))
plt.bar(basket['Weekday'].value_counts().index, basket['Weekday'].value_counts().values, width=0.7, align='center',linewidth=1)
plt.title('Number of Transaction (Weekday)')
plt.show()
weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekdaymax = weekday[basket['Weekday'].value_counts().index[0]]
weekdaymaxtran = basket['Weekday'].value_counts().values[0]
weekdaymin = weekday[basket['Weekday'].value_counts().index[6]]
weekdaymintran = basket['Weekday'].value_counts().values[6]

print(f'On {weekdaymax}, there are the most amount of transaction, which is {weekdaymaxtran}, at the bakery. \nOn {weekdaymin}, there are the least amount transaction, which is {weekdaymintran}, at the bakery.')

## Hourly transactions analysis
plt.figure(figsize=(6,3))
plt.bar(basket['Hours'].value_counts().index, basket['Hours'].value_counts().values, width=.5, align='center',linewidth=1)
plt.title('Numer of Transactions (Hourly)')
plt.show()
Hoursmax = basket['Hours'].value_counts().index[0]
Hoursmaxtran = basket['Hours'].value_counts().values[0]
Hoursmin = basket['Hours'].value_counts().index[14]
Hoursmintran = basket['Hours'].value_counts().values[14]

print(f'On {Hoursmax}\'o clock, there are the most amount of transaction, which is {Hoursmaxtran}, at the bakery. \nOn {Hoursmin}\'o clock, there are the least amount transaction, which is {Hoursmintran}, at the bakery.')


#market basket analysis

market_basket = basket.groupby(['Transaction', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
market_basket1 = market_basket.applymap(encode_units)

## support level: how frequently the itemset appears in the dataset.
## confidence level: how often the rule has been found as true
frequent_itemsets = apriori(market_basket1, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# any rule has lower than 60% of confidence rule isn't really reliable.
# however, this dataset isn't big enough to have confidence level to be this high.
# with confidence level of .6, it only result one rule. so I decided to use .5
# we also want lift level to be higher than .8

interesting_rules = rules[(rules['lift'] >=.8)&(rules['confidence']>=.5)]
interesting_rules = interesting_rules.sort_values(by='lift', ascending=False)
print(interesting_rules)

