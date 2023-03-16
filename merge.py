import pandas as pd

f1 = pd.read_csv('/home/kls/data/tabletennisdata/migu_new_data21.csv')
f2 = pd.read_csv('/home/kls/data/tabletennisdata/migu_new_data0.csv')
file = [f1, f2]
train = pd.concat(file)
train.to_csv("/home/kls/data/tabletennisdata/migu_new_datav3.csv", index=0, sep=',')
print('merge successful')
