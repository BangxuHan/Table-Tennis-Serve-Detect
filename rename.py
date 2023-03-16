import os
import sys

root = '/home/kls/data/tabletennisdata/WTT/一镜'

filelist = os.listdir(root)
# filelist.sort(key=lambda x: int(x.split('.')[-2]))
filelist.sort()
currentpath = os.getcwd()
os.chdir(root)
for i, filename in enumerate(filelist):
    print(filename)
#     os.rename(filename, 'wtt' + str(i+1).zfill(4) + '.mp4')
# os.chdir(currentpath)
# sys.stdin.flush()
# print('update successful')
