import cPickle
from random import randint, choice
from string import lowercase
from sys import maxint
from time import ctime

data=[]

doms = ('com', 'edu', 'net', 'org', 'gov')
# myfile=open('data.txt','w+')
#
# for i in range(randint(5, 10)):
#
#     # generate time in string format
#     dtint = randint(0, 2000000000 - 1)
#     print dtint
#     dtstr = ctime(dtint)
#
#     # generate user name, length:4~7
#     shorter = randint(4, 7)
#     em = ''
#     for j in range(shorter):
#         em += choice(lowercase)
#
#     # generate domain name, length:shorter~12
#     longer = randint(shorter, 12)
#     dn = ''
#     for j in range(longer):
#         dn += choice(lowercase)
#     d='%s::%s@%s.%s::%d-%d-%d ' % (dtstr, em, dn, choice(doms), dtint, shorter, longer)
#
#     print d
#     myfile.write(d+'\n')
#     data.append(d)
# myfile.close()
#
# cPickle.dump(data,open("data.pkl","wb"))
#cPickle.load(open("data.pkl","rb"))

import re

# p = re.compile(r'^(Mon|Tue|Wed)')
p = re.compile(r'^(\w{3})')

f = open('data.txt', 'r')

for eachLine in f.readlines():
    m = p.match(eachLine.strip())
    if m is not None:
        print m.groups()

f.close()