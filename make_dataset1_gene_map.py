# /usr/local/bin

import re

## read probe_id and gene name as dict
with open('GDS4971/GPL6947.annot 2') as file:
    table1 = [[digit for digit in line.split('\t')] for line in file]
dict1 = {}
gene1 = {}
fmap = open('GDS4971/probe_gene_map.txt', 'w')
for i in range(31,48832):
	if table1[i][2] != '' and table1[i][2] not in gene1:
		fmap.write(table1[i][0] + '\t' + table1[i][2] + '\n')
		dict1[table1[i][2]] = table1[i][15] 
		gene1[table1[i][2]] = 1
fmap.close()	
