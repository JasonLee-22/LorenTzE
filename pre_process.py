from main import args
fs = []
dataset = args.dataset
entities = []
rels = []
ts = []
if dataset == 'ICESW14':
    f1 = open('./data/ICEWS14/train.txt')
    f2 = open('./data/ICEWS14/test.txt')
    fs = [f1, f2]
elif dataset=='GDELT':
    f1 = open('./data/GDELT/train.txt')
    f2 = open('./data/GDELT/test.txt')
    f3 = open('./data/GDELT/valid.txt')
    fs = [f1,f2,f3]


samples = f1.readlines() + f2.readlines() + f3.readlines()
for i in samples:
    h, r, t, time = i.split()[:-1]
    entities.append(int(h))
    entities.append(int(t))
    rels.append(int(r))
    ts.append(time)
entities = set(entities)
rels  =set(rels)
ts = set(ts)
print(len(entities), len(rels), len(ts))
'''for i in samples:
    h, r, t, time = i.split()[:-1]
    if h not in entities:
        entities.append(h)
    if t not in entities:
        entities.append(t)
    if r not in rels:
        rels.append(r)
    if time not in ts:
        ts.append(time)'''

'''f = open('./data/ICEWS14/stat2.txt', 'w')
f.write('\t'.join([str(len(entities)), str(len(rels)), str(len(ts))]))

print(len(entities), len(rels), len(ts))'''