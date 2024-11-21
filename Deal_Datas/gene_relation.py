locs = []
# Open file containing SNP or gene info and read each line to get the first elementand store in 'locs' list
with open('snp_gene_HeadingDate.txt', 'r') as gs:
    for line in gs.readlines():
        locs.append(line.split()[0])
print(locs)

s = ''
# Open input file 'RiceNet.txt' for reading and output file 'gene_re_HeadingDate.txt' for writing
with open('RiceNet.txt', 'r') as rn, open('gene_re_HeadingDate.txt', 'w') as gr:
    for line in rn.readlines():
        info = line.split()
        # Check if both gene identifiers in the line are in the 'locs' list
        if info[0] in locs and info[1] in locs:
            # Build a formatted string with gene indices from 'locs' and additional info
            s += str(locs.index(info[0])) + '\t' + str(locs.index(info[1])) + '\t' + info[2] + '\n'
            gr.write(s)