import glob
""" Assure that all the signatures and exome data are in the same order
    In terms of mutations """

def get_ordering_exome(f):
    ordering = open(f).read().split("\n")[1:]
    ordering = [i.split("\t")[0] for i in ordering][:-1]
    return ordering

def get_ordering_signature(f):
    ordering = open(f).read().split("\n")[1:]
    ordering = [i.split("\t")[2] for i in ordering]
    return ordering

exome_files = glob.glob("../alexsandrov/exome/*.txt")

#Set the correct order
correct_ordering = get_ordering_exome(exome_files[0])

# Use a dict, so that ordering[0] -> 0
# This allows us to use python's sort function
# With the compare function being ordering_dict[i]
ordering_dict = {}
for i,mutation in enumerate(correct_ordering):
    ordering_dict[mutation] = i

# Ensure that all the other exome_files have the same order
for f in exome_files:
    file_ordering = get_ordering_exome(f)
    assert file_ordering == correct_ordering 

# Ensure that the signature file has the same order
signature_file = "../alexsandrov/signatures.txt"
lines = open(signature_file).read().split("\n")
lines = sorted(lines,key = lambda x: ordering_dict[x.split("\t")[2]] if x.split("\t")[2] in ordering_dict  else -1)

w = open(signature_file,"w")
w.write("\n".join(lines))
w.close()

ordering = get_ordering_signature(signature_file)

assert ordering == correct_ordering 
