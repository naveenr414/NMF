import glob

z = glob.glob("./run_three_10_trials/tmp*")

score_list = {}
baseline = {}
by_number = {}

for i in z:
    f = open(i).read().split("\n")[1:-1]
    exec("lr="+f[2])
    lr = lr[1]
    hidden = float(f[3].split(" ")[0])
    lamb = int(f[3].split(" ")[1])
    score = float(f[4].split(": ")[1])
    time = float(f[5].split(" ")[-1])

    if hidden not in score_list:
        score_list[hidden] = []


    score_list[hidden].append((lamb,score))
    if lamb == 0:
        baseline[hidden] = score

    if lamb not in by_number:
        by_number[lamb] = {}

    by_number[lamb][hidden] = score

for i in score_list.keys():
    score_list[i] = sorted(score_list[i],key=lambda x: x[1])
    score_list[i] = [(a,round(b/baseline[i],3)) for a,b in score_list[i]]

for i in score_list.keys():
    print(score_list[i])
