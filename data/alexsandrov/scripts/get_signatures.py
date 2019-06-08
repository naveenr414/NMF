def get_signature_numbers(cancer,header=""):
    cancer = cancer.capitalize()
    which_signatures = open(header+"../signature_map.txt").read().split("\n")
    found = False
    others = []
    for i in which_signatures:
        if cancer in i:
            found = True
            others = i.split(" ")[1:]
            if "Other" in i:
                others.remove("Other")

            others = [i for i in others if i!='']

    if(found):
        return others
    else:
        raise Exception("Couldn't find Cancer")
    
def get_signatures(cancer,header=""):
    numbers = get_signature_numbers(cancer,header=header)
    all_signatures = open(header+"../signatures.txt").read().split("\n")
    header = all_signatures[0].replace("Signature ","Signature")
    signatures = []

    for number in numbers:
        current_signature = []
        col_number = header.split("\t").index("Signature"+number)
        for i in range(1,len(all_signatures)):
            current_signature.append(float(all_signatures[i].split("\t")[col_number]))

        signatures.append(current_signature)

    return signatures 
