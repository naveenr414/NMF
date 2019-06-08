import numpy as np
import pandas as pd

def parse(file_name):
    """ Parse mutations file"""
    f = open(file_name).read().split("\n")[:-1]
    cols = f[0].split("\t")

    all_patients = {}
    for i in range(1,len(f)):
        f[i] = f[i].split("\t")
        name = f[i][0]
        current_patient = {}
        for j in range(1,len(cols)):
            current_patient[cols[j].lower()] = f[i][j]
        all_patients[name] = current_patient

    return all_patients

def select_rows(data,current_rows,target_rows):
    allowed_rows = [i for i in range(len(current_rows)) if current_rows[i] in target_rows]
    return data[allowed_rows]

def select_rows_and_columns(data,current_rows,target_rows):
    return select_rows(select_rows(data.T,current_rows,target_rows).T,current_rows,target_rows)

def parse_dna(file_name,given_patients,selected_pathways=["BRCA1","BRCA2","RAD51"]):
    """ Parse pathways file"""
    f = open(file_name).read().split("\n")[:-1]
    f = [i.split("\t") for i in f]
    dna_pathways = f[0]
    
    has_pathway = []
    used_patients = []
    for sample in f[1:]:
        if sample[0] in given_patients:
            used_patients.append(sample[0])
            for pathway in selected_pathways:
                if int(sample[dna_pathways.index(pathway)]):
                    has_pathway.append(True)
                    break
            else:
                has_pathway.append(False)

    weight_matrix = np.zeros((len(used_patients),len(used_patients)))

    for i in range(len(used_patients)):
        for j in range(len(used_patients)):
            weight_matrix[i][j] = int(has_pathway[i] == has_pathway[j])
    
    return weight_matrix,used_patients
    
def parse_counts(file_name):
    """ Parse mutations"""
    f = open(file_name).read().split("\n")[:-1]
    categories = f[0].split("\t")[1:]
    f = f[1:]
    numbers = []
    patients = []
    for i in range(len(f)):
        data = f[i].split("\t")
        patients.append("-".join(data[0].split("-")[:3]))
        numbers.append([float(x) for x in data[1:]])

    return np.array(numbers).astype(np.float32),patients,categories

def parse_cna(file_name):
    f = open(file_name).read().split("\n")[1:-1]
    matrix = np.zeros((len(f),len(f[0].split("\t"))-1))
    for i in range(len(f)):
        line = f[i].split("\t")[1:]
        for j in range(len(line)):
            matrix[i,j] = float(line[j])

    matrix = matrix.T
    distances = np.zeros((len(matrix),len(matrix)))

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i,j] = np.sum(np.abs(matrix[i]-matrix[j]))

    distances = np.max(distances)-distances
    distances/=np.max(distances)
    return distances

def parse_cna_patients(file_name):
    patients =  open(file_name).read().split("\t")[1:]
    patients = ["-".join(i.split("-")[:-1]) for i in patients]
    return patients

def categorical_matrix(file_name,patients,category):
    patient_dict = parse(file_name)
    category = category.lower()
    matrix = np.zeros((len(patients),len(patients)))
    for i in range(len(patients)):
        for j in range(len(patients)):
            name_one = patients[i]
            name_two = patients[j]


            if name_one in patient_dict and name_two in patient_dict:
                if category in patient_dict[name_one]:
                    value_one = patient_dict[name_one][category]
                    value_two = patient_dict[name_two][category]

                    if value_one not in ['NaN',''] and value_two not in ['NaN','']:
                        if value_one == value_two:
                            matrix[i,j] = 1
    return matrix

def number_matrix(file_name,patients,category):
    """ Things like age"""
    patient_dict = parse(file_name)
    category = category.lower()
    matrix = np.zeros((len(patients),len(patients)))
    for i in range(len(patients)):
        for j in range(len(patients)):
            name_one = patients[i]
            name_two = patients[j]


            if name_one in patient_dict and name_two in patient_dict:
                if category in patient_dict[name_one]:
                    value_one = patient_dict[name_one][category]
                    value_two = patient_dict[name_two][category]

                    if value_one not in ['NaN',''] and value_two not in ['NaN','']:
                        matrix[i,j] = float(value_one)-float(value_two)
    matrix/=np.sum(np.abs(matrix))
    return np.sum(matrix)-matrix

def matrix(file_name,patients):    
    weights = np.zeros((len(patients),len(patients)))
    f = open(file_name).read().split("\n")
    for i in f:
        name_one = i.split("\t")[0]
        name_two = i.split("\t")[1]
        score = float(i.split("\t")[2])

        weights[patients.index(name_one),patients.index(name_two)] = score
        weights[patients.index(name_two),patients.index(name_one)] = score

    for i in range(len(patients)):
        weights[i,i] = 1


    return weights

def parse_pathways(file_name,pathways_wanted):
    df = pd.read_csv(file_name, sep='\t')
    names = df['id'].tolist()
    df = df[pathways_wanted]
    has_mutation = []
    for i,row in df.iterrows():
        for pathway in pathways_wanted:
            if row[pathway] != 0:
                has_mutation.append(1)
                break
        else:
            has_mutation.append(0)

    similarity_matrix = np.zeros((len(names),len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            similarity_matrix[i,j] = int(has_mutation[i] == has_mutation[j])
    
    return similarity_matrix,names
