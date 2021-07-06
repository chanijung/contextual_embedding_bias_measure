#Load Dataset

Title=[]
Salary=[]
flag=0
def isFloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

with open('data/employeesalaries2017.csv') as f:
    for line in f:
        if flag==0:
            flag=1
            continue
        row = line.split(',')
        Title.append(row[2])
        if isFloat(row[8]):
            Salary.append(float(row[8]))
        else:
            Title.pop()
            
            
        
Title_sorted = sorted(Title,key=dict(zip(Title, Salary)).get,reverse=True)

unique_titles= set()

Top_Titles= []

f = open("data/job_titles.txt", "a")

for i in Title_sorted:
    if i in unique_titles:
        continue
    else:
        Top_Titles.append(i.lower())
        unique_titles.add(i)
        w = i.lower()
        w = w.replace(" ", "_")
        f.write(w+"\n")


