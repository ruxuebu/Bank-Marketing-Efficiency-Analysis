import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#job = {"\"admin.\"":[1,0,0,0,0,0,0,0,0,0,0,0],"\"unknown\"":[0,1,0,0,0,0,0,0,0,0,0,0],"\"unemployed\"":[0,0,1,0,0,0,0,0,0,0,0,0],"\"management\"":[0,0,0,1,0,0,0,0,0,0,0,0],"\"housemaid\"":[0,0,0,0,1,0,0,0,0,0,0,0],"\"entrepreneur\"":[0,0,0,0,0,1,0,0,0,0,0,0],"\"student\"":[0,0,0,0,0,1,0,0,0,0,0,0],"\"blue-collar\"":[0,0,0,0,0,0,0,1,0,0,0,0],"\"self-employed\"":[0,0,0,0,0,0,0,0,1,0,0,0],"\"retired\"":[0,0,0,0,0,0,0,0,0,1,0,0],"\"technician\"":[0,0,0,0,0,0,0,0,0,0,1,0],"\"services\"":[0,0,0,0,0,0,0,0,0,0,0,1]}
#marital = {"\"married\"":[1,0,0],"\"divorced\"":[0,1,0],"\"single\"":[0,0,1]}
#education ={"\"unknown\"":[1,0,0,0],"\"secondary\"":[0,1,0,0],"\"primary\"":[0,0,1,0],"\"tertiary\"":[0,0,0,1]}
#binary = {"\"yes\"":[1,0],"\"no\"":[0,1]}
#contact = {"\"unknown\"":[1,0,0],"\"telephone\"":[0,1,0],"\"cellular\"":[0,0,1]}
#month = {"\"jan\"":[1,0,0,0,0,0,0,0,0,0,0,0],"\"feb\"":[0,1,0,0,0,0,0,0,0,0,0,0],"\"mar\"":[0,0,1,0,0,0,0,0,0,0,0,0],"\"apr\"":[0,0,0,1,0,0,0,0,0,0,0,0],"\"may\"":[0,0,0,0,1,0,0,0,0,0,0,0],"\"jun\"":[0,0,0,0,0,1,0,0,0,0,0,0],"\"jul\"":[0,0,0,0,0,1,0,0,0,0,0,0],"\"aug\"":[0,0,0,0,0,0,0,1,0,0,0,0],"\"sep\"":[0,0,0,0,0,0,0,0,1,0,0,0],"\"oct\"":[0,0,0,0,0,0,0,0,0,1,0,0],"\"nov\"":[0,0,0,0,0,0,0,0,0,0,1,0],"\"dec\"":[0,0,0,0,0,0,0,0,0,0,0,1]}
#poutcome ={ "\"unknown\"":[1,0,0,0],"\"other\"":[0,1,0,0],"\"failure\"":[0,0,1,0],"\"success\"":[0,0,0,1]}
result = {"\"yes\"":1,"\"no\"":0}
job = {"\"admin.\"":[1],"\"unknown\"":[2],"\"unemployed\"":[3],"\"management\"":[4],"\"housemaid\"":[5],"\"entrepreneur\"":[6],"\"student\"":[7],"\"blue-collar\"":[8],"\"self-employed\"":[9],"\"retired\"":[10],"\"technician\"":[11],"\"services\"":[12]}
marital = {"\"married\"":[1],"\"divorced\"":[2],"\"single\"":[3]}
education ={"\"unknown\"":[1],"\"secondary\"":[2],"\"primary\"":[3],"\"tertiary\"":[4]}
binary = {"\"yes\"":[0],"\"no\"":[1]}
contact = {"\"unknown\"":[1],"\"telephone\"":[2],"\"cellular\"":[3]}
month = {"\"jan\"":[1],"\"feb\"":[2],"\"mar\"":[3],"\"apr\"":[4],"\"may\"":[5],"\"jun\"":[6],"\"jul\"":[7],"\"aug\"":[8],"\"sep\"":[9],"\"oct\"":[10],"\"nov\"":[11],"\"dec\"":[12]}
poutcome ={ "\"unknown\"":[1],"\"other\"":[2],"\"failure\"":[3],"\"success\"":[4]}
#result = {"\"yes\"":[1,0],"\"no\"":[0,1]}



#sigmoid,1/(1+e^-x)


def dataTransfer(data):
    udata =[]
    lab_data =[]

    udata.append(int(data[0]))


    udata+=job[data[1]]
    udata+=marital[data[2]]
    udata+=education[data[3]]
    udata+=binary[data[4]]

    udata.append(int(data[5]))

    udata+=binary[data[6]]
    udata+=binary[data[7]]
    udata+=contact[data[8]]

    udata.append(int(data[9]))


    udata+=month[data[10]]

    udata.append(int(data[11]))
    udata.append(int(data[12]))
    udata.append(int(data[13]))
    udata.append(int(data[14]))
 
    udata+=poutcome[data[15]]
    lab_data.append(result[data[16]])
    return udata, lab_data

def read_data(name):
    data = []
    lab = []
    t_data =[]
    t_lab =[]
    with open(name) as f:
         f_csv = csv.reader(f)
         headers = next(f_csv)
         for row in f_csv:
             sdata = row[0].split(";")
             t_data,t_lab = dataTransfer(sdata)
             data.append(np.ravel(np.array(t_data)))
             lab.append(t_lab)
    f.close()
    return np.array(data),np.ravel(np.array(lab))
  
def weight_f(W,train_data,train_lab,test_data,test_lab):
     clf = RandomForestClassifier(n_estimators=100,max_depth=15,criterion='entropy',  class_weight={1: W} )
     clf.fit(train_data, train_lab)
     pred = clf.predict(test_data)
     y = 0
     n = 0
     yy = 0
     nn = 0
     yt = 0
     nt =0
     for i in range(len(pred)):
         if(test_lab[i] == 1):
             yt+=1
         if(pred[i] == 1):
             y+=1
             if(pred[i] == test_lab[i]):
                 yy+=1
         else:
             n+=1
             if(pred[i] == test_lab[i]):
                 nn+=1
     return (yy/yt,y/len(pred),yy/y)

    

def main():
    train_data,train_lab = read_data("bank.csv")
    test_data,test_lab = read_data("bank-full.csv")
    x = []
    y = []
    W =1
    e=[]
    for i in range(30):
        a,b,c=weight_f(W,train_data,train_lab,test_data,test_lab)
        x.append(1-a)
        y.append(b)
        e.append(c)
        if(W!=1):
            W = W*1.5
        else:
            W = 100
    #plt.plot(y, x,"o")
    #plt.xlabel('Cost')
    #plt.title('Potential customer cover rate vs Cost')
    #plt.ylabel('Potential customer cover rate')
    #plt.show()


    plt.plot(x, e,"o")
    plt.xlabel('Cost')
    plt.title(' Potential customer cover rate vs Cost')
    plt.ylabel('Potential customer cover rate')
    plt.show()


    
    

    

main()