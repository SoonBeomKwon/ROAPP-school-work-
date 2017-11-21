import csv
import random
random.seed()

def weightTemperature(x): #당뇨병 체온 낮다
    if x>=36.5:
        return 0
    elif x<35:
        return 10
    else:
        return x*6.66667-243.333

def weightGlucose(x):  #혈당
    if x<=126:
        return 0
    elif x>200:
        return 50
    else:
        return x*(20/37)-(2520/37)

def weightBloodPressure(x): #혈압  --고치는중
    if x>=140:
        return 20
    elif x<90:
        return 0
    else:
        return x*(2/5)-36

def weightStep(x): #걸음수
    if x>=10000:
        return 0
    elif x<1000:
        return 10
    else:
        return x*(-1/900)+100/9

def weightBMI(x): #BMI
    if x<=20:
        return 0
    elif x>35:
        return 10
    else:
        return x*(2/3)-(40/3)

csvfile=open('./diabetes.csv','wt')
writer=csv.writer(csvfile,delimiter=',')

for i in range(10000):
    t=random.gauss(36.5,2)
    h=random.gauss(150,40)

    o=random.gauss(100,35)
    if o<70:
        o=70
    s=random.gauss(7000,3000)
    if s<0:
        s=0
    b=random.gauss(25,4)

    total=weightTemperature(t)+weightGlucose(h)+weightBloodPressure(o)+weightStep(s)+weightBMI(b)

    if total>50:
        writer.writerow([t,h,o,s,b,'1'])
    else:
        writer.writerow([t,h,o,s,b,'0'])
