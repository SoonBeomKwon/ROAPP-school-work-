import csv
import random
random.seed()

def weightTemperature(x):
    if x<=36.5:
        return 0
    elif x>38:
        return 20
    else:
        return x*13.333-486.6667

def weightHeartbeat(x):
    if x<=80:
        return 0
    elif x>150:
        return 40
    else:
        return x*(4/7)-(320/7)

def weightOxygen(x):
    if x>=100:
        return 0
    elif x<70:
        return 10
    else:
        return x*(-1/3)+(100/3)

def weightStep(x):
    if x>=10000:
        return 0
    elif x<1000:
        return 20
    else:
        return x*(-1/450)+200/9

def weightBMI(x):
    if x<=20:
        return 0
    elif x>35:
        return 10
    else:
        return x*(2/3)-(40/3)

csvfile=open('./insomnia.csv','wt')
writer=csv.writer(csvfile,delimiter=',')

for i in range(10000):
    t=random.gauss(36.5,2)
    h=random.gauss(100,25)
    o=random.gauss(100,15)
    if o>100:
        o=100
    s=random.gauss(7000,3000)
    if s<0:
        s=0
    b=random.gauss(25,4)

    total=weightTemperature(t)+weightHeartbeat(h)+weightOxygen(o)+weightStep(s)+weightBMI(b)

    if total>50:
        writer.writerow([t,h,o,s,b,'1'])
    else:
        writer.writerow([t,h,o,s,b,'0'])
