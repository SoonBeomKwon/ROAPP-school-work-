import csv
import random
random.seed()

def weightBreath(x): #호흡 
    if x>=44:
        return 20
    elif x<17:
        return 0
    else:
        return x*(20/27)-(340/27)

def weightHeartbeat(x): #심박수 
    if x<=80:
        return 0
    elif x>150:
        return 20
    else:
        return x*(2/7)-(160/7)

def weightBrainpulse(x): #뇌파 
    if x>=30:
        return 20
    elif x<8:
        return 0
    else:
        return x*(10/11)-(80/11)

def weightStep(x):
    if x>=10000:
        return 0
    elif x<1000:
        return 20
    else:
        return x*(-1/450)+200/9

def weightOxygen(x): #산소포화도 
    if x>=100:
        return 0
    elif x<70:
        return 20
    else:
        return x*(-2/3)+(200/3)

csvfile=open('./pwoman.csv','wt')
writer=csv.writer(csvfile,delimiter=',')

for i in range(10000):
    t=random.gauss(22,10)
    if t<5:
        t=5
    h=random.gauss(100,25)
    o=random.gauss(20,10)
    if o<1:
        o=1
    s=random.gauss(7000,3000)
    if s<0:
        s=0
    b=random.gauss(100,15)
    total=weightBreath(t)+weightHeartbeat(h)+weightBrainpulse(o)+weightStep(s)+weightOxygen(b)
    if total>50:
        writer.writerow([t,h,o,s,b,'1'])
    else:
        writer.writerow([t,h,o,s,b,'0'])
