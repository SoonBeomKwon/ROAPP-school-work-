import numpy as np
import tensorflow as tf
import sys
import random
random.seed()
np.random.seed()
tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator=data-np.min(data,0)
    denominator=np.max(data,0)-np.min(data,0)
    #noise term prevents the zero division
    return numerator/(denominator+1e-7)

def printProgress(iteration, total, prefix='',suffix='',decimals=1,barLength=100):
    formatStr="{0:."+str(decimals)+"f}"
    percent=formatStr.format(100*(iteration/float(total)))
    filledLength=int(round(barLength*iteration/float(total)))
    bar='#'*filledLength+'-'*(barLength-filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration==total:
        print("")
    sys.stdout.flush()

TB_SUMMARY_DIR='./tb/diabetes_DEEP'
iteration=50000

xy=np.loadtxt('pima-indians-diabetes.csv',delimiter=',',dtype=np.float32)
np.random.shuffle(xy)
x_data_t=xy[:,0:-1]
x_data=MinMaxScaler(x_data_t)
y_data=xy[:,[-1]]

#train/test split
train_size=int(len(y_data)*0.7)
test_size=len(y_data)-train_size
trainX, testX=np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])
trainY, testY=np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])
testX_t=np.array(x_data_t[train_size:len(x_data_t)])

X=tf.placeholder(tf.float32, shape=[None,8])
Y=tf.placeholder(tf.float32, shape=[None,1])

with tf.variable_scope('layer1') as scope:
    W1=tf.get_variable("W1",shape=[8,10],initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.Variable(tf.random_normal([10],name='bias1'))
    layer1=tf.nn.relu(tf.matmul(X,W1)+b1)

    tf.summary.histogram("X", X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias", b1)
    tf.summary.histogram("layer", layer1)

with tf.variable_scope('layer2') as scope:
    W2=tf.get_variable("W2",shape=[10,10],initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.Variable(tf.random_normal([10],name='bias2'))
    layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)

    tf.summary.histogram("weights", W2)
    tf.summary.histogram("bias", b2)
    tf.summary.histogram("layer", layer2)

with tf.variable_scope('layer3') as scope:
    W3=tf.get_variable("W3",shape=[10,10],initializer=tf.contrib.layers.xavier_initializer())
    b3=tf.Variable(tf.random_normal([10],name='bias3'))
    layer3=tf.nn.relu(tf.matmul(layer2,W3)+b3)

    tf.summary.histogram("weights", W3)
    tf.summary.histogram("bias", b3)
    tf.summary.histogram("layer", layer3)

with tf.variable_scope('layer4') as scope:
    W4=tf.get_variable("W4",shape=[10,1],initializer=tf.contrib.layers.xavier_initializer())
    b4=tf.Variable(tf.random_normal([1],name='bias4'))
    hypothesis=tf.sigmoid(tf.matmul(layer3,W4)+b4)

    tf.summary.histogram("weights", W4)
    tf.summary.histogram("bias", b4)
    tf.summary.histogram("hypothesis", hypothesis)

cost=-tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

tf.summary.scalar("loss", cost)

#Summary
summary=tf.summary.merge_all()

predicted=tf.cast(hypothesis>0.5,dtype=tf.float32)
compare=tf.cast(tf.equal(predicted,Y),dtype=tf.float32)
accuracy=tf.reduce_mean(compare)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #Create summary writer
    writer=tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)
    global_step=0

    print("Start learning!")
    for step in range(iteration):
        s,_=sess.run([summary,train], feed_dict={X: trainX, Y: trainY})
        writer.add_summary(s,global_step=global_step)
        global_step+=1
        printProgress(step,iteration,'Progress: ','Complete',1,50)
    print("Learning Finished!")

    h,p,a=sess.run([hypothesis,predicted,accuracy], feed_dict={X:testX,Y:testY})

    yes_c=0
    no_c=0
    for i in range(len(p)):
        if p[i]==testY[i]:
            yes_c+=1
        else:
            no_c+=1
    print("Accuracy: ", a, "Yes: ",yes_c,", No: ",no_c,", total: ",yes_c+no_c)

    while(True):
        print("Press Enter(exit q enter)")
        key=input()
        if key=='q':
            break
        sample_t=np.array([random.uniform(3.8-3.4, 3.8+3.4),
                        random.uniform(120.9-32.0, 120.9+32.0),
                        random.uniform(69.1-19.4, 69.1+19.4),
                        random.uniform(20.5-16.0, 20.5+16.0),
                        random.uniform(79.8-115.2, 79.8+115.2),
                        random.uniform(32.0-7.9, 32.0+7.9),
                        random.uniform(0.5-0.3, 0.5+0.3),
                        random.uniform(33.2-11.8, 33.2+11.8)])
        print(sample_t)
        sample=MinMaxScaler(sample_t)
        sample=np.array([sample])
        sample_h,sample_p=sess.run([hypothesis,predicted], feed_dict={X:sample})

        print("{},{}".format(sample_h,sample_p))
        if sample_p==1.0:
            print("Diabetes is suspected.")
        else:
            print("You are healthy.")
