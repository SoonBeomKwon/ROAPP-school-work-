import numpy as np
import tensorflow as tf
import sys
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

timesteps=seq_length=5
data_dim=8
hidden_dim=10
output_dim=1
learning_rate=0.01
iterations=10000

TB_SUMMARY_DIR='./tb/diabetes_RNN'

xy=np.loadtxt('insomnia.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
x_data=MinMaxScaler(x_data)
y_data=xy[:,[-1]]

#build a dataset
dataX=[]
dataY=[]
for i in range(0,len(y_data)-data_dim):
    _x=x_data[i:i+data_dim]
    _y=y_data[i+data_dim]
    #print(_x,"->",_y)
    dataX.append(_x)
    dataY.append(_y)

#train/test split
train_size=int(len(dataY)*0.7)
test_size=len(dataY)-train_size
trainX,testX=np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY,testY=np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X=tf.placeholder(tf.float32, shape=[None,data_dim, seq_length], name="X")
Y=tf.placeholder(tf.float32, shape=[None,1], name="Y")

#build a LSTM network
with tf.variable_scope('LSTM') as scope:
    cell=tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _state=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    Y_pred=tf.contrib.layers.fully_connected(outputs[:,-1],output_dim,activation_fn=tf.sigmoid)

    tf.summary.histogram("X",X)
    tf.summary.histogram("outputs",outputs)
    tf.summary.histogram("Y_pred",Y_pred)

loss=tf.reduce_mean(tf.square(Y_pred-Y))
optimizer=tf.train.AdamOptimizer(learning_rate)
train=optimizer.minimize(loss)

tf.summary.scalar("loss",loss)

#TensorBoard Summary
summary=tf.summary.merge_all()

predicted=tf.cast(Y_pred>0.5,dtype=tf.float32)

targets=tf.placeholder(tf.float32, [None,1])
predictions=tf.placeholder(tf.float32,[None,1])
rmse=tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))
accuracy=tf.reduce_mean(tf.cast(tf.equal(predictions,targets),dtype=tf.float32))

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    #Create summary writer
    writer=tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)
    global_step=0;

    for i in range(iterations):
        s,_,step_loss=sess.run([summary,train,loss],feed_dict={X: trainX, Y: trainY})
        printProgress(i,iterations,'Progress: ','Complete',1,50)
        writer.add_summary(s,global_step=global_step)
        global_step+=1

    y_pred, test_predict=sess.run([Y_pred, predicted], feed_dict={X:testX})
    rmse=sess.run(rmse,feed_dict={targets: testY, predictions: test_predict})
    a=sess.run(accuracy, feed_dict={targets:testY, predictions:test_predict})

    yes_c=0
    no_c=0

    yes_yes_c=0
    yes_no_c=0
    no_yes_c=0
    no_no_c=0
    for i in range(len(testY)):
        print("[predict, predict_r, real]","[",y_pred[i],", ",test_predict[i],", ",testY[i],"]", end=" ")
        if test_predict[i]==testY[i]:
            print("Yes")
            yes_c+=1
        else:
            print("No")
            no_c+=1

        if test_predict[i]==1:
            if test_predict[i]==testY[i]:
                yes_yes_c+=1
            else:
                yes_no_c+=1
        else:
            if test_predict[i]==testY[i]:
                no_no_c+=1
            else:
                no_yes_c+=1
    print("RMSE: {}".format(rmse))
    print("accuracy: {}".format(a))
    print("Yes: ",yes_c,", No: ",no_c,", total: ",yes_c+no_c)
    print("predict(yes) real(yes): {}, predict(yes) real(no): {}, total: {}, accuracy: {}".format(yes_yes_c, yes_no_c, yes_yes_c+yes_no_c, (yes_yes_c/(yes_yes_c+yes_no_c))))
    print("predict(no) read(no): {}, predict(no) real(yes): {}, total: {}, accuracy: {}".format(no_no_c, no_yes_c, no_no_c+no_yes_c, (no_no_c/(no_no_c+no_yes_c))))
