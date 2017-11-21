import numpy as np
import tensorflow as tf
import sys
tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator=data-np.min(data,0)
    denominator=np.max(data,0)-np.min(data,0)
    #noise term prevents the zero division
    return numerator/(denominator+1e-7)

timesteps=seq_length=5
data_dim=8
hidden_dim=10
output_dim=1
learning_rate=0.01
iterations=10000

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

saver=tf.train.Saver()

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    saver.restore(sess, 'save_ins')

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
