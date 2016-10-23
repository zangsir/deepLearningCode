# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import re
import matplotlib.pyplot as plt
import os.path
import random





#these functions are called by load_data in the case of there are a few nominal columns in the beginning of the data
def get_num_features(line):
    pat='(,\d,\d,\d,\d,\d,\d,\d,\d,\d,\d,\d,\d+,\d+,\d+,\d+,\d+,)+'#entity data
    #pat = '(,\d+,\d+,\d+,\d+,\d+,)+'#for non-entity data
    match=re.search(pat,line)
    if match:
        return match.start()
    
def strip_first_col(fname):
    with open(fname, 'r') as fin:
        next(fin)
        for line in fin:
            #print (line)
            try:
                #print (line[get_num_features(line)+1:])
                yield line[get_num_features(line)+1:]
            except IndexError:
                continue

#data = np.loadtxt(strip_first_col(datafile),skiprows=1,delimiter=',')



def load_nominal(datafile):
    output=[]
    f=open(datafile,'r').read().split('\n')
    
    for line in f[1:]:
        end_nominal=get_num_features(line)
        output.append(line[:end_nominal])
    
    return np.array(output)


def load_data(datafile,header=True):
    if header==True:
        data = np.loadtxt(strip_first_col(datafile),skiprows=1,delimiter=',')
    else:
        data = np.loadtxt(datafile, delimiter=',',skiprows=1)

    # first ten values are the one hot encoded y (target) values
    y = data[:, -1]
    
    data = data[:, :-1]  # x data
    # data = data - data.mean(axis = 1)
    data -= data.min()  # scale the data so values are between 0 and 1
    data /= data.max()  # scale
    out = []
    labels=[]
    print(data.shape)
    label_dict={1:[0,1],0:[1,0]}
    # populate the tuple list with the data
    for i in range(data.shape[0]):
        fart = list((data[i, :].tolist()))  # don't mind this variable name
        out.append(fart)
        #two classes one hot coding:two classes are [NS,S] y=1:[0,1]; y=0:[1,0]
        labels.append(label_dict[y[i]])

    return np.array(out,dtype=np.float32),np.array(labels,dtype=np.float32)




def randomize(dataset, labels,nominals):
    seed = 2335
    random.seed(seed)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    shuffled_nominals = nominals[permutation]
    return shuffled_dataset, shuffled_labels, shuffled_nominals



def get_singleton(labels):
    inds_notsgt=[]
    inds_sgtn=[]
    for i in range(len(labels)):
        if labels[i][0]==1:
            inds_sgtn.append(i)
        else:
            inds_notsgt.append(i)
    return inds_notsgt,inds_sgtn




def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



def plot_tf(train,valid):
    x=np.linspace(0,len(train),len(train))
    plt.plot(x,train,'bo',label='train')
    plt.plot(x,valid,'rx',label='validation')
    plt.legend(loc=4)




def precision_recall(predictions,labels):
#TP=np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1) == 0)

    TP=0
    predicted_sgt=np.sum(np.argmax(predictions,1)==0)
    real_sgt=np.sum(np.argmax(labels,1)==0)
    #print('[predicted_sgt:'+str(predicted_sgt)+"  real_sgt:"+str(real_sgt)+']')

    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(labels[i]) == 0:
            TP+=1
    if predicted_sgt!=0:
        precision=float(TP)/predicted_sgt
    else:
        precision=0
    
    recall=float(TP)/real_sgt
    #print ("["+str(precision)+str(recall)+']')
    if precision==recall==0:
        F1=0
    else:
        F1=(2*precision*recall)/(precision+recall)
    return precision,recall,F1



def get_error_cases(predictions,labels):
    false_sgt=[]
    false_nsgt=[]
    for i in range(len(predictions)):
            if np.argmax(predictions[i]) == 0 and  np.argmax(labels[i]) == 1:
                false_sgt.append(i)
            elif np.argmax(predictions[i]) == 1 and  np.argmax(labels[i]) == 0:
                false_nsgt.append(i)
    return false_sgt,false_nsgt





#set of a tf graph
def run_tf(num_nodes= 1024,batch_size = 128,num_steps = 10000,report_step=250, learning_rate=0.1):
    
    input_size=X.shape[1]#dimension of each input vector
    print('================num_nodes_hidden,batch_size:',num_nodes,batch_size)
    print('================input_feature_size:',input_size)
    print('================learning rate:',learning_rate)
    num_labels=2

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, input_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_1 = tf.Variable(
          tf.truncated_normal([input_size, num_nodes]))
        biases_1 = tf.Variable(tf.zeros([num_nodes]))
        weights_2 = tf.Variable(
          tf.truncated_normal([num_nodes, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        relu_layer=tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)#notice the shape of tf_train_dataset and weights_1
        logits = tf.matmul(relu_layer, weights_2) + biases_2
        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
         tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
        test_prediction =  tf.nn.softmax(
         tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)


    
    print ('=================num steps:',num_steps)
    train_acc=[]
    valid_acc=[]
    train_F1=[]
    valid_F1=[]
    loss_log=[]
    false_sgt_log=[]
    false_nsgt_log=[]



    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % report_step == 0):
                #print("Minibatch loss at step %d: %f" % (step, l))
                #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

                #print("Minibatch prec,recall,F1: %.3f %.3f %.3f"  % precision_recall(predictions, batch_labels))
                #print("validation prec,recall,F1: %.3f %.3f %.3f" % precision_recall(valid_prediction.eval(), valid_labels))

                #get some error cases in training (not yet validation errors)
                false_sgt,false_nsgt=get_error_cases(predictions,batch_labels)
                false_sgt_batch=offset+np.array(false_sgt)
                false_nsgt_batch=offset+np.array(false_nsgt)
                false_sgt_log.append(false_sgt_batch)
                false_nsgt_log.append(false_nsgt_batch)

                loss_log.append(l)
                train_acc.append(accuracy(predictions, batch_labels))
                valid_acc.append(accuracy(valid_prediction.eval(), valid_labels))
                train_F1.append(precision_recall(predictions, batch_labels))
                valid_F1.append(precision_recall(valid_prediction.eval(), valid_labels))

                #print("===================")
        print ('final loss:', loss_log[-1])
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        print("Test precision, recall, F1: %.3f %.3f %.3f" % precision_recall(test_prediction.eval(), test_labels))
        output_prob=False
        if output_prob==True:
            sgt_probs = [k for k in test_prediction.eval() if np.argmax(k) == 0]
            for arr in sgt_probs:
                print (arr[0],",")
    plt.figure(0)
    plot_tf(train_acc,valid_acc)
    plt.title('accuracy')
    plt.savefig('plots/accuracy.pdf')

    #plot F1 scores
    f1t=[i[2] for i in train_F1]
    f1v=[i[2] for i in valid_F1]
    plt.figure(1)
    plot_tf(f1t,f1v)
    plt.title('F1 score')
    plt.savefig('plots/F1_score.pdf')
    
    #plot precision
    prect=[i[0] for i in train_F1]
    precv=[i[0] for i in valid_F1]
    plt.figure(2)
    plot_tf(prect,precv)
    plt.title('Precision')
    plt.savefig('plots/precision.pdf')

    # plot loss
    plt.figure(3)
    x=range(len(loss_log))
    plt.plot(x,loss_log)
    plt.ylim(0.4,1)
    plt.title('loss function')
    plt.savefig('plots/loss.pdf')
    return train_F1,valid_F1,loss_log, false_sgt_log,false_nsgt_log
















#main
print ("""============================================================================
============================================================================
============================================================================""")
import datetime
now = datetime.datetime.now()
print (now.isoformat())
output_error=True
#datafile='Data/singleton_data_new.tab'
datafile = 'Data/entity_data_onehot.csv'
datafile_first=datafile.split('.')[0]
#npy file is serialized numpy object file to save time
npy_file = datafile_first+".npy"
npy_labels_file=datafile_first+'_labels.npy'
if os.path.isfile(npy_file):
    print ("reading serialized numpy data file...")
    X=np.load(npy_file)
    labels=np.load(npy_labels_file)
else:
    print ('loading data file into numpy array...')
    X,labels = load_data(datafile)
    np.save(npy_file, X)
    np.save(npy_labels_file, labels)
nominals=load_nominal(datafile)
#shuffle dataset
dataset_shf,labels_shf, nominal_shf = randomize(X, labels, nominals)



#sample sgt and non-sgt data
ind_notsgt,ind_sgt=get_singleton(labels_shf)
singleton_data=dataset_shf[ind_sgt]
notsgt_data=dataset_shf[ind_notsgt]
singleton_labels=labels_shf[ind_sgt]
notsgt_labels=labels_shf[ind_notsgt]
singleton_nominal=nominal_shf[ind_sgt]
notsgt_nominal=nominal_shf[ind_notsgt]

#print (len(notsgt_data)+len(singleton_data)==len(labels_shf))
print ("non-singleton,singleton in the original dataset:", len(notsgt_data),len(singleton_data))



#construct the entire undersampled dataset
random.seed(2335)
num_notsgt_samples=100000
notsgt_inds=random.sample(xrange(len(notsgt_data)),num_notsgt_samples)
undersampled_notsgt=notsgt_data[notsgt_inds]
undersampled_notsgt_labels=notsgt_labels[notsgt_inds]
undersampled_notsgt_nominal=notsgt_nominal[notsgt_inds]

#construct the whole data set
undersampled_data=np.concatenate((undersampled_notsgt,singleton_data), axis=0)
undersampled_labels=np.concatenate((undersampled_notsgt_labels,singleton_labels), axis=0)
undersampled_nominal=np.concatenate((undersampled_notsgt_nominal,singleton_nominal), axis=0)

#construct the whole labels
#undersampled_notsgt_labels=
print ("final undersampled data set size:",len(undersampled_data))




#divide train, valid, test

dataset_shf_und,labels_shf_und,nominal_shf_und = randomize(undersampled_data, undersampled_labels,undersampled_nominal)
train_size=int(len(undersampled_data)*0.9)
val_size=int(len(undersampled_data)*0.05)
test_isze=int(len(undersampled_data)*0.05)
#get sizes of division
train_dataset=dataset_shf_und[:int(train_size),:]
valid_dataset=dataset_shf_und[int(train_size):int(train_size+val_size),:]
test_dataset=dataset_shf_und[int(train_size+val_size):,:]
train_labels=labels_shf_und[:int(train_size)]
valid_labels=labels_shf_und[int(train_size):int(train_size+val_size)]
test_labels=labels_shf_und[int(train_size+val_size):]
train_nominal=nominal_shf_und[:int(train_size)]
valid_nominal=nominal_shf_und[int(train_size):int(train_size+val_size)]
test_nominal=nominal_shf_und[int(train_size+val_size):]


#divide the dataset
train_dataset, train_labels, train_nominal = randomize(train_dataset, train_labels, train_nominal)
test_dataset, test_labels, test_nominal = randomize(test_dataset, test_labels, test_nominal)
valid_dataset, valid_labels, valid_nominal = randomize(valid_dataset, valid_labels, valid_nominal)
print (train_dataset.shape)


# tune hyper-parameters
t,v,l,fs,fns=run_tf(24,128,20000,10,0.1)

if output_error:
    print ("train_false_non-singleton:\n")
    # print ("FNS:",fns[-1])
    for line in train_nominal[fns[-1]]:
        print (line)
    print ("=================\ntrain_false_singleton:\n")
    if len(fs[-1])>0:
        # print ("last fs:",fs[-1])
        for line in train_nominal[fs[-1]]:
            print (line)
    # else:
      #  print (fs)



