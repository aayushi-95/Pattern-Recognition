import pandas as pd
import matplotlib.pyplot as pt
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Calculate Posterier Probablities of Misclassified Data
def postprob(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

#Load MNIST data set into a csv file
def parse_data(hwd_f, label_f, res_f, n):
    h = open(hwd_f, "rb")
    o = open(res_f, "w")
    l = open(label_f, "rb")
    h.read(16)
    l.read(8)
    imgs = []
    for i in range(n):
        img = [ord(l.read(1))]
        for j in range(28*28):
            img.append(ord(h.read(1)))
        imgs.append(img)
    for img in imgs:
        o.write(",".join(str(pix) for pix in img)+"\n")
    h.close()
    o.close()
    l.close()

#Conversion starts
parse_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
"MNIST_Database.csv", 60000)

parse_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
"MNIST_Database1.csv", 10000)

#Converting ans storing in matrix form for easy extrction and data manipulation
my_data=pd.read_csv("MNIST_Database.csv").as_matrix()
my_data_test = pd.read_csv("MNIST_Database1.csv").as_matrix()

#Naive Bayes Bernoulli distribution model
classifier = BernoulliNB() 

#Training of database begins
train_data=my_data[0:60000,1:]
train_label=my_data[0:60000,0]
classifier.fit(train_data,train_label) 

#Probabliies of class digits are
print("The class probabilities are:")
i=0
dig = classifier.predict_proba(train_data)
for i in range(10):
    print("Target digit =%s, Class probailities of the digits: \n%s" % (i, dig[i]))
    print("\n")
#Training ends    

#Testing begins
test_data=my_data_test[0:10000,1:]
test_label=my_data_test[0:10000:,0]

predict=classifier.predict(test_data)

#Analyzing performance of model by generating Confusion Matrix
ConfusionMatrix=confusion_matrix(test_label,predict)
print("The confusion matrix is:")
print(ConfusionMatrix)
pt.imshow(ConfusionMatrix)
print("\n")

#Testing set: Performance Merits
acc_rate =sklearn.metrics.accuracy_score(test_label, predict, sample_weight=None)
print("The testing set accuracy is:", acc_rate) 
print("The error rate of the MNIST testing set is:", 1 - acc_rate) 
print("\n")

#Classification Report
target_names = ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9']
print(classification_report(test_label, predict, target_names=target_names)) 
print("\n")

#Randomy picked first 5 misclassified digits and the posterier probablity calculation 
digit = [0]*10
actual_digit = [0]*10
error = 0

print("The five misclassified digits and their respective posterior probabilities are below:")
for i in range(0,10):
    digit[i] = postprob(train_label, lambda x: x == i)
    actual_digit[i] = (len(digit[i])/50000.00)

for i in range(0,9999):
    if predict[i]!=test_label[i]:
        if error < 5:
            error += 1
            num=test_data[i]
            num.shape=(28,28)
            pt.imshow(num,cmap='gist_heat')
            print("Predicted value:", predict[i], "Correct Value:", test_label[i], "Posterior probability:", actual_digit)
            print("\n")
            pt.show()









