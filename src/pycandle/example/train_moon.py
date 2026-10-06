from pycandle.data_prep import load_moon, train_test_split
from pycandle.GD import GD, accuracy
from pycandle.nn import MLP

TEST_RATIO = 0.2
EPOCHS = 20
LR = 0.002

print('Loading data')
X, y = load_moon()

print('Spliting train/test set')
X_train, y_train, X_test, y_test = train_test_split(X,y, TEST_RATIO)
print(f'Training set has {len(X_train)} rows')
print(f'Testset has {len(X_test)} rows')


print('initialize MLP model')
model= MLP(len(X_train[0]), [4,3,2,1])

print('Running Gradient Descent')
history = GD(X,y, model, EPOCHS, LR)

    
    
print('Computing accuracy')
print(f'Accuracy of training set: {accuracy(X_train, y_train, model)}')
print(f'Accuracy of testset: {accuracy(X_test, y_test, model)}')
    

 

 
