# SVM tree
We implement a tree of SVMs (Software Vector Machine) class with Python. We also use some libraries: Numpy, Sklearn, NetworkX and Matplotlib.

## How to use it? 
It’s a very simple class and very easy to use. It only have 6 methods and we are going to explain them below. 

First of all, download the repository.

```
git clone https://github.com/treeSVM/treeSVM.git
```

Then move the folder to your project and import it in the python file.

```
from SVMTree import SVMTree
```

Now we have all set and ready to go. We just need to create a variable and assign SVMTree constructor.

```
svm = SVMTree()
``` 

### Methods

#### -Fit(X,Y)
This method allow you to train your svm tree. You need 2 parameters: 

X —> test data  
Y —> test target 

Both params need to be in Numpy array format. X may be a matrix, while Y need to be a single array of elements.

```
fit(xTrain, yTrain)
```

 #### -Predict(X)
This method gives you the prediction for a dataset. Again, it need to be in Numpy array format. 
 
```
predict(xTest)
```

#### -GetPerformance(Y)
This method pint the results obtained from comparing the prediction generated from your svm tree and the results passed in yTest as parameter. 

```
getPerformance(yTest) 
```

#### -PrintTree()
This method prints a plot of all of the tree’s nodes in its corresponding position. 

```
printTree() 
```

#### -SaveModel(path/name)
This methods saves your model in the destination ‘path’, creating a new binary file with 'modelName'. 

```
-saveModel(‘path/modelName’) 
```

#### -ReloadModel(path/name)
This method reload your model specified with 'modelName' from the origin ‘path’   

```
reloadModel(‘path/modelName’) 
```
