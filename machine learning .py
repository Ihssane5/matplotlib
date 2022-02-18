#!/usr/bin/env python
# coding: utf-8

# In[4]:


## we will try to genarate a random set of numbers 
import numpy as np 
import matplotlib.pyplot as plt 
x = np.random.uniform(0,5,250) ## we wil generate 250 numbers between 0 and 5 and then plot the result 
plt.hist(x,5) # we split into 5 groups
plt.show()
print(np.percentile(x,75))


# In[13]:


## we will create a normal distribution gaussian distribution 
## by def the values are concentrated around a given value and not betwenn two values as we did before
import numpy as np 
import matplotlib.pyplot as plt 
x=np.random.normal(5.0,1.0,100000) ## 5 is the mean and 1 is the standard deviation 
plt.hist(x,100)
plt.show()


# In[14]:


## we will try a linear regression model as well 
## it's when we try to find the relationship between two variables and the importance exist when we try to predict future income 
## we will create our date 
import matplotlib.pyplot as plt 
x = [5,7,8,8,17,11,12,9,6]
y=[99,66,111,77,85,86,77,88,55]
plt.scatter(x,y)
plt.xlabel("the age of the car")
plt.ylabel("the speed of the car ")
plt.title("linear regression on the car dataset")
## we will import our library 
from scipy import stats
slope,intercept,r,p,std_err = stats.linregress(x,y)
# we will define our function 
def myfunction(x):
    return x*slope + intercept ##our function will return a new values of y 
my_model = list(map(myfunction,x))
plt.scatter(x,y)  ##we plot the original x and y values 
plt.plot(x,my_model) ## we plot our linear regression model 
plt.show() ## it the realation between our variables is nonlinear then a regression model will not be useful to us at all
## look if there is a linear correlation between x and y values using the r parameter 
print("the coefficient of correlatio of your values is :",r)
print("unfort the realtion between your values is not linear at all soo a regression model can't be applied ")
## but we will althought a linear one is soo bad for our data try it in order to predict the future values instead 
speed = myfunction(10)
print("the spped of a 10 years old car is: ",speed)


# In[39]:


## we will try a polynomial regression model now 
## we use it when there is no relation ship between our variables 
import matplotlib.pyplot as plt 

x = [5,7,8,8,17,11,12,9,6]
y=[99,66,111,77,85,86,77,88,55] # we will use the same data  as before 


import numpy as np
mymodel =np.poly1d(np.polyfit(x,y,3)) ## i think 3 mean that we have a polynome of the third degree
## and it's one dimentional polynome 
myline = np.linspace(1,22,2000) ## the last parameter make a smooth line 
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()
## look if there is a polynomial relationship between our values 
from sklearn.metrics import r2_score
print("the coefficient of correlation between x and y values is:",r2_score(y,mymodel(x)))
print("unfortunately there no polynomial relation ship between x and y values ")
## and then we can use it to make predictions 
speed = mymodel(17)
print("the speed of  a 17 yeras old car is :",speed)


# In[ ]:


## we will try to use the multiple regression now 
## it's when the input is not a single value as the simple one but instead a multiple values 
## the values that we make our prediction based on 
# yuppy we will use a real data set one 
# we will predict the emission of the co2 based on multiple features we have 


# In[61]:


import pandas as pd 
car_data =pd.read_csv("C:/Users/asus/Downloads/cars.csv")
print(car_data)
## we will split the independant variables from the dependent ones 
print(car_data.columns)
x = car_data[['Volume',"Weight"]]
y = car_data['CO2']
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x,y)
## our model is ready now ## we will use it to make predictions 
print("what the  CO2 emission of your car if its volume is 2300 and its weigt is equal 1600")
print("the answer is :",model.predict([[2300,1600]]),"of co2")
## we will look then at the coefficient that determine the relation between two independant variables 
print(model.coef_)


# In[73]:


## we will look at the scaling of our features that a lot of times can be very helpful for us when the data have diffenrent units and different size
## to scale our data we will use the standardization the value - mean/ standard deviation
import numpy as numpy 
import pandas as pd 
cars_data = pd.read_csv("C:/Users/asus/Downloads/cars.csv")
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x = cars_data[["Weight","Volume"]]
y = cars_data["CO2"]
scalex = scale.fit_transform(x)
print(scalex)
## we define our model 
model =linear_model.LinearRegression()
model.fit(scalex,y) ## when we scale our data we have to applied it as well when predicting values as well 
scaled = scale.transform([[2300,1.3]])
predictedco2= model.predict([scaled[0]])
print("the predicted value of co2 is :",predictedco2)


# In[96]:


## then  we will take a look to the train and test part 
# it's a method to evaluate the accuracy of our model 
## we will start with our data set 
import numpy as np 
import matplotlib.pyplot as plt 
numpy.random.seed(2)
x =np.random.normal(3,1,100)
y = np.random.normal(150,40,100)/x
plt.xlabel("the time spent in the market")
plt.ylabel("the value of the purchase ")
plt.title("the shopping habits")
plt.scatter(x,y)
plt.show()
## we will split our data  into  a train and a test dataset 
train_x=x[:80]
train_y = y[:80]
test_x = x[80:]
test_y =y[80:]
print("the plot of the training set")
plt.scatter(train_x,train_y)
plt.show()
print("the plot of the testing set")
plt.scatter(test_x,test_y)
plt.show()


# In[97]:


## i think the best model that fit our data well is a polynomial one 
#let's create one 
model = np.poly1d(np.polyfit(train_x,train_y,4)) #the degree of  the polynome is 4 
myline = np.linspace(0,6,100)
plt.scatter(train_x,train_y)
plt.plot(myline,model(myline))
plt.show()
## we see that our model is overfitting the data 
## wa can look at the r squared coefficient to see if the model fit our data
from sklearn.metrics import r2_score
r2=r2_score(train_y,model(train_x))
print("the squred coefficient is equal to:",r2)
print("the model suits the data well on the training data ")
## we will use our model to the test data 
coeff2 = r2_score(test_y,model(test_y))
print("now the coefficient on the testing data is :",coeff2)
print("the model fit the testing data as well but not as in the training set")
## then we will move to making predictions using our model 
print("the purchase of a custumer that spent 5 minutes is about:",model(5))


# In[98]:


## we will try the decision tree as well 
# the decision  tree is made in order to make future decison  based on our past aexperiences 
# in this examples we will use the data of persons (attend to the show or not) based on the age the nationality the rank and theexperience 

import pandas 
from sklearn import tree
import pydotplus   # i don't know how to download isn't already installed 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
import matplotlib.image as pltimg
## the decision tree handle just the numerical values only 
d={"UK":0,"USA":1,'N':2}
df["nationality"] =df["nationality"].map(d)
a = {"yes":1,"no":2}
df["Go"] = df["Go"].map(a)
## we will seperate the target columns from the features ones
features = ["age","Experience","Rank","Natinality"]
x = df[features]
y = df["GO"]
## we will create our decision tree now and save it as a image 
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x,y)
data = tree.export_graphviz(dtree,out_file=None,feature_names=features)
graph= pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')
img= pltimg.imread("mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
## we will use our model to make prediction 
# print(dtree([[the number of the features]])


