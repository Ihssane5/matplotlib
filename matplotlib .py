#!/usr/bin/env python
# coding: utf-8

# In[2]:


## we will import our library 
import matplotlib 
print("the version of our matpltlib is :",matplotlib.__version__)


# In[10]:


import matplotlib.pyplot as plt 
import numpy as np 
## we will plot our first graph together dear 
x_points=np.array([0,6])
y_points = np.array([0,250])

plt.plot(x_points,y_points) #" the plot method by default draw line 
plt.show()


# In[12]:


## if we only need the markers 
import numpy as np 
import matplotlib.pyplot as plt 
x_points = np.array([5,8])
y_points= np.array([6,7])
plt.plot(x_points,y_points,"o")


# In[46]:


###### draw  a line 
import matplotlib.pyplot as plt 
import numpy as np 
x_points = np.array([1,2,6,8]) ## make each point circled 
y_points = np.array([3,8,1,3])
plt.plot(x_points,"*-r",marker = "*",ms=20, mec="r",mfc="hotpink",
        linestyle= "dashed",linewidth=6) #solid line and "*:r" dashed line,## and then the marker side 
                                       # marker edge color , maker facecolor #"inside the edge"
print("that's your plot")
## we can create a label to the plot 
plt.xlabel("the days")
plt.ylabel("your progress")


# and we  can also  have a title 
plt.title("your steps ")
plt.show()


# In[61]:


# we can try to use the font properties 
import matplotlib.pyplot as plt 
import numpy as np 
x = np.array([3,5,7,9])
y = np.array([6,8,3,3])

font1 = {"name":"ihssane","color":"red","size":18}
font2={"name":"ihssane","color":"pink","size":18}
plt.title("sport watch data",fontdict=font1,loc="left") ## center by default
plt.xlabel("average pulse",fontdict=font2)
plt.ylabel("calories burned",fontdict=font2)
plt.plot(x,y)
## we can a grid to our plot by default it's both axis 
plt.grid(axis = "x", color="red",linestyle ="dotted", linewidth=8)
plt.show()


# In[82]:


## subplot method help us to display two plots in one figure 
##plot1
import matplotlib.pyplot as plt 
import numpy as np
plt.suptitle("here the suptitle")
x_points =np.array([3,6,8,9])
y_points=np.array([5,8,7,2])
plt.subplot(2,1,2)      ## the figures are displayed in two rows one columns and the index two  
plt.plot(x_points,y_points)
plt.title("arbitary one")
plt.show()
x_points = np.array([4,7,8])
y_points = np.array([5,7,9])
plt.subplot(2,1,1)
plt.plot(x_points,y_points)
plt.title("just another essay")
## we can ad a super title 

plt.show()


# In[100]:


## we can try the matplotlib scatter  now 
import matplotlib.pyplot as plt 
import numpy as np 
x_values = np.array([5,7,9,7,2,1])
y_values = np.array([4,6,8,9,4,8])
plt.scatter(x_values,y_values,c="hotpink")
x_points = np.array([8,9,5,3,7,4])
y_points = np.array([2,6,8,2,2,4])
plt.scatter(x_points,y_points,c="#88c999")
plt.title("that's my plot")
#well we can do more interesting staffs here 
colors = np.array(["red","brown","hotpink","grey","green","cyan"])
plt.scatter(x_points,y_points,color = colors,cmap="nipy_spectral",s=500,alpha=0.5) ## color bar # the alpha parameter play on the transparence of our points 
plt.colorbar() ## now we have our color bar with our plot
plt.show()


# In[114]:


## we will learn to build bars 
x =np.array(["usa","morocco","german"])
y = np.array([55,77,88])
plt.barh( x,y, height=0.4, color = "hotpink" ) # if we want to plot horizontally height or width
plt.show()


# In[120]:


## then we will move to the histograms 
## we will generate a data 250 values concentrate around 170 and with an standard deviation of 10
x = np.random.normal(170,10,250)
## then we print the values 

print("our histograme is ready ")

plt.xlabel('values')
plt.ylabel("the number")
plt.title("the histograme")
plt.hist(x)
plt.show()
## matplotlib pie chart 


# In[138]:


## matplotlib pie chart 
import numpy as np
import matplotlib.pyplot as plt
x = np.array([18,15,10,4])
labels =np.array(["me","ikrame","ines","mery"])
explode=[0.2,0.5,0,0]
color=np.array(["hotpink","red","green","yellow"])
plt.pie(x,labels=labels,startangle=90,explode=explode,shadow=True,colors=color) ## the angle is by default equal to0
plt.title('our pie  chart')
plt.legend(title="me and my sisters")
plt.show()
print('our  part are equal to',x/sum(x))

