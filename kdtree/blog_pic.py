#encoding=utf8
import matplotlib.pyplot as plt
import numpy as np

point=[(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)]
x = [p[0] for p in point]
y = [p[1] for p in point]


figure = plt.figure()
ax1 = figure.add_subplot(111)
ax1.set_title("point position")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim((0,10))
plt.ylim((0,10))
ax1.scatter(x,y,s=10,c='r')
ax1.plot([7,7],[0,10],linewidth=1, color='b')
ax1.plot([0,7],[4,4],linewidth=1, color='b')
ax1.plot([7,10],[6,6],linewidth=1, color='b')
ax1.plot([4,4],[4,10],linewidth=1, color='b')
ax1.plot([2,2],[0,4],linewidth=1, color='b')
ax1.plot([8,8],[0,6],linewidth=1, color='b')
#第一次判断
# circle = plt.Circle((2.1,3.1),0.1414,color='b',fill=False,clip_on=False)
# ax1.scatter(2.1,3.1,s=10,color='r',marker='s')
# ax1.add_artist(circle)

#第二次判断
# circle = plt.Circle((2,4.5),3.041,color='b',fill=False,clip_on=False)
# ax1.scatter(2,4.5,s=10,color='r',marker='s')
# ax1.add_artist(circle)

#第二次判断
circle = plt.Circle((2,4.5),1.5,color='b',fill=False,clip_on=False)
ax1.scatter(2,4.5,s=10,color='r',marker='s')
ax1.add_artist(circle)
plt.show()

