import numpy as np
from vtkplotter import *
import vtkplotter
from matplotlib.colors import LinearSegmentedColormap

X, Y, Z = np.mgrid[:30, :30, :30]
# scaled distance from the center at (15, 15, 15)
scalar_field = 3 * ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225
scalar_field[:15,:,:] = 0
print('scalar min, max =', np.min(scalar_field), np.max(scalar_field))

color_list = ['red','yellow', 'black']
color_values = [1.5, 2.0, 2.5]

c = [(val, col) for val, col in zip(color_values, color_list)]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('viridis', 12)
color_list = np.array([[0., 0., 0.3, 0.1],
                       [0.9, 0.95, 1., 0.2],
                       [0.2, 1., 0.2, 0.8],
                       [1., 0.2, 0.2, 0.6],
                       [1., 1., 0, 1.],
                       [0., 0.7, 0., 1.]])
newcmp = ListedColormap(color_list)



#look_up_table = vtkplotter.colors.makeLUT([(val, col) for val, col in zip(color_values, color_list)])
vol = Volume(scalar_field) #, c=color_list, alpha=[1.,0.5,0.2])
lego = vol.legosurface(vmin=1.1, vmax=4, cmap=newcmp)
text1 = Text2D('Make a Volume from a numpy object', c='blue')
text2 = Text2D('lego isosurface representation\nvmin=1, vmax=2', c='darkred')

show([vol, lego], N=2)
#show([(vol,text1), (lego,text2)], N=2)

from vtkplotter import *
import vtkplotter
import numpy as np
plist = np.array([[0,0,0], [0,0,1],[0,0,2],[0,0,3],[1,1,-1]])
r = 20
c = np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,1,0]])
points = vtkplotter.shapes.Points(plist, r=r, c=c, alpha=0.7)
show(points)
