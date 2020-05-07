import numpy as np
import vtkplotter as vtkp

save_pth = '/Users/erikpersson/PycharmProjects/AutoDrone/plots/objs_ver2'
gt_pos = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_objs/ground_truth.npy')
obstac = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_objs/obstacles.npy')
objs = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_objs/objects.npy')
visited = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/local3_epoch_2270_objs/visited.npy')
ceiling_level = -1.2

print("Shape of gt_pos {}".format(gt_pos.shape))
print("Shape of obstac {}".format(obstac.shape))
print("Shape of objs {}".format(objs.shape))
print("Shape of visited {}".format(visited.shape))

gt_color = [0, 0, 255, 150]
obj_color = [255, 255, 0, 255]
obstac_color = [255, 51, 51, 140]
visited_color = [0, 255, 0, 70]

obstac_list = []
for o in obstac:
    if o[0] < 25 and o[1] < 10 and o[2] > ceiling_level:
        obstac_list.append(o)
obstac = np.array(obstac_list)

visited_list = []
for v in visited:
    if v[0] < 25 and v[1] < 10:
        visited_list.append(v)
visited = np.array(visited_list)

plist1 = np.concatenate((visited, gt_pos, objs), axis=0)
c1 = []
for _ in range(visited.shape[0]):
    c1.append(visited_color)
for _ in range(gt_pos.shape[0]):
    c1.append(gt_color)
for _ in range(objs.shape[0]):
    c1.append(obj_color)

c2 = c1.copy()
for _ in range(obstac.shape[0]):
    # c2.insert(0, obstac_color)
    c2.append(obstac_color)

# plist2 = np.concatenate((obstac, plist1), axis=0)
plist2 = np.concatenate((plist1, obstac), axis=0)


points1 = vtkp.shapes.Points(plist1, r=20, c=c1)
points2 = vtkp.shapes.Points(plist2, r=20, c=c2)

axes_dict = dict(
    xyGrid=False,
    yzGrid=False,
    showTicks=True,
    xLabelSize=0,
    yLabelSize=0,
    zLabelSize=0,
    xtitle=' ',
    ytitle=' ',
    ztitle=' ',
    axesLineWidth=3
)
vp = vtkp.Plotter(axes=axes_dict, interactive=False, size=(10000,10000))
vp += points1
vp.show(azimuth=70, elevation=165,roll=-90)
vtkp.screenshot(save_pth+'.png')

vp = vtkp.Plotter(axes=axes_dict, interactive=False, size=(10000,10000))
vp += points2
vp.show(azimuth=70, elevation=165,roll=-90)
vtkp.screenshot(save_pth+'_with_obstac.png')
