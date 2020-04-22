import numpy as np
import vtkplotter as vtkp


gt_pos = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/ground_truth.npy')
obstac = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/obstacles.npy')
objs = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/objects.npy')
visited = np.load('/Users/erikpersson/PycharmProjects/AutoDrone/verification/test/visited.npy')

print("Shape of gt_pos {}".format(gt_pos.shape))
print("Shape of obstac {}".format(obstac.shape))
print("Shape of objs {}".format(objs.shape))
print("Shape of visited {}".format(visited.shape))

gt_color = [0, 0, 255, 150]
obj_color = [255, 128, 0, 255]
visited_color = [0, 255, 0, 80]

plist = np.concatenate((visited, gt_pos, objs), axis=0)
c = []
for _ in range(visited.shape[0]):
    c.append(visited_color)
for _ in range(gt_pos.shape[0]):
    c.append(gt_color)
for _ in range(objs.shape[0]):
    c.append(obj_color)

points = vtkp.shapes.Points(plist, r=20, c=c)
vtkp.show(points, axes=8)
