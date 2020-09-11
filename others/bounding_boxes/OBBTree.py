import pyvista as pv

from pyvista import examples

# tree = examples.download_shark()
tree = pv.read("floor.ply")

boxes = pv.PolyData(tree)
obb = tree.obbTree
for i in range(11):
    obb.GenerateRepresentation(i, boxes)

    p = pv.Plotter(notebook=False)
    p.add_mesh(boxes, show_edges=False, opacity=1, color="green")
    p.add_mesh(tree)
    p.camera_position = [(4.654560897873877, 9.805049092700603, 6.589148021170971),
                         (0.0036373138427734375, 0.15992683172225952, 1.8102989941835403),
                         (-0.19274900699224976, -0.35943994310476324, 0.9130447675795237)]
    p.show()
    #p.screenshot('stairs2_{0}.png'.format(i))
