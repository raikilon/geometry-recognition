import bpy

bpy.ops.wm.read_homefile(use_empty=True)

bpy.ops.import_scene.obj(filepath="examples/dirty_1.obj", filter_glob="*.obj", axis_forward='-Z', axis_up='Y')

# Saving the names because the objects list updates when new object are created
names = []
for o in bpy.data.objects:
    if o.type == "MESH":
        names.append(o.name)
print(len(names))
for o in names:
    bpy.data.objects[o].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[o]

    bpy.ops.object.mode_set(mode='EDIT')
    # Remove duplicates vertices to speed up the separation
    # bpy.ops.object.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()

    # split into loose parts
    bpy.ops.mesh.separate(type='LOOSE')

    # object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # bpy.ops.object.select_all(action='DESELECT')

    parts = bpy.context.selected_objects

    for i, part in enumerate(parts):
        part.select_set(True)
        bpy.context.view_layer.objects.active = part
        bpy.ops.export_scene.obj(filepath="results/dirty/object_{0}.obj".format(i), filter_glob="*.obj",
                                 use_selection=True)
        bpy.ops.object.select_all(action='DESELECT')
