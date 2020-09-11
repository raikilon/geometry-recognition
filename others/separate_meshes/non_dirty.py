import bpy

bpy.ops.wm.read_homefile(use_empty=True)

bpy.ops.import_scene.obj(filepath="nondirty.obj", filter_glob="*.obj", axis_forward='-Z', axis_up='Y')

bpy.ops.object.select_all(action='DESELECT')
# bpy.context.active_object.select_set(False)

i = 0
for o in bpy.data.objects:
    if o.type == "MESH":
        bpy.data.objects[o.name].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[o.name]
        # print(o.name)
        bpy.ops.export_scene.obj(filepath="results/non_dirty1/object_{0}.obj".format(i), use_selection=True,
                                 use_animation=False, use_mesh_modifiers=True,
                                 use_edges=True,
                                 use_smooth_groups=False, use_smooth_groups_bitflags=False,
                                 use_normals=True,
                                 use_uvs=False, use_materials=False, use_triangles=True,
                                 use_nurbs=False,
                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                 group_by_material=False, keep_vertex_order=True, global_scale=1,
                                 path_mode='AUTO',
                                 axis_forward='-Z', axis_up='Y')

        bpy.ops.object.select_all(action='DESELECT')

        # bpy.data.objects[o.name].select_set(False)
        # bpy.context.active_object.select_set(False)

        i += 1
