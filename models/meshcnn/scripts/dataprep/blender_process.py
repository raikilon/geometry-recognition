import bpy
import os


def clean_corrupted_off(dataset):
    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            for dataset_name in ["train", "test"]:
                for file in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    bpy.ops.wm.read_homefile(use_empty=True)
                    file_name = os.path.join(class_name.path, dataset_name, file)
                    if file_name[-3:] == "off":
                        with open(file_name) as fin:
                            lines = fin.readlines()
                        if len(lines[0]) > 4:
                            lines[0] = "{}\n{}".format(lines[0][:3], lines[0][3:])
                            with open(file_name, 'w') as fout:
                                for line in lines:
                                    fout.write(line)
                        bpy.ops.import_mesh.off(filepath=file_name, filter_glob="*.off", axis_forward='-Z', axis_up='Y')
                        os.remove(file_name)
                        file_name = file_name[:-4] + '.obj'
                        bpy.ops.object.select_all(action='DESELECT')

                        for o in bpy.data.objects:
                            # Check for given object names
                            if o.name == file[:-4]:
                                o.select_set(True)
                                bpy.context.view_layer.objects.active = o
                                imported = o
                        bpy.ops.object.mode_set(mode='EDIT')
                        # remove loose vertices
                        bpy.ops.mesh.delete_loose()
                        # object mode
                        bpy.ops.object.mode_set(mode='OBJECT')

                        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
                        maxDimension = 1.0
                        scaleFactor = maxDimension / max(imported.dimensions)
                        imported.scale = (scaleFactor, scaleFactor, scaleFactor)
                        imported.location = (0, 0, 0)

                        bpy.ops.export_scene.obj(filepath=file_name, check_existing=False, filter_glob="*.obj;*.mtl",
                                                 use_selection=True, use_animation=False, use_mesh_modifiers=True,
                                                 use_edges=True,
                                                 use_smooth_groups=False, use_smooth_groups_bitflags=False,
                                                 use_normals=True,
                                                 use_uvs=False, use_materials=False, use_triangles=True,
                                                 use_nurbs=False,
                                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                                 group_by_material=False, keep_vertex_order=True, global_scale=1,
                                                 path_mode='AUTO',
                                                 axis_forward='-Z', axis_up='Y')


def create_manifold(dataset):
    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            for dataset_name in ["train", "test"]:
                for file in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    bpy.ops.wm.read_homefile(use_empty=True)
                    file_name = os.path.join(class_name.path, dataset_name, file)
                    if file_name[-3:] == "obj":
                        command = "../../../../others/Manifold/build/manifold {} {}".format(file_name, file_name)
                        os.system(command)


def remove_disconnected(dataset):
    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            for dataset_name in ["train", "test"]:
                for file in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    bpy.ops.wm.read_homefile(use_empty=True)
                    file_name = os.path.join(class_name.path, dataset_name, file)
                    if file_name[-3:] == "obj":
                        bpy.ops.wm.read_homefile(use_empty=True)
                        bpy.ops.import_scene.obj(filepath=file_name, axis_forward='-Z', axis_up='Y',
                                                 filter_glob="*.obj",
                                                 use_edges=True,
                                                 use_smooth_groups=True, use_split_objects=False,
                                                 use_split_groups=False,
                                                 use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')

                        ob = bpy.context.selected_objects[0]
                        bpy.context.view_layer.objects.active = ob

                        # Check for disconnected components..
                        # edit mode
                        bpy.ops.object.mode_set(mode='EDIT')
                        # split into loose parts
                        bpy.ops.mesh.separate(type='LOOSE')
                        # object mode
                        bpy.ops.object.mode_set(mode='OBJECT')

                        parts = bpy.context.selected_objects
                        # sort by number of verts (last has most)
                        parts.sort(key=lambda o: len(o.data.vertices))
                        # print
                        # for part in parts:
                        #    print(part.name, len(part.data.vertices))
                        # pop off the last
                        parts.pop()
                        # remove the rest
                        for o in parts:
                            bpy.data.objects.remove(o)

                        ob = bpy.context.selected_objects[0]
                        bpy.context.view_layer.objects.active = ob
                        ob.select_set(state=True)

                        bpy.ops.export_scene.obj(filepath=file_name, check_existing=False, filter_glob="*.obj;*.mtl",
                                                 use_selection=True, use_animation=False, use_mesh_modifiers=True,
                                                 use_edges=True,
                                                 use_smooth_groups=False, use_smooth_groups_bitflags=False,
                                                 use_normals=True,
                                                 use_uvs=False, use_materials=False, use_triangles=True,
                                                 use_nurbs=False,
                                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                                 group_by_material=False, keep_vertex_order=True, global_scale=1,
                                                 path_mode='AUTO',
                                                 axis_forward='-Z', axis_up='Y')


def reduce_genus(dataset, genus):
    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            for dataset_name in ["train", "test"]:
                for file in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    bpy.ops.wm.read_homefile(use_empty=True)
                    file_name = os.path.join(class_name.path, dataset_name, file)
                    if file_name[-3:] == "obj":

                        g = compute_genus(file_name)
                        command = "../../../../others/Manifold/build/manifold {} {}".format(file_name, file_name)
                        while abs(g) > genus:
                            os.system(command)
                            g = compute_genus(file_name)
                            # print(g)


def compute_genus(file_name):
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.obj(filepath=file_name, axis_forward='-Z', axis_up='Y',
                             filter_glob="*.obj",
                             use_edges=True,
                             use_smooth_groups=True, use_split_objects=False,
                             use_split_groups=False,
                             use_groups_as_vgroups=False, use_image_search=True,
                             split_mode='ON')

    ob = bpy.context.selected_objects[0]
    # Compute genus
    me = ob.data
    v = len(me.vertices)
    e = len(me.edges)
    f = len(me.polygons)
    g = 1 - ((v - e + f) / 2)
    return g


def decimate(dataset, n_faces):
    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            for dataset_name in ["train", "test"]:
                for file in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    bpy.ops.wm.read_homefile(use_empty=True)
                    file_name = os.path.join(class_name.path, dataset_name, file)
                    if file_name[-3:] == "obj":

                        bpy.ops.wm.read_homefile(use_empty=True)
                        bpy.ops.import_scene.obj(filepath=file_name, axis_forward='-Z', axis_up='Y',
                                                 filter_glob="*.obj",
                                                 use_edges=True,
                                                 use_smooth_groups=True, use_split_objects=False,
                                                 use_split_groups=False,
                                                 use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')

                        ob = bpy.context.selected_objects[0]
                        bpy.context.view_layer.objects.active = ob

                        mod = ob.modifiers.new(name='Decimate', type='DECIMATE')
                        bpy.context.object.modifiers['Decimate'].use_collapse_triangulate = True
                        nfaces = len(ob.data.polygons)
                        if nfaces < n_faces:
                            # subdivide mesh
                            bpy.context.view_layer.objects.active = ob
                            mod1 = ob.modifiers.new(name='Subsurf', type='SUBSURF')
                            mod1.subdivision_type = 'SIMPLE'
                            bpy.ops.object.modifier_apply(modifier=mod1.name)
                            # now triangulate
                            mod1 = ob.modifiers.new(name='Triangluate', type='TRIANGULATE')
                            bpy.ops.object.modifier_apply(modifier=mod1.name)
                            nfaces = len(ob.data.polygons)
                        ratio = n_faces / float(nfaces)
                        mod.ratio = float('%s' % ('%.6g' % (ratio)))
                        bpy.ops.object.modifier_apply(modifier=mod.name)
                        bpy.ops.object.select_all(action='DESELECT')

                        ob.select_set(state=True)
                        bpy.ops.export_scene.obj(filepath=file_name, check_existing=False, filter_glob="*.obj;*.mtl",
                                                 use_selection=True, use_animation=False, use_mesh_modifiers=True,
                                                 use_edges=True,
                                                 use_smooth_groups=False, use_smooth_groups_bitflags=False,
                                                 use_normals=True,
                                                 use_uvs=False, use_materials=False, use_triangles=True,
                                                 use_nurbs=False,
                                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                                 group_by_material=False, keep_vertex_order=True, global_scale=1,
                                                 path_mode='AUTO',
                                                 axis_forward='-Z', axis_up='Y')


def delete_data(dataset, n_faces):
    i = 0
    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            for dataset_name in ["train", "test"]:
                for file in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    bpy.ops.wm.read_homefile(use_empty=True)
                    file_name = os.path.join(class_name.path, dataset_name, file)
                    bpy.ops.import_scene.obj(filepath=file_name, axis_forward='-Z', axis_up='Y',
                                             filter_glob="*.obj",
                                             use_edges=True,
                                             use_smooth_groups=True, use_split_objects=False,
                                             use_split_groups=False,
                                             use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')
                    ob = bpy.context.selected_objects[0]
                    nfaces = len(ob.data.polygons)
                    if nfaces > n_faces:
                        i += 1
                        os.remove(file_name)
                        print("f:{},n:{}".format(file_name, nfaces))
    print(i)


clean_corrupted_off("../../../../../datasets/ModelNet10")
