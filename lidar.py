import bpy
import bmesh
import numpy as np
import heapq
import csv
import cv2
import math


def remove_specified_values(arr, value):
    while value in arr:
        arr.remove(value)


def detect_second(arr):
    max1 = max(arr)
    second = 0
    for i in range(len(arr)):
        if (arr[i] != max1) & (arr[i] > second):
            second = arr[i]
    return (second)


def img_size(img):
    width, height = img.size
    return width, height


def make_depthmap(img):
    pixels_list = list(img.pixels[:])
    depth_map = pixels_list[0::4]
    return depth_map  # list


def normalized_map(arr, img):
    net_max = detect_second(arr)
    min1 = min(arr)
    max1 = max(arr)
    pixel = np.ndarray(len(arr))
    for i in range(len(arr)):
        if arr[i] != max1:
            pixel[i] = 255 - ((arr[i] - min1) / (net_max - min1)) * 255
        else:
            pixel[i] = 0
    W, H = img_size(img)
    np_pixel_resize = np.array(pixel)
    np_pixel_resize.resize(H, W)
    normalized_depthmap = np.flipud(np_pixel_resize)
    return normalized_depthmap


def fov_area(fov, ray_center):  # fovは画角の角度。画角が正方形だから引数が１つ,ray_centerの大きさは焦点距離になる。
    cog_vertices: List[List[Any]] = []
    ray_center_norm = np.linalg.norm(ray_center)  # レイの中心ベクトルのノルム
    cog_size = ray_center_norm
    ray_center_x = ray_center[0] * cog_size / ray_center_norm  # ray_centerの大きさを重心ベクトルと同じ大きさにする。
    ray_center_y = ray_center[1] * cog_size / ray_center_norm
    ray_center_z = ray_center[2] * cog_size / ray_center_norm
    cog_mid_z0 = []
    if ray_center_x != 0:
        for i in fov:
            a = 1 + (ray_center_y / ray_center_x) ** 2
            b = - ((cog_size ** 2 - ray_center_z ** 2) * ray_center_y) / (ray_center_x ** 2)
            c = ((cog_size ** 2 - ray_center_z ** 2) ** 2 / ray_center_x ** 2) - (
                    cog_size / math.cos(math.radians(i / 2))) ** 2 + ray_center_z ** 2
            t = answer2eq(a, b, c)
            cog_mid_z0.append(
                [(cog_size ** 2 - ray_center_z ** 2 - ray_center_y * t[1]) / ray_center_x, t[1], ray_center_z])
    elif ray_center_x == 0:
        for i in fov:
            b = - math.sqrt(ray_center_y ** 2 + ray_center_z ** 2) * math.tan(math.radians(i / 2))
            cog_mid_z0.append([b, ray_center_y, ray_center_z])

            # cog_mid_z0は画角の左右の辺の中点を表す。

    if ray_center_x != 0:
        theta = math.atan2(ray_center_y, ray_center_x)
        psi = math.atan2(ray_center_z, math.sqrt(ray_center_x ** 2 + ray_center_y ** 2))

    elif (ray_center_x == 0) & (ray_center_y != 0):
        theta = math.pi / 2
        psi = math.atan(ray_center_z / ray_center_y)

    elif (ray_center_x == 0) & (ray_center_y == 0):
        theta = math.pi / 2
        # 0なのか90なのか
        if ray_center_z > 0:
            psi = math.pi / 2
        elif ray_center_z < 0:
            psi = - math.pi / 2
    cog_mid_z0_trans_horizon = (L_x(-cog_size) @ R_y(psi) @ R_z(-theta) @ np.append(cog_mid_z0[0], 1))[1]
    cog_mid_z0_trans_vertical = (L_x(-cog_size) @ R_y(psi) @ R_z(-theta) @ np.append(cog_mid_z0[1], 1))[1]

    # cog_mid_z0_transは座標変換された画角の上下の辺の中点を表す。
    cog_trans = np.array([[0, cog_mid_z0_trans_horizon, cog_mid_z0_trans_vertical],
                          [0, cog_mid_z0_trans_horizon, -cog_mid_z0_trans_vertical],
                          [0, -cog_mid_z0_trans_horizon, -cog_mid_z0_trans_vertical],
                          [0, -cog_mid_z0_trans_horizon, cog_mid_z0_trans_vertical]])
    for j in cog_trans:
        cog_vertice_item = R_z(theta) @ R_y(-psi) @ L_x(cog_size) @ np.append(j, 1)  # 逆変換してる。
        cog_vertice_coor = cog_vertice_item.tolist()
        cog_vertices.append([cog_vertice_coor[0], cog_vertice_coor[1], cog_vertice_coor[2]])
    return cog_vertices


def rays_func(fov, ray_center, res):  # rays_funcはlidarから出てくるレイベクトルを生成している。resは解像度
    v4 = fov_area(fov, ray_center)  # ４つの画角の頂点ベクトル
    side_vectors = []  # 基底ベクトルを生成
    side_vectors_unit = []  # 単位基底ベクトルを生成
    dot = []
    for i in range(2):
        side_vectors.append(np.array(v4[i + 1]) - np.array(v4[i]))
    x = side_vectors[0] / res[1]
    y = side_vectors[1] / res[0]
    for i in range(res[1]):
        for j in range(res[0]):
            dot.append((x * i + y * j) + x / 2 + y / 2 + np.array(v4[0]))
    return np.array(dot)


def cos(a):
    return math.cos(a)


def sin(a):
    return math.sin(a)


def R_y(a):
    return np.array([[cos(a), 0, sin(a), 0], [0, 1, 0, 0], [-sin(a), 0, cos(a), 0], [0, 0, 0, 1]])


def R_z(a):
    return np.array([[cos(a), -sin(a), 0, 0], [sin(a), cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def L_x(a):
    return np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def answer2eq(a, b, c):
    ans = [(- b - math.sqrt(b ** 2 - a * c)) / a, (- b + math.sqrt(b ** 2 - a * c)) / a]
    return ans


def adjust_verts(ray, verts_dist):
    verts = []
    max1 = max(verts_dist)
    count = 0
    for i in range(len(ray)):
        if verts_dist[i] != max1:
            verts.append((ray[i] * verts_dist[i]).tolist())
        else:
            count += 1
    verts = np.array(verts)
    print(count)
    return verts


def Rot_x(radians):
    return np.array([[1, 0, 0], [0, cos(radians), -sin(radians)], [0, sin(radians), cos(radians)]])


def Rot_y(radians):
    return np.array([[cos(radians), 0, sin(radians)], [0, 1, 0], [-sin(radians), 0, cos(radians)]])


def Rot_z(radians):
    return np.array([[cos(radians), -sin(radians), 0], [sin(radians), cos(radians), 0], [0, 0, 1]])


def verts_loc(camera_name, adj_verts):
    camera = bpy.context.scene.objects['Camera2']
    x = camera.rotation_euler.x
    y = camera.rotation_euler.y
    z = camera.rotation_euler.z
    loc = np.array(camera.location)
    Rot_mat = Rot_z(z) @ Rot_y(y) @ Rot_x(x)
    for i in range(len(adj_verts)):
        adj_verts[i] = Rot_mat @ (adj_verts[i]) + loc
    return adj_verts


def verts_distance(img):
    depthmap = make_depthmap(img)
    W, H = img_size(img)
    np_dist = np.array(depthmap)
    np_dist.resize(H, W)
    np_dist_flip = np.flipud(np_dist)  # jyouge hanntenn
    verts_dist = np.ravel(np_dist_flip)  # to 1jigenn
    return verts_dist


def verts_location(camera_name, hor_deg, ver_deg, res_x, res_y, img):
    ray = rays_func([hor_deg, ver_deg], [0, 0, -1], [res_x, res_y])
    verts_dist = verts_distance(img)
    adj_verts = adjust_verts(ray, verts_dist)
    verts = verts_loc(camera_name, adj_verts)
    return verts


def new_object(verts, mesh_name, object_name, collection_name):
    mymesh = bpy.data.meshes.new(mesh_name)
    mymesh.from_pydata(verts, [], [])  # 作成部分
    mymesh.update()
    new_object = bpy.data.objects.new(object_name, mymesh)
    new_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(new_collection)
    new_collection.objects.link(new_object)


def to_xyz1(l):
    l_str = list(map(str, l))
    l_join = ' '.join(l_str)
    return l_join


def to_xyz2(l):
    verts_str = list(map(to_xyz1, l))
    verts_join = '\n'.join(verts_str)
    return verts_join


def make_verts(camera_name, hor_deg, ver_deg, res_x, res_y, img, mesh_name, object_name, collection_name):
    verts = verts_location(camera_name, hor_deg, ver_deg, res_x, res_y, img)
    new_object(verts, mesh_name, object_name, collection_name)
    return verts


def write_depthmap_png(img, path_depthmap_png):
    depthmap = make_depthmap(img)
    normalized_depthmap = normalized_map(depthmap, img)
    cv2.imwrite(path_depthmap_png, normalized_depthmap)


def write_verts_xyz(verts, path_depthmap_xyz):
    verts_xyz = to_xyz2(verts)
    with open(path_depthmap_xyz, mode='w') as f:
        f.write(verts_xyz)


def get_verts(object_name):
    object = bpy.data.objects[object_name]
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')
    object_mesh = bpy.context.object.data
    object_emesh = bmesh.from_edit_mesh(object_mesh)
    object_verts = np.array(list((map(lambda x: np.array(x.co), object_emesh.verts))))
    return object_verts


if __name__ == '__main__':
    bpy.ops.render.render()
    img = bpy.data.images['Viewer Node']
    camera_name = 'Camera2'
    hor_deg = 54.1
    ver_deg = 41.91
    res_x = 640
    res_y = 480
    mesh_name = 'newmesh'
    object_name = 'MyObject'
    collection_name = 'new_collection'
    path_depthmap_png = 'C:/Users/masam/depth_map.png'
    path_camera_xyz = 'C:/Users/masam/camera.xyz'
    path_monkey_xyz = 'C:/Users/masam/monkey.xyz'

    monkey_verts = get_verts('monkey')
    write_depthmap_png(img, path_depthmap_png)
    verts = make_verts(camera_name, hor_deg, ver_deg, res_x, res_y, img, mesh_name, object_name, collection_name)
    write_verts_xyz(monkey_verts, path_monkey_xyz)
    write_verts_xyz(verts, path_camera_xyz)

"""

#頂点定義
verts = []
verts.append( [0.2,1.5,1.5] )
verts.append( [1.5,1.5,1.5] )
verts.append( [1.5,0.2,0.2] )

#面を、それを構成する頂点番号で定義
faces =[]
faces.append( [0,1,2] )

#頂点と頂点番号からメッシュを作成
mymesh = bpy.data.meshes.new("mymesh")
mymesh.from_pydata(verts,[],faces) #作成部分
mymesh.update()

#オブジェクトを作成し、メッシュを登録
new_object = bpy.data.objects.new("My_Object", mymesh)

#オブジェクトのマテリアルを作成
#mat = bpy.data.materials.new("Mat1")
#mat.diffuse_color = (0.3,0.7,0.5)
#new_object.active_material = mat

#オブジェクトを登録
new_collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(new_collection)
new_collection.objects.link(new_object)


"""






