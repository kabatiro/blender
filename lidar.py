import bpy
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


bpy.ops.render.render()

img = bpy.data.images['Viewer Node']
W, H = img.size
print('Hello')
a = np.array(img.pixels[:])
list = list(img.pixels[:])
list_R = list[0::4]
# list_R is depth_map
a.resize(H, W * 4)


# print(len(list))
# print(len(list_R))
# print(max(list_R))
# detect_second(list_R)

def normalized(arr):
    net_max = detect_second(arr)
    min1 = min(arr)
    max1 = max(arr)
    pixel = np.ndarray(len(arr))
    for i in range(len(arr)):
        if arr[i] != max1:
            pixel[i] = 255 - ((arr[i] - min1) / (net_max - min1)) * 255
        else:
            pixel[i] = 0
    return pixel


pixel = normalized(list_R)
np_pixel_resize = np.array(pixel)
np_pixel_resize.resize(H, W)
# print(max(pixel))
# print(min(pixel))
# print(pixel)
np_dist = np.array(list_R)
np_dist.resize(H, W)
np_dist_flip = np.flipud(np_dist)
np_dist_ravel = np.ravel(np_dist_flip)

cv2.imwrite('C:/Users/masam/depth_map.png', cv2.flip(np_pixel_resize, 0))


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
    return dot


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


dot = np.array(rays_func([54.1, 41.91], [0, 0, -1], [640, 480]))

print('len is' + str(len(dot)))

print(dot.shape)
verts = []
max1 = max(np_dist_ravel)
count = 0
for i in range(len(dot)):
    if np_dist_ravel[i] != max1:
        verts.append((dot[i] * np_dist_ravel[i]).tolist())
    else:
        count += 1
verts = np.array(verts)
print('destroy')
print(count)


def Rot_x(radians):
    return np.array([[1, 0, 0], [0, cos(radians), -sin(radians)], [0, sin(radians), cos(radians)]])


def Rot_y(radians):
    return np.array([[cos(radians), 0, sin(radians)], [0, 1, 0], [-sin(radians), 0, cos(radians)]])


def Rot_z(radians):
    return np.array([[cos(radians), -sin(radians), 0], [sin(radians), cos(radians), 0], [0, 0, 1]])


camera = bpy.context.scene.objects['Camera2']
monkey = bpy.context.scene.objects['monkey']
print(monkey.location)
x = camera.rotation_euler.x
y = camera.rotation_euler.y
z = camera.rotation_euler.z
loc = np.array(camera.location)
Rot_mat = Rot_z(z) @ Rot_y(y) @ Rot_x(x)

print(loc)
for i in range(len(verts)):
    verts[i] = Rot_mat @ (verts[i]) + loc
# verts = dot + np.array(camera2.location)

mymesh = bpy.data.meshes.new("mymesh")
mymesh.from_pydata(verts, [], [])  # 作成部分
mymesh.update()
new_object = bpy.data.objects.new("My_Object", mymesh)

new_collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(new_collection)
new_collection.objects.link(new_object)

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



