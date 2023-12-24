import math
import sympy
import ezdxf
import shapely
from ezdxf.entities import *
from shapely.geometry import *
import matplotlib.pyplot as plt

drawing = ezdxf.readfile(r"datasets/Drawing1.dxf")
msp = drawing.modelspace()
g = msp.groupby(key=lambda e: e.dxf.layer)


def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def calculate_arc_points(center, radius, start_angle, end_angle, num_points):
    points = []
    angle_diff = end_angle - start_angle

    for i in range(num_points + 1):  # 包括起始点和终止点
        theta = start_angle + i * angle_diff / num_points
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        points.append((x, y))

    return points


def func(start_point, end_point, bulge):
    # 固定值
    direction = -1 if bulge < 0 else 1
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    distance = math.sqrt(dx * dx + dy * dy)
    half_distance = distance / 2.0
    mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
    # 弧度
    theta = 4 * math.atan(abs(bulge))
    half_theta = theta / 2
    c_theta = 2 * math.pi - theta
    half_c_theta = c_theta / 2
    # 计算长度
    if theta == math.pi:
        radius = half_distance
        l = radius
    elif theta < math.pi:
        radius = half_distance / math.sin(half_theta)
        l = math.sqrt(abs(radius ** 2 - half_distance ** 2))
    elif theta > math.pi:
        radius = half_distance / math.sin(half_c_theta)
        l = math.sqrt(abs(radius ** 2 - half_distance ** 2))

    # 计算斜率
    k0 = dy / dx
    k = -1 / k0
    # 构造方程
    # y = k(x - m) + n
    # (x - m) ^ 2 + (y - n) ^ 2 = L ^ 2
    m, n = mid_point
    # 定义变量
    x, y = sympy.symbols('x y')
    eq1 = k * (x - m) - (y - n)
    eq2 = (x - m) ** 2 + (y - n) ** 2 - l ** 2
    centers = sympy.solve([eq1, eq2], (x, y))
    # 计算从起点到圆心的向量和从圆心到终点的向量
    fs = []
    for center in centers:
        vector1 = (center[0] - start_point[0], center[1] - start_point[1])
        vector2 = (end_point[0] - center[0], end_point[1] - center[1])
        cp = cross_product(vector1, vector2)
        f = (cp > 0 and bulge > 0) or (cp < 0 and bulge < 0)
        # if f:
        # 计算起止角度
        start_angle = math.atan2(start_point[1] - center[1], start_point[0] - center[0])
        end_angle = math.atan2(end_point[1] - center[1], end_point[0] - center[0])
        # 计算
        arc_points = calculate_arc_points(center, radius, start_angle, end_angle, 5)
        fs.extend(arc_points)
    return fs


result = []
for i, layer in enumerate(g.keys()):
    entities = g[layer]
    for j, entity in enumerate(entities):
        if isinstance(entity, LWPolyline):
            lw_data = [(x[0], x[1], x[4]) for x in entity.lwpoints]
            # print(lw_data)
            for k in range(len(lw_data) - 1):
                start_point, end_point = lw_data[k][:2], lw_data[k + 1][:2]
                bulge = lw_data[k][2]
                if bulge == 0:
                    ls = LineString([start_point, end_point])
                    print(ls)
                    result.append(ls)
                else:
                    res = func(start_point, end_point, bulge)
                    ls = LineString(res)
                    print(res)
                    result.append(ls)

# Initialize a plot
fig, ax = plt.subplots()

# Parse and plot each LINESTRING
for line_string in result:
    x, y = line_string.xy
    ax.plot(x, y)

# Set axis labels and title
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.set_title('Plot of LINESTRINGs')

# Set the aspect of the plot to be equal
ax.set_aspect('equal', adjustable='box')

# Show the plot
plt.show()
