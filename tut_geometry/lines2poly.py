"""
目标：将LineString重塑成Polygon
1.使用unary_union函数将所有线条合并成一个多几何体对象。
2.使用polygonize函数将所有线条转换为多边形。
3.遍历所有多边形，将每个多边形的顶点作为节点，将相交的边作为连边，构造连通图。
4.使用连通图算法，找到所有连通分量，对每个连通分量进行多边形化。
"""
import shapely
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.ops import polygonize_full

l1 = LineString([(0, 0), (1, 0)])
l2 = LineString([(1, 0), (1, 1)])
l3 = LineString([(1, 1), (0, 1)])
l4 = LineString([(0, 1), (0, 0)])
#
l5 = LineString([(1, 0), (1, 2)])
l6 = LineString([(1, 2), (2, 2)])
l7 = LineString([(2, 2), (2, 0)])
l8 = LineString([(2, 0), (1, 0)])
l = [l1, l2, l3, l4, l5, l6, l7, l8]

# 合并线段数组
lm = linemerge(l)
# 线条取并集
uu = unary_union(lm)
#
pf = polygonize_full(uu)
pfn = [x for x in pf[0].geoms if not x.is_empty]
print(pfn)