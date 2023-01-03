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

lm = linemerge(l)
uu = unary_union(lm)
pf = polygonize_full(uu)
pfn = [x for x in pf[0].geoms if not x.is_empty]
print(pfn)
