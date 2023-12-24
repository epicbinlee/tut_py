import torch
import numpy as np
import logging
import ezdxf
from ezdxf.colors import RGB
from ezdxf.groupby import groupby
from ezdxf.entities import Point, Line, LWPolyline, Circle, Arc, Text, MText, Insert, Polyline, Spline
from shapely.geometry import Point, LineString, GeometryCollection
from math import cos, sin, radians

"""
说明：
本文件用于解析dxf文件，测试。
"""


def parse_lw_polyline(entity):
    points = [(x[0], x[1]) for x in entity.lwpoints]
    if entity.closed:
        points.append(points[0])
    return LineString(points)


def parse_line(entity):
    st = list(entity.dxf.start)[0:2]
    ed = list(entity.dxf.end)[0:2]
    return LineString([st, ed])


def parse_text(entity):
    point = Point(list(entity.dxf.insert)[0:2])
    text = entity.dxf.text
    return text, point


def parse_circle(entity):
    center = entity.dxf.center
    radius = entity.dxf.radius

    angle_step = 1

    points = []
    for i in range(int(360 / angle_step)):
        angle = radians(i * angle_step)
        x = center.x + radius * cos(angle)
        y = center.y + radius * sin(angle)
        points.append((x, y))

    # 创建LineString对象
    line_string = LineString(points)
    return line_string


def parse_spline(entity):
    control_points = np.array([x[0:2] for x in list(entity.control_points)])
    fit_points = np.array([x[0:2] for x in list(entity.fit_points)])
    knots = list(entity.knots)[1:]
    weights = list(entity.weights)
    pass


class CadParser(object):
    def __init__(self, dxf_name, dxf_fp):
        self.dxf_name = dxf_name
        self.dxf_fp = dxf_fp
        self.process()

    def process(self):
        drawing = ezdxf.readfile(self.dxf_fp)
        msp = drawing.modelspace()
        grp = groupby(entities=msp, dxfattrib='layer')
        for layer, entities in grp.items():
            logging.info(f'正在读取: {layer}')
            for entity in entities:
                # https://ezdxf.readthedocs.io/en/stable/colors.html#module-ezdxf.colors
                color_index = entity.dxf.color
                if color_index == 0:
                    color_index = 1
                elif color_index == 256:
                    color_index = drawing.layers.get(layer).color
                color = ezdxf.colors.aci2rgb(color_index).to_hex()
                # print(f'{entity.dxftype()}: {color}')

                logging.info(f'正在解析: {entity.dxftype()}')
                geometry = None
                if isinstance(entity, Point):
                    self.parse_point(entity)
                elif isinstance(entity, Line):
                    geometry = parse_line(entity)
                elif isinstance(entity, LWPolyline):
                    geometry = parse_lw_polyline(entity)
                elif isinstance(entity, Circle):
                    parse_circle(entity)
                elif isinstance(entity, Arc):
                    self.parse_arc(entity)
                elif isinstance(entity, Insert):
                    self.parse_insert(entity)
                elif isinstance(entity, Polyline):
                    self.parse_polyline(entity)
                elif isinstance(entity, Spline):
                    parse_spline(entity)
                elif isinstance(entity, Text) and isinstance(entity, MText):
                    text, geometry = parse_text(entity)

                if geometry is not None:
                    print(f"type: {entity.dxftype()}, geometry: {geometry.wkt}, color: {color}")

    def parse_point(self, entity):
        pass

    def parse_arc(self, entity):
        pass

    def parse_insert(self, entity):
        pass

    def parse_polyline(self, entity):
        pass


if __name__ == '__main__':
    # dxf_name = r'Floor Plan Sample.dxf'
    # dxf_fp = r'D:\develop\python\tut_py\tut_parser\datasets\Floor Plan Sample.dxf'

    dxf_name = r'spline2.dxf'
    dxf_fp = r'D:\develop\python\tut_py\tut_ezdxf\datasets\spline2.dxf'
    CadParser(dxf_name, dxf_fp)
