import numpy as np

def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]

class BoxesConnector(object):
    def __init__(self,rects,imageW,max_dist=None,overlap_threshold=None):
        print("max")