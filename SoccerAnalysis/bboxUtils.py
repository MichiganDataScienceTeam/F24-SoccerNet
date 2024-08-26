def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

def get_bbox_height(bbox):
    return bbox[3] - bbox[1]

def get_top_right(bbox):
        x1, y1, x2, y2 = bbox
        return [x2, y1]

def get_top_left(bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1]

def get_bottom_right(bbox):
        x1, y1, x2, y2 = bbox
        return [x2, y2]

def get_bottom_left(bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y2]

def get_left_middle(bbox):
        x1, y1, x2, y2 = bbox
        return [x1, (y1 + y2) // 2]

def get_right_middle(bbox):
        x1, y1, x2, y2 = bbox
        return [x2, (y1 + y2) // 2]

def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def is_bbox_inside(outer_bbox, inner_bbox):
    x1_out, y1_out, x2_out, y2_out = outer_bbox
    x1_in, y1_in, x2_in, y2_in = inner_bbox

    return (x1_out <= x1_in <= x2_out and
            x1_out <= x2_in <= x2_out and
            y1_out <= y1_in <= y2_out and
            y1_out <= y2_in <= y2_out)

def calculate_bbox_size(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def is_bbox_not_none(bbox):
        if bbox is None:
            return False
        return True
