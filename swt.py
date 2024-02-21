import os
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import ConvexHull
import math

Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])


def get_grayscale(im) :

    gamma_coeff= 2.2
    im = im.astype(np.float32) / 255.
    im = im ** (1. / gamma_coeff)
    im = np.mean(im, axis=2)
    return np.expand_dims(im, axis=2)
def get_gradients(im) :
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)
def apply_swt(edges, gradients, dark_on_bright = True) :

    swt = np.squeeze(np.ones_like(edges)) * np.Infinity

    # For
    # pixel, let's obtain the normal direction of its gradient.
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2)
    norms[norms == 0] = 1
    inv_norms = 1. / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)
    # We keep track of all the rays found in the image.
    rays = []
    # Find a pixel that lies on an edge.
    height, width = edges.shape
    for y in range(height):
        for x in range(width):
            # Edges are either 0. or 1.
            if edges[y, x] < .5:
                continue
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt, dark_on_bright=dark_on_bright)
            if ray:
                rays.append(ray)
    # Multiple rays may cross the same pixel and each pixel has the smallest
    # stroke width of those.
    # A problem are corners like the edge of an L. Here, two rays will be found,
    # both of which are significantly longer than the actual width of each
    # individual stroke. To mitigate, we will visit each pixel on each ray and
    # take the median stroke length over all pixels on the ray.
    #print(swt)
    average=0
    lenstw=np.array([len(ray)for ray in rays])
    mean=np.mean(lenstw)
    lenP=[]
    for i in lenstw:
        if (i>mean):
            lenP.append(i)
    mean1=np.mean(np.array(lenP))
    max=mean1+mean
    for ray in rays:
        median = np.median([swt[p.y, p.x] for p in ray])
        if (len(ray)<max):
            for p in ray:
                swt[p.y, p.x] = min(median, swt[p.y, p.x])
        else :
            for p in ray:
               swt[p.y, p.x] =0
    swt[swt == np.Infinity] = 0

    #print("///////////////////////////////////",median)
    return swt
def swt_process_pixel(pos, edges, directions, out, dark_on_bright = True):
    """
    Obtains the stroke width starting from the specified position.
    :param pos: The starting point
    :param edges: The edges.
    :param directions: The normalized gradients
    :param out: The output image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    """
    # Keep track of the image dimensions for boundary tests.
    height, width = edges.shape[0:2]

    # The direction in which we travel the gradient depends on the type of text
    # we want to find. For dark text on light background, follow the opposite
    # direction (into the dark are); for light text on dark background, follow
    # the gradient as is.
    gradient_direction = -1 if dark_on_bright else 1

    # Starting from the current pixel we will shoot a ray into the direction
    # of the pixel's gradient and keep track of all pixels in that direction
    # that still lie on an edge.
    ray = [pos]

    # Obtain the direction to step into
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]

    # Since some pixels have no gradient, normalization of the gradient
    # is a division by zero for them, resulting in NaN. These values
    # should not bother us since we explicitly tested for an edge before.
    assert not (np.isnan(dir_x) or np.isnan(dir_y))

    # Traverse the pixels along the direction.
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        # Advance to the next pixel on the line.
        steps_taken += 1
        cur_x = int(np.floor(pos.x + gradient_direction * dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + gradient_direction * dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        # If we reach the edge of the image without crossing a stroke edge,
        # we discard the result.
        if not ((0 <= cur_x < width) and (0 <= cur_y < height)):
            return None
        # The point is either on the line or the end of it, so we register it.
        ray.append(cur_pos)
        # If that pixel is not an edge, we are still on the line and
        # need to continue scanning.
        if edges[cur_y, cur_x] < .5:  # TODO: Test for image boundaries here
            continue
        # If this edge is pointed in a direction approximately opposite of the
        # one we started in, it is approximately parallel. This means we
        # just found the other side of the stroke.
        # The original paper suggests the gradients need to be opposite +/- PI/6.
        # Since the dot product is the cosine of the enclosed angle and
        # cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
        # this threshold.
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y
        if dot_product >= -0.866:
            return None
        # Paint each of the pixels on the ray with their determined stroke width
        stroke_width = np.sqrt((cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y))
        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x])
        return ray

    # noinspection PyUnreachableCode
    assert False, 'This code cannot be reached.'
def connected_components(swt, threshold= 1.) :
    """
    Applies Connected Components labeling to the transformed image using a flood-fill algorithm.
    :param swt: The Stroke Width transformed image.
    :param threshold: The Stroke Width ratio below which two strokes are considered the same.
    :return: The map of labels.
    """
    height, width = swt.shape[0:2]
    labels = np.zeros_like(swt, dtype=np.uint32)
    next_label = 0
    components = []  # List[Component]
    for y in range(height):
        for x in range(width):
            stroke_width = swt[y, x]
            if (stroke_width <= 0) or (labels[y, x] > 0):
                continue
            next_label += 1
            neighbor_labels = [Stroke(x=x, y=y, width=stroke_width)]
            component = []
            while len(neighbor_labels) > 0:
                neighbor = neighbor_labels.pop()
                npos, stroke_width = Position(x=neighbor.x, y=neighbor.y), neighbor.width
                if not ((0 <= npos.x < width) and (0 <= npos.y < height)):
                    continue
                # If the current pixel was already labeled, skip it.
                n_label = labels[npos.y, npos.x]
                if n_label > 0:
                    continue
                # We associate pixels based on their stroke width. If there is no stroke, skip the pixel.
                n_stroke_width = swt[npos.y, npos.x]
                if n_stroke_width <= 0:
                    continue
                # We consider this point only if it is within the acceptable threshold and in the initial test
                # (i.e. when visiting a new stroke), the ratio is 1.
                # If we succeed, we can label this pixel as belonging to the same group. This allows for
                # varying stroke widths due to e.g. perspective distortion or elaborate fonts.
                if (stroke_width / n_stroke_width >= threshold) or (n_stroke_width / stroke_width >= threshold):
                    continue
                labels[npos.y, npos.x] = next_label
                component.append(npos)
                # From here, we're going to expand the new neighbors.
                neighbors = {Stroke(x=npos.x - 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y + 1, width=n_stroke_width)}
                neighbor_labels.extend(neighbors)
            if len(component) > 0:
                components.append(component)
    return labels, components

def minimum_area_bounding_box(points) :
    hull = ConvexHull(points)
    for i in range(len(hull.vertices) - 1):
        # Select two vertex pairs and obtain their orientation to the X axis.
        a = points[hull.vertices[i]]
        b = points[hull.vertices[i + 1]]
        # TODO: Find orientation. Note that sine = abs(cross product) and cos = dot product of two vectors.
        # print(a, b)
    return points
def discard_non_text(swt, components) :
  
    valid_components = []  # type: List[Component]
    labels = np.zeros_like(swt).astype(np.uint8)
    for component in components:
        # If the variance of the stroke widths in the component is more than
        # half the average of the stroke widths of that component, it is considered invalid.
        average_stroke = np.mean([swt[p.y, p.x] for p in component])
        variance = np.var([swt[p.y, p.x] for p in component])

        if (variance > average_stroke):

            continue
        else:
            for p in component:
                labels[p.y,p.x]=255
            valid_components.append(component)
        # Natural scenes may create very long, yet narrow components. We prune
        # these based on their aspect ratio.
        points = np.array([[p.x, p.y] for p in component], dtype=np.uint32)
        #minimum_area_bounding_box(points)
        #print(variance)
    return labels, valid_components
def variance_discard_non_text( img : Image, components: List[Component]) -> Tuple[Image, List[Component]]:
    valid_components = []  # type: List[Component]
    not_valid_components=[]  # type: List[Component]
    labels = np.zeros_like(img).astype(np.uint8)
    for component in components:
        variance = math.sqrt(np.var([img[p.y, p.x] for p in component]))
        if (variance> 50):
            #print("var :: ",variance)
            not_valid_components.append(component)
            continue
        else:
            for p in component:
                labels[p.y,p.x]=255
            valid_components.append(component)
    return labels, valid_components
def aspect_ration_discard_non_text( img:Image, components: List[Component]) -> Tuple[Image, List[Component]]:
    valid_components = []  # type: List[Component]
    labels = np.zeros_like(img).astype(np.uint8)
    for component in components:
        compon= np.array([[p.y, p.x] for p in component])
        box= cv2.boundingRect(compon)
        if (len(compon)>3):
            hull=None
            try:
                hull = ConvexHull(compon)
            except :
                hull=None
            if (hull):
                area = hull.area

                aspect1=box[2]
                aspect2=box[3]
                areaBox=aspect2*aspect1
                if (0.25<(aspect1/aspect2)<4):
                    if((10*areaBox)>(area)>(0.1*areaBox)):

                            for p in component:
                                labels[p.y, p.x] = 255
                            valid_components.append(component)
    return labels, valid_components
def filter( img:Image, components: List[Component],ration=4,max=400,min=4) -> Tuple[Image, List[Component]]:
    valid_components = []  # type: List[Component]
    labels = np.zeros_like(img).astype(np.uint8)
    for component in components:
        compon= np.array([[p.y, p.x] for p in component])
        box= cv2.boundingRect(compon)
        ss=len(compon)
        if (len(compon)>3):
            hull=None
            try:
                hull = ConvexHull(compon)
            except :
                hull=None
            if (hull):
                area = hull.area
                aspect1=box[2]
                aspect2=box[3]
                areaBox=aspect2*aspect1
                if (min<=area<max):
                    if((ration*(areaBox))>(ss)>((1/ration)*(areaBox))):
                        if ((ration * (area)) > (ss) > ((1/ration) * (area))):
                              for p in component:
                               labels[p.y, p.x] = 255
                              valid_components.append(component)
    return labels, valid_components
