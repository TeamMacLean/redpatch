import numpy as np
from skimage import measure

import redpatch as rp


class LesionCentre(object):
    """
    Represents a lesion centre.

    Modifies a scikit RegionProp and requires information from a
    parent RegionProp

    :param: rprop skimage.measure._regionprops._RegionProperties -- region to make a lesion centre
    :param: parent_rprop skimage.measure._regionprops._RegionProperties -- parent (enclosing) region of rprop
    :param: scale float -- measure of the scale of the image, either None (scale in image pixels) or real units from scale card
    :param: pixel_length -- length of side of one pixel, available when scale
    :param: max_ratio float -- threshold of region length / width to pass filter
    :param: minimum_size float -- minimum size to pass filter. calculated in units of scale
    :param: prop_across_parent float -- proportion of width of enclosing object the lesion centre must be to pass filter. EG 0.1 indicates that the lesion centre must be greater than 10 percent of the current width across the parent object
    """

    def __init__(
        self,
        rprop: measure._regionprops._RegionProperties = None,
        parent_rprop: measure._regionprops._RegionProperties = None,
        scale: float = None,
        pixel_length=None,
        max_ratio: float = None,
        minimum_size: float = None,
        prop_across_parent: float = None,
    ):
        for attr_name in dir(rprop):
            if not attr_name.startswith("_"):
                try:
                    val = getattr(rprop, attr_name)
                    setattr(self, attr_name, val)
                except AttributeError:
                    pass

        # Handle scikit-image attribute naming changes (axis_major_length vs major_axis_length)
        if hasattr(rprop, "axis_major_length") and not hasattr(
            self, "major_axis_length"
        ):
            self.major_axis_length = rprop.axis_major_length
        if hasattr(rprop, "axis_minor_length") and not hasattr(
            self, "minor_axis_length"
        ):
            self.minor_axis_length = rprop.axis_minor_length

        self.parent_lesion_region = parent_rprop.label

        try:
            self.long_axis_to_short_axis_ratio = (
                self.major_axis_length / self.minor_axis_length
            )
        except (ZeroDivisionError, AttributeError):
            self.long_axis_to_short_axis_ratio = float("inf")

        self.offset_to_parent = self._calc_offset_to_parent(parent_rprop)

        self.corrected_bbox = self._correct_bbox(parent_rprop)
        self.corrected_coords = self._correct_coords(parent_rprop)
        self.corrected_centroid = self._correct_centroids()
        self.prop_across_parent = self._get_prop_across_parent(parent_rprop)
        if scale:
            self.scale = scale
            self.size = self.area * (pixel_length**2)
        else:
            self.scale = "NA"
            self.size = "NA"

        # if scale:
        #    self.size = rp.pixel_volume_to_circular_area(self.area, scale)
        self.passed = self._passes(
            max_ratio=max_ratio,
            minimum_size=minimum_size,
            prop_across_parent=prop_across_parent,
        )

    def _correct_bbox(self, parent_rprop):
        return [int(x) + int(y) for x, y in zip(parent_rprop.bbox, self.bbox)]

    def _calc_offset_to_parent(self, parent_rprop):
        offsets = parent_rprop.bbox[0:2]
        return offsets

    def _correct_coords(self, parent_rprop):
        p = np.asarray(self.coords)
        min_row, min_col, _, _ = parent_rprop.bbox
        return np.add(p, [min_row, min_col])

    def _correct_centroids(self):
        res = (
            self.centroid[0] + self.offset_to_parent[0],
            self.centroid[1] + self.offset_to_parent[1],
        )
        return res

    def _get_prop_across_parent(self, parent_rprop):
        try:
            row_needed = int(self.centroid[0])
            row = list(parent_rprop.image[row_needed, :])
            col = list(parent_rprop.image[:, row_needed])
        except (
            IndexError
        ):  # if due to rounding of corrected value index is out of bounds
            min_row, _, max_row, _ = parent_rprop.bbox
            row = [True] * (max_row - min_row)
        left_leaf_margin = row.index(1)
        try:
            right_leaf_margin = row[left_leaf_margin:].index(0) + left_leaf_margin
        except ValueError:
            right_leaf_margin = len(row) - 1
        col = self.centroid[1]
        dist_to_left = abs(left_leaf_margin - col)
        dist_to_right = abs(right_leaf_margin - col)
        res = min([dist_to_left, dist_to_right])
        prop_across_parent = res / len(row)
        return prop_across_parent

    def __getitem__(self, item):
        return getattr(self, item)

    def _passes(self, max_ratio=None, minimum_size=None, prop_across_parent=None):
        length_pass = self.long_axis_to_short_axis_ratio <= max_ratio
        try:
            size_pass = self.size >= minimum_size
        except TypeError:
            size_pass = self.area >= minimum_size
        distance_pass = self.prop_across_parent >= prop_across_parent
        if length_pass and size_pass and distance_pass:
            return True
        else:
            return False
