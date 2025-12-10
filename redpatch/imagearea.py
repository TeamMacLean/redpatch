class ImageArea(object):
    """
    Base class for image areas that are created by rp.get_subimages()
    Copies user attributes from skimage.measure.regionprops and
    adds a few attributes

    :ivar size: computed size of area if scale is provided in cm2
    :ivar passed: computed filter pass/fail. Can be NA if area does not have filter criteria
    :ivar parent_lesion_region: ID of parent (enclosing region) where exists
    :ivar prop_across_parent: computed distance across parent region width a lesion centre falls. Is NA for other regions
    :ivar subimage_centre: centre pixels of image area
    :scale: scale of the image
    :param: pixel_length -- length of side of one pixel, available when scale

    """

    def __init__(self, rprop, scale, pixel_length):
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

        if scale:
            self.scale = scale
            self.size = (pixel_length**2) * self.area
        else:
            self.scale = "NA"
            self.size = "NA"
        try:
            self.long_axis_to_short_axis_ratio = (
                self.major_axis_length / self.minor_axis_length
            )
        except (ZeroDivisionError, AttributeError):
            self.long_axis_to_short_axis_ratio = float("inf")

        # self.size = "NA"
        self.passed = "NA"
        self.parent_lesion_region = "NA"
        self.prop_across_parent = "NA"
        self.subimage_centre = self.centroid

    def __getitem__(self, item):
        return getattr(self, item)


class HealthyArea(ImageArea):
    pass


class LesionArea(ImageArea):
    def __init__(self, rprop, scale, pixel_length, min_lesion_area=None):
        super().__init__(rprop, scale, pixel_length)
        if self.scale == "NA":
            if self.area < min_lesion_area:
                self.passed = False
            else:
                self.passed = True
        else:
            # self.size = self.area / self.scale
            if self.size < min_lesion_area:
                self.passed = "FALSE"
            else:
                self.passed = "TRUE"


class LeafArea(ImageArea):
    pass
