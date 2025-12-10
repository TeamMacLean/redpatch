import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from shapely.geometry.polygon import Polygon
from skimage import color, io, measure

import redpatch as rp


def _get_object_properties(label_array: np.ndarray, intensity_image: np.ndarray = None):
    """given a label array returns a list of computed RegionProperties objects."""
    return measure.regionprops(label_array, intensity_image=intensity_image)


def get_sub_images(
    imfile,
    file_settings=None,
    dest_folder=None,
    min_lesion_area=None,
    scale=None,
    pixel_length=None,
    max_lc_ratio=None,
    min_lc_size=None,
    lc_prop_across_parent=None,
):
    """
    extracts different leaves from a single image file, returning them as individual SubImage objects.

    :param: imfile str -- path to the image
    :param: file_settings str -- FilterSettings object with image segmentation options
    :param: dest_folder str -- folder in which to place results files
    :param: min_lesion_area float -- minimum area for a lesion to pass filter. In either pixels or actual size if 'scale' passed
    :param: scale float -- pixels per real unit length, if known or computed earlier.
    :param: max_lc_ratio float -- maximum length/width ratio of lesion centre to pass filter
    :param: min_lc_size float -- minimum lesion centre size. Computed in real units if 'scale' passed. Computed as area of circle with same pixel volume as the centre.
    :param: lc_prop_across_parent float -- minimum proportion lesion centre must be across the width of the parent lesion (in the row the centre centroid occurs) to pass filter
    """
    im = rp.load_as_hsv(imfile)
    leaf_area_mask = rp.griffin_leaf_regions(
        im,
        h=file_settings["leaf_area"]["h"],
        s=file_settings["leaf_area"]["s"],
        v=file_settings["leaf_area"]["v"],
    )
    labelled_leaf_area, _ = rp.label_image(leaf_area_mask)
    leaf_area_properties = rp.get_object_properties(labelled_leaf_area)
    leaf_areas_to_keep = rp.filter_region_property_list(
        leaf_area_properties, rp.is_not_small
    )
    cleaned_leaf_area = rp.clean_labelled_mask(labelled_leaf_area, leaf_areas_to_keep)
    final_labelled_leaf_area, _ = rp.label_image(cleaned_leaf_area)
    three_d_final_labelled_leaf_area = np.dstack(
        (final_labelled_leaf_area, final_labelled_leaf_area, final_labelled_leaf_area)
    )  # stupid hack because get object properties needs 3d label array to match image
    props = rp.subimage._get_object_properties(
        three_d_final_labelled_leaf_area, intensity_image=im
    )
    sub_labels = [p.image.astype(int)[:, :, -1] for p in props]
    sub_images = [p.intensity_image for p in props]
    cleared_leaf_sub_images = [
        rp.clear_background(sub_images[i], sub_labels[i])
        for i in range(len(sub_labels))
    ]
    sub_image_objs = []
    for sub_i_idx, sub_i in enumerate(cleared_leaf_sub_images, 1):
        sub_image_objs.append(
            rp.SubImage(
                sub_i,
                sub_i_idx,
                imfile,
                file_settings=file_settings,
                dest_folder=dest_folder,
                min_lesion_area=min_lesion_area,
                scale=scale,
                pixel_length=pixel_length,
                max_lc_ratio=max_lc_ratio,
                min_lc_size=min_lc_size,
                lc_prop_across_parent=lc_prop_across_parent,
            )
        )
    return sub_image_objs


class SubImage(object):
    """
    class representing sub-image (leaf containing area) of a larger image.

    :ivar sub_i: the subimage
    :ivar sub_i_idx: the index of the subimage from the subimage list
    :ivar scale: the scale of the image if computed
    :ivar pixel_length: the length of a pixel side in real units
    :ivar imtag: the name of the file this subimage is referred to in the output
    :ivar annot_imtag: the name of the annotated image file this subimage is referred to in the output
    :ivar parent_image_file: the name of the file this subimage is derived from
    :ivar healthy_obj_props: list of HealthyAreas found in the subimage
    :ivar leaf_area_props: list of LeafAreas found in the subimage
    :ivar lesion_area_props: list of LesionAreas found in the subimage
    :ivar lesion_centre_props: list of LesionCentres found in the subimage
    """

    def __init__(
        self,
        sub_i,
        sub_i_idx,
        parent_image_file,
        file_settings=None,
        dest_folder=None,
        min_lesion_area=None,
        scale=None,
        pixel_length=None,
        max_lc_ratio=None,
        min_lc_size=None,
        lc_prop_across_parent=None,
    ):
        self.sub_i = sub_i
        self.index = sub_i_idx
        if scale:
            self.scale = scale
            self.pixel_length = pixel_length
        else:
            self.scale = "NA"
            self.pixel_length = "NA"
        self.imtag = os.path.join(
            dest_folder,
            "{}_sub_image_{}{}".format(
                os.path.basename(parent_image_file), sub_i_idx, ".jpg"
            ),
        )
        self.annot_imtag = os.path.join(
            dest_folder,
            "{}_sub_image_{}{}".format(
                os.path.basename(parent_image_file), sub_i_idx, "_annotated.jpg"
            ),
        )
        self.parent_image_file = parent_image_file
        self.healthy_obj_props = self._get_healthy_areas(
            sub_i, file_settings, scale, pixel_length
        )
        self.leaf_area_props = self._get_leaf_areas(
            sub_i, file_settings, scale, pixel_length
        )
        self.lesion_area_props = self._get_lesion_areas(
            sub_i, file_settings, scale, pixel_length, min_lesion_area=min_lesion_area
        )  # 0 to many per image
        self.lesion_centre_props = self._get_lesion_centres(
            sub_i,
            self.lesion_area_props,
            scale=scale,
            pixel_length=pixel_length,
            max_lc_ratio=max_lc_ratio,
            min_lc_size=min_lc_size,
            lc_prop_across_parent=lc_prop_across_parent,
        )

    def _get_healthy_areas(self, im, fs, scale, pixel_length):
        """
        Finds healthy areas according to filtersettings in fs

        :param im: the image to search
        :param fs: a FilterSettings object
        :return: list of HealthyArea objects
        """
        healthy_mask, _ = rp.griffin_healthy_regions(
            im,
            h=fs["healthy_area"]["h"],
            s=fs["healthy_area"]["s"],
            v=fs["healthy_area"]["v"],
        )
        labelled_healthy_area, _ = rp.label_image(healthy_mask)
        labelled_healthy_area_properties = rp.get_object_properties(
            labelled_healthy_area
        )
        return [
            rp.HealthyArea(o, scale, pixel_length)
            for o in labelled_healthy_area_properties
        ]

    def _get_leaf_areas(self, im, fs, scale, pixel_length):
        """
        Finds whole leaf areas according to filtersettings in fs

        :param im: the image to search
        :param fs: a FilterSettings object
        :return: list of LeafArea objects
        """
        leaf_area_mask = rp.griffin_leaf_regions(
            im, h=fs["leaf_area"]["h"], s=fs["leaf_area"]["s"], v=fs["leaf_area"]["v"]
        )
        labelled_leaf_area, _ = rp.label_image(leaf_area_mask)
        leaf_area_properties = rp.get_object_properties(labelled_leaf_area)
        return [rp.LeafArea(o, scale, pixel_length) for o in leaf_area_properties]

    def _get_lesion_areas(self, im, fs, scale, pixel_length, min_lesion_area=None):
        """
        Finds lesion areas according to filtersettings in fs

        :param im: the image to search
        :param fs: a FilterSettings object
        :param min_lesion_area: the minimum area to set a LesionArea objects passed attribute to TRUE
        :return: list of LesionArea objects
        """
        lesion_area_mask, _ = rp.griffin_lesion_regions(
            im,
            h=fs["lesion_area"]["h"],
            s=fs["lesion_area"]["s"],
            v=fs["lesion_area"]["v"],
        )
        labelled_lesion_area, _ = rp.label_image(lesion_area_mask)
        labelled_lesion_area_properties = rp.get_object_properties(labelled_lesion_area)
        return [
            rp.LesionArea(o, scale, pixel_length, min_lesion_area=min_lesion_area)
            for o in labelled_lesion_area_properties
        ]

    def _get_lesion_centres(
        self,
        im,
        lesion_area_region_prop_list,
        scale=None,
        pixel_length=None,
        max_lc_ratio=None,
        min_lc_size=None,
        lc_prop_across_parent=None,
    ):
        """
        Finds lesion centres in LesionAreas
        :param im: image to find LesionCentres in
        :param lesion_area_region_prop_list: LesionAreas to search in
        :param scale: scale of the image
        :param max_lc_ratio: maximum length to width for the LesionCentre to be marked as passed
        :param min_lc_size: minimum size (real units or pixels dependent on scale) for LesionCentre to be marked as passed
        :param lc_prop_across_parent: minimum proportion lesion centre must be across the width of the parent lesion (in the row the centre centroid occurs) to pass filter
        :return: list of LesionCentres
        """

        annotated_lesion_centre_props = []
        for lesion_area in lesion_area_region_prop_list:
            # if lesion_area.area >= min_lesion_area:
            raw_lesion_centre_props = rp.griffin_lesion_centres(im, lesion_area)
            for r in raw_lesion_centre_props:
                annotated_lesion_centre_props.append(
                    rp.LesionCentre(
                        rprop=r,
                        parent_rprop=lesion_area,
                        scale=scale,
                        pixel_length=pixel_length,
                        max_ratio=max_lc_ratio,
                        minimum_size=min_lc_size,
                        prop_across_parent=lc_prop_across_parent,
                    )
                )
        return annotated_lesion_centre_props

    def _calc_size(self, img):
        """
        get size in inches of an annotated output image of img.shape at 72 DPI
        :param img: img to use
        :return: (width_inches, height_inches)
        """
        img = img
        h, w, _ = img.shape
        dpi = 72
        inches_w = w / dpi
        inches_h = h / dpi
        return tuple([inches_w, inches_h])

    def _make_polygons_for_image(self, list_of_rprops):
        """
        make overlay colour patches for the annotated output image
        :param list_of_rprops: ImageArea or subclass for which to make patches for
        :return: list of polygons
        """
        polys = []
        for rprop in list_of_rprops:
            if isinstance(rprop, rp.LesionCentre):
                coords = rprop.corrected_coords
            else:
                coords = rprop.coords
            if len(coords) > 2:  # need 3 points for a polygon
                p = np.asarray(coords)
                # Swap row,col to x,y
                p = p[:, [1, 0]]
                # Convert to list of tuples for Shapely 2.x compatibility
                try:
                    poly = Polygon(p.tolist())
                    x, y = poly.exterior.xy
                    polys.append([x, y])
                except Exception:
                    # Skip invalid polygons
                    pass
        return polys

    def write_annotated_sub_image(self):
        """
        create output annotated subimage jpeg with results overlay
        :return: None
        """
        size = self._calc_size(self.sub_i)
        fig = plt.figure(figsize=size)
        img = skimage.img_as_ubyte(color.hsv2rgb(self.sub_i))
        plt.imshow(img)
        hcol = (127 / 255, 191 / 255, 63 / 255, 0.5)
        lcol = (243 / 255, 80 / 255, 21 / 255, 0.5)
        ccol = (248 / 255, 252 / 255, 17 / 255, 0.66)
        healthy_polys = self._make_polygons_for_image(self.healthy_obj_props)
        lesion_polys = self._make_polygons_for_image(self.lesion_area_props)
        centre_polys = self._make_polygons_for_image(self.lesion_centre_props)

        for p in healthy_polys:
            plt.plot(p[0], p[1], color=hcol)

        for p in lesion_polys:
            plt.plot(p[0], p[1], color=lcol)

        for p in centre_polys:
            plt.plot(p[0], p[1], color=ccol)

        ax = plt.gca()
        for c in self.lesion_centre_props:
            ax.annotate(
                str(c.label),
                xy=tuple(reversed(c.corrected_centroid)),
                xycoords="data",
                color="white",
            )

        h_patch = mpatches.Patch(color=hcol, label="Healthy")
        l_patch = mpatches.Patch(color=lcol, label="Lesion")
        c_patch = mpatches.Patch(color=ccol, label="Centres")
        plt.legend(
            bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure,
            handles=[h_patch, l_patch, c_patch],
            loc="upper right",
        )
        plt.savefig(
            self.annot_imtag,
            dpi=72,
        )
        plt.close(fig)

    def write_sub_image(self):
        """
        write out subimage (nonannotated)
        :return: None
        """
        io.imsave(self.imtag, skimage.img_as_ubyte(color.hsv2rgb(self.sub_i)))

    def _make_pandas(
        self, regions, area_type=None, image_file=None, sub_image_index=None
    ):
        """
        makes a pandas dataframe of the results

        :param regions: ImageArea or subclass list
        :param area_type: type of the image_area
        :param image_file: the image the regions are derived from
        :param sub_image_index: the index of the subimage the regions are derived from
        :return: pandas.dataframe
        """
        nrow = len(regions)
        d = {}
        # d = {p: [rp[p] for rp in regions] for p in props}
        d["label"] = [r.label for r in regions]
        d["pixels_in_area"] = [r.area for r in regions]
        d["area_type"] = [area_type] * nrow
        d["image_file"] = [image_file] * nrow
        d["sub_image_index"] = [sub_image_index] * nrow
        d["scale"] = [r.scale for r in regions]
        d["passed"] = [r.passed for r in regions]
        d["size"] = [r.size for r in regions]
        d["parent_lesion_region"] = [r.parent_lesion_region for r in regions]
        d["long_axis_to_short_axis_ratio"] = [
            r.long_axis_to_short_axis_ratio for r in regions
        ]
        d["subimage_centre"] = [
            ",".join(map(str, [int(i) for i in r.centroid])) for r in regions
        ]
        if area_type == "lesion_centre":
            d["subimage_centre"] = [
                ",".join(map(str, [int(i) for i in lc.corrected_centroid]))
                for lc in regions
            ]

        return pd.DataFrame(d)

    def create_results_dataframe(self, on_webserver=False):
        """
        generates a pandas dataframe of results from the SubImage object
        :return: pandas.dataframe
        """
        hdf = self._make_pandas(
            self.healthy_obj_props,
            area_type="healthy_region",
            image_file=self.parent_image_file,
            sub_image_index=self.index,
        )
        ldf = self._make_pandas(
            self.lesion_area_props,
            area_type="lesion_region",
            image_file=self.parent_image_file,
            sub_image_index=self.index,
        )
        ladf = self._make_pandas(
            self.leaf_area_props,
            area_type="leaf_area",
            image_file=self.parent_image_file,
            sub_image_index=self.index,
        )
        if not on_webserver:
            lcdf = self._make_pandas(
                self.lesion_centre_props,
                area_type="lesion_centre",
                image_file=self.parent_image_file,
                sub_image_index=self.index,
            )
            return hdf.append([ladf, ldf, lcdf], ignore_index=True)
        return hdf.append([ladf, ldf]).drop("parent_lesion_region", axis=1)
