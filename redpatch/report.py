"""

report

A module for creating redpatch run reports; used by 'redpatch-batch-process'


Workflow Overview
-----------------

1. Create object, specify run folder, image objects and filter settings
2. Add calculated annotated sub images for images
3. write to html


Basic Usage
-----------

1. Import module and create object

    .. highlight:: python
    .. code-block:: python

        import redpatch as rp
        fs = rp.FilterSettings()
        fs.read("settings.yml")
        source_folder = "some/folder"
        image_files = ["img1.jpg", "img2.jpg]
        report = rp.RPReport(source_folder, image_files, fs)


2. Add subimages and annotations

    .. highlight:: python
    .. code-block:: python

        report.add_subimages(image_file, sub_image_tag)
        report.add_annotated_subimages(image_file, sub_image_tag)


3. Write report and images to results folder

    .. highlight:: python
    .. code-block:: python

        report.write("some/folder")

"""

import base64
import datetime
import os

from yattag import Doc


class RPReport(object):
    """
    Class representing an HTML report page. Used by batch-process
    """

    def __init__(self, source_directory, images, filter_settings):
        self.source_directory = source_directory
        self.images = images
        self.filter_settings = filter_settings
        self.sub_images = {i: [] for i in images}
        self.summary = None
        self.annotated_sub_images = {i: [] for i in images}

    def add_subimages(self, s):
        self.sub_images[s.parent_image_file].append(s.imtag)

    def add_annotated_subimages(self, s):
        self.annotated_sub_images[s.parent_image_file].append(s.annot_imtag)

    def _image_to_data_uri(self, image_path, outdir):
        """Convert an image file to a base64 data URI for embedding in HTML.

        This makes the report.html self-contained and downloadable without
        broken image links.
        """
        # The image_path is typically relative to the current working directory
        # (e.g., "uploads/uuid/output/full/image.jpg")
        # Try the path as-is first, then try relative to outdir
        if os.path.isabs(image_path):
            full_path = image_path
        elif os.path.exists(image_path):
            full_path = image_path
        else:
            # Try resolving relative to outdir (just the basename)
            full_path = os.path.join(outdir, os.path.basename(image_path))

        # Determine MIME type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        try:
            with open(full_path, "rb") as f:
                image_data = f.read()
            b64_data = base64.b64encode(image_data).decode("utf-8")
            return f"data:{mime_type};base64,{b64_data}"
        except (IOError, OSError) as e:
            # If image can't be read, fall back to original path
            print(f"Warning: Could not embed image {image_path}: {e}")
            return image_path

    def _header(self):
        return "Redpatch Report"

    def _run_info(self):
        return "Run at: {}\n Source Directory {} \n Filter Settings {}".format(
            datetime.datetime.now(), self.source_directory, self.filter_settings
        )

    def _css(self):
        return """
        <style>
        body {
            background-color: linen;
            font-family: sans-serif;
            margin: 50px;
        }
        .img_div img {
            max-width: 100%;
            max-height: 400px;
        }

        </style>
        """

    def write(self, outdir):
        doc, tag, text, line = Doc().ttl()
        doc.asis("<!DOCTYPE html>")
        with tag("html"):
            doc.asis(self._css())
            with tag("body"):
                with tag("h1"):
                    text(self._header())
                with tag("h2"):
                    text("Run Details")
                with tag("ul"):
                    line("li", "Run at: {}".format(datetime.datetime.now()))
                    line("li", "Source Directory: {}".format(self.source_directory))
                    line("li", "Filter Settings")
                    with tag("ul"):
                        line(
                            "li",
                            "Leaf Area: H: {}, S: {}, V: {}".format(
                                self.filter_settings["leaf_area"]["h"],
                                self.filter_settings["leaf_area"]["s"],
                                self.filter_settings["leaf_area"]["v"],
                            ),
                        )
                        line(
                            "li",
                            "Healthy Area: H: {}, S: {}, V: {}".format(
                                self.filter_settings["healthy_area"]["h"],
                                self.filter_settings["healthy_area"]["s"],
                                self.filter_settings["healthy_area"]["v"],
                            ),
                        )
                        line(
                            "li",
                            "Lesion Area: H: {}, S: {}, V: {}".format(
                                self.filter_settings["lesion_area"]["h"],
                                self.filter_settings["lesion_area"]["s"],
                                self.filter_settings["lesion_area"]["v"],
                            ),
                        )
                        line(
                            "li",
                            "Lesion Centre: H: {}, S: {}, V: {}".format(
                                self.filter_settings["lesion_centre"]["h"],
                                self.filter_settings["lesion_centre"]["s"],
                                self.filter_settings["lesion_centre"]["v"],
                            ),
                        )
                    line("li", "Images Used")
                    with tag("ul"):
                        for i in self.images:
                            line("li", i)
                with tag("h2"):
                    text("Image Breakdown")
                    for i in self.images:
                        with tag("h3"):
                            text(i)
                            with tag("div", klass="img_div"):
                                for idx, s in enumerate(self.sub_images[i]):
                                    with tag("div", klass="sub_img_div"):
                                        text("Sub-Image: {}".format(idx + 1))
                                        # Embed images as base64 data URIs for self-contained HTML
                                        doc.stag(
                                            "img",
                                            src=self._image_to_data_uri(s, outdir),
                                        )
                                        doc.stag(
                                            "img",
                                            src=self._image_to_data_uri(
                                                self.annotated_sub_images[i][idx],
                                                outdir,
                                            ),
                                        )

        with open(os.path.join(outdir, "report.html"), "w") as out:
            out.write(doc.getvalue())
