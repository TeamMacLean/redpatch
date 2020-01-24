.. image:: https://zenodo.org/badge/205859428.svg
   :target: https://zenodo.org/badge/latestdoi/205859428

====
redpatch
====

A package to find disease lesions in plant leaf images

Installation
============

``pip install redpatch``



Running in Interactive Mode
===========================

Examination of single images can be done in interactive mode. Walkthrough examples are provided in Jupyter notebooks. To start these use ``redpatch-start`` on the command line.


Running in Batch Mode
=====================

Batch processing is done on a folder of images. The script ``batch-process.py`` is used to run the process. It needs three pieces of information to run.

1. A source folder - the folder to read images from,  must contain images and nothing else
2. A destination folder - the folder the script will write results to. If it doesn't exist at run time it will be created. Existing folders may have their contents overwritten
3. A filter settings file - a YAML format file specifying the HSV values used in each segmentaion step. A default one can be created by the script.

Creating the filter settings file
---------------------------------

A default filter settings file can be generated as follows:

``redpatch-batch-process.py --create_default_filter ~/Desktop/default_filter.yml``

The settings will be written to the specified file.


Analysing a folder with images with no scale card
-------------------------------------------------

The basic call for the basic case is:

``redpatch-batch-process.py ---source_folder ~/Desktop/input_images --destination_folder ~/Desktop/test_out --filter_settings ~/Desktop/default_filter.yml``

The script will run and output will be produced in the destination folder. The same call works whether the folder contains one or many images and if the images contain one or many leaves,


Analysing a folder with images with a scale card
-------------------------------------------------

If all the images contain a scale card, and the _same_ scale card, then you can get information about area added to the output. Use the ``--scaled_card_side_length`` option to give the size of the scale card in centimetres.

``redpatch-batch-process.py --scale_card_side_length 5 --source_folder ~/Desktop/input_images --destination_folder ~/Desktop/test_out --filter_settings ~/Desktop/default_filter.yml``


Including a full results table
------------------------------

By default the script produces a summary results table. You will usually want a full account of every object found. A tidy format table of these results can be produced during the run with the ``--create_tidy_output`` option.

``redpatch-batch-process.py --create_tidy_output --source_folder ~/Desktop/input_images --destination_folder ~/Desktop/test_out --filter_settings ~/Desktop/default_filter.yml``




