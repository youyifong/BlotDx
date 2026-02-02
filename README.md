# BlotDx: HSV Western Blot Diagnostic

This is the private repo for work on HSV Western Blot diagnostic. The public repo for results dissemination is BlotDx.

A sheet refers to a physical sheet of blot images. Each sheet contains up to 24 pairs of blot images. Each pair corresponds to one sample. Blot and strip are used interchangeably.

## Directory structure:

- Image_Annotation contains the json files for the gt bounding boxes for DET and SEG.

- Py_manus1 contains python scripts for manuscript 1

- Py_pre_post_processing contains the scripts for data pre-processing and post-processing.

- R contains R scripts

- VGG Image Annotator. Copied from C:/Program Files to here so that we can both use the same json files, without the need to update the file path inside json files.


## Notes on VGG Image Annotator (VIA)

- Project/load json project file. Edit the json file in a simple text editor to set default file path correctly. Alternatively, we now put the VGG program folder in the repo folder. This way the path can be relative path.
 
- There is a little trick in VIA. To edit a corner, there is no need to drag it, which is harder to do, rather, one can simply click at a new position and the corner will be repositioned. This helps a lot in editing.

- When a project is saved, a json file is downloaded.
 
- Run VIAprojectJSON_to_masks.py to generate mask files.

