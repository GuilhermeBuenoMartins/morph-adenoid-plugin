# morph-adenoid-plugin
A tool based on deep morphological networks to aid in the diagnosis of adenoid hypertrophy.

## Installation and use
First of all, the MorphAdenoid was created as an ImageJ's plugin and, to use it, is necessary the download and installation of the free software [ImageJ](https://imagej.net/ij/download.html) or acessing: https://imagej.net/ij/download.html.
After installing the software, you must download the [MorphAdenoid.zip](https://drive.google.com/file/d/1zCjr_P6zh0Lft6P4ElbCwWJPPIuoV9Cn/view?usp=drive_link) that is available in: https://drive.google.com/file/d/1zCjr_P6zh0Lft6P4ElbCwWJPPIuoV9Cn/view?usp=drive_link.
This zip file will must contain the following content:
- One file named `morph_adenoid_plugin.jar`;
- One folder `model` containing:
    - `layer_0.h5`;
    - `layer_1.h5`;
    - `layer_3.h5`;
    - `layer_4.h5`; and
    - `layer_5.h5`.

Move all the aforementioned content (`morph_adenoid_plugin.jar` and the `model` folder) to the `jars` folder in ImageJ's `plugins` folder. If you have encountering problems to find ImageJ's `plugins` folder, open the software, access the `File -> Show Folder -> Plugins`.
It will open your file explorer of your Operation System where the ImageJ's `plugins` folder is located.
Thus, when moved the files, it is just restart the software ImageJ and use it with RGB image or an video according to restriction of your ImageJ.