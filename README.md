# Tutorial
By Do Ngoc Tuyen ngoctuyendo.hust@gmail.com
## Facebook tagged images filter and preprocessing
- First step make sure you have a folder with tagged image cralw from Facebook. The folder have struture like that:

```Shell
    faces/
       img_1.png
       img_2.png
       ...
       img_n.png
```
- The next step you must to run the script face_filter.py

``python face_filter.py --input_dir /path/to/faces_folder --output_dir /path/to/output_folder``
## Model download
- Download mtcnn model from https://drive.google.com/drive/folders/1T8rMQLGiC9oGZtHIktTZxDsYGLlUblRO?usp=sharing and paste it into project folder:
```Shell
    root/
       mtcnn-model/
```
