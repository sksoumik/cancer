### Skin cancer detection using computer vision. 

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer.

#### Dataset
[Dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) is collected from kaggle. 

The images are provided in DICOM format. This can be accessed using commonly-available libraries like `pydicom`, and contains both image and metadata. It is a commonly used medical imaging data format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

#### ML Model
SE-ResNeXt-50

