from mongoengine import Document, StringField, IntField, FloatField, ListField, DictField, ReferenceField


from mongoengine import Document, StringField, IntField

# Image Class
class Image(Document):
    name = StringField(required=True)
    path = StringField(required=True)
    height = IntField(required=True)
    width = IntField(required=True)
    size = IntField(required=True)
    image_id = StringField(primary_key=True)
    num_pixels = IntField(required=True)

# NuClick Annotation
from mongoengine import ReferenceField

class NuclickAnnotation(Document):
    annotation_id = StringField(primary_key=True)
    label = StringField(required=True)
    x_coordinate = FloatField(required=True)
    y_coordinate = FloatField(required=True)
    image_id = ReferenceField(Image)

# SAM Annotation
class SAMannotation(Document):
    annotation_id = StringField(primary_key=True)
    label = StringField(required=True)
    x_coordinate = FloatField(required=True)
    y_coordinate = FloatField(required=True)
    height = IntField(required=True)
    width = IntField(required=True)
    image_id = ReferenceField(Image)

from mongoengine import Document, StringField, ReferenceField, ListField, DictField

class Annotation(Document):
    image_id = ReferenceField('Image')  # Assumes you have an Image class
    patch_id = StringField(required=True)
    label = StringField(required=True)
    type = StringField(required=True)  # e.g., 'Rectangle', 'Ellipse', 'Polygon', etc.
    coordinates = ListField(DictField())  # Stores coordinates as a list of dictionaries

