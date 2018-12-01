import os


def extract_line(img):
    with open("list_attr_celeba.csv", "r") as f:
        attrs = list(f)[img]
        print(attrs, type(attrs))
        ja = attrs.split(",")
        print(ja)
