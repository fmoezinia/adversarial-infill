def extract_line(img):
    with open("list_attr_celeba.csv", "r") as f:
        attrs = list(f)[img]
        ja = attrs.split(",")
        attrs = [int(i) for i in ja[1:]]
        return attrs

# print extract_line(3)
