class ClassMember:
    cls_attribute = 4
    __slots__ = ["other"]

    def __init__(self, other):
        self.other = other

    def __str__(self):
        return f"cls: {self.cls_attribute} + other: {self.other}"

if __name__ == "__main__":
    c = ClassMember("hi")
    print(c)
