class PrivateClass:
    __slots__ = ["normal", "__special"]

    def __init__(self, normal, special):
        self.normal = normal
        self.__special = special


    def get_special(self):
        return self.__special

if __name__ == "__main__":
    pc = PrivateClass("normal", "special")
    special = pc.get_special()
    print(special)
