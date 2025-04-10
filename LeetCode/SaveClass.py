class SaveClass:
    __slots__ = ["name", "score"]
    def __init__(self, name, score):
        self.name = name
        self.score = score


if __name__ == "__main__":
    footballer = SaveClass("thomas", 3)
    footballer.age = 20
