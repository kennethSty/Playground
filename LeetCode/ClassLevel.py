class ClassLevel:

    y = 5
    
    def __init__(self, x):
        self.x = x

    @classmethod
    def f(cls, a):
        print(cls.y, a)

    
