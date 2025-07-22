class CallableClass:

    __slots__ = ["sound"]
    
    def __init__(self, sound: str):
        self.sound = sound

    def __call__(self):
        print(self.sound)

if __name__ == "__main__":
    c = CallableClass("bark")
    c()
        
