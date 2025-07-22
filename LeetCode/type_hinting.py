from typing import Optional 

def f(x: Optional[str]) -> bool:
    is_str = isinstance(x, str)
    if is_str and x is not None:
        return True
    else:
        return False

if __name__ == "__main__":
    result = f("hi")
    print(result)
