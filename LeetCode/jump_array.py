from typing import List

def can_jump(nums: List[int]):
    reachable = 0
    for i in range(len(nums)):
        if i > reachable:
            return False
        else:
            reachable = max(i + nums[i], reachable)
    return True

if __name__ == "__main__":
    nums = [1, 2, 0, 1]
    result = can_jump(nums)
    print(result)
