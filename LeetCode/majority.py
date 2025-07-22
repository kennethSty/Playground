from typing import List

def majority(nums: List[int]):
    candidate = nums[0]
    count = 1

    for i in range(1, len(nums)):
        if count == 0:
            candidate = nums[i]
        count += (1 if nums[i] == candidate else -1)
    return candidate
    
if __name__ == "__main__":
    nums = [2,2,1,1,1,2,2]
    result = majority(nums)
    print(result)
