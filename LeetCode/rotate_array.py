from typing import List

def rotate_array(nums: List[int], k: int):
    nums.reverse()
    nums[:k] = reversed(nums[:k]) 
    nums[k:] = reversed(nums[k:])

if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7]
    k = 3
    rotate_array(nums, k)
    print(nums)
    
