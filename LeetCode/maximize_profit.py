from typing import List

def max_profit(nums: List[int]):
    if len(nums) <= 1:
        return 0
    else:
        max_profit = 0
        buy = nums[0]
        for i in range(1, len(nums)):
            if nums[i] - buy > max_profit:
                max_profit = nums[i] - buy
            elif nums[i] < buy:
                buy = nums[i]

        return max_profit

if __name__ == "__main__":
    nums = [0, 0, 1, 0, 6, 0, 9]
    result = max_profit(nums)
    print(result) 
    
                
            
