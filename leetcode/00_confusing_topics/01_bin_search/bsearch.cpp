#include <iostream>
#include <vector>

// 二分查找
int bsearch(std::vector<int>& nums, int value) {
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (nums[mid] == value) {
            return mid;
        } else if (nums[mid] < value) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return -1;
}

int bsearch_internally(std::vector<int>& nums, int low, int high, int value) {
    if (low > high) {
        return -1;
    }
    int mid = low + (high - low) / 2;
    if (nums[mid] == value) {
        return mid;
    } else if (nums[mid] < value) {
        return bsearch_internally(nums, mid + 1, high, value);
    } else {
        return bsearch_internally(nums, low, high - 1, value);
    }

    return -2;
}
// 二分查找递归实现
int bsearch_recu(std::vector<int>& nums, int value) {
    return bsearch_internally(nums, 0, nums.size() - 1, value);
}

// 查找第一个值等于给定值的元素
int bsearch_first(std::vector<int>& nums, int value) {
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (nums[mid] > value) {
            high = mid - 1;
        } else if (nums[mid] < value) {
            low = mid + 1;
        } else {
            if ((mid == 0) || (nums[mid - 1] != value)) {
                return mid;
            } else {
                high = mid - 1;
            }
        }
    }

    return -1;
}

// 查找最后一个值等于给定值的元素
int bsearch_last(std::vector<int>& nums, int value) {
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (nums[mid] > value) {
            high = mid - 1;
        } else if (nums[mid] < value) {
            low = mid + 1;
        } else {
            if ((mid == nums.size() - 1) || (nums[mid + 1] != value)) {
                return mid;
            } else {
                low = mid + 1;
            }
        }
    }

    return -1;
}

// 查找第一个大于等于给定值的元素
int bsearch_up(std::vector<int>& nums, int value) {
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (nums[mid] >= value) {
            if ((mid == 0) || (nums[mid - 1] < value)) {
                return mid;
            } else {
                high = mid - 1;
            }
        } else {
            low = mid + 1;
        }
    }

    return -1;
}


// 查找最后一个小于等于给定值的元素
int bsearch_down(std::vector<int>& nums, int value) {
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (nums[mid] > value) {
            high = mid - 1;
        } else {
            if ((mid == nums.size() - 1) || (nums[mid + 1] > value)) {
                return mid;
            } else {
                low = mid + 1;
            }
        }
    }

    return -1;
}


int main() {
    std::vector<int> nums = { 8, 11, 19, 23, 27, 33, 45, 55, 67, 98};

    int idx = bsearch(nums, 19);

    std::cout << "bsearch: " <<  idx << std::endl;

    std::vector<int> nums2 = { 1,2,3,4,6,8,8,8,11,18};

    idx = bsearch_first(nums2, 8);
    std::cout << "bsearch_first: " <<  idx << std::endl;

    idx = bsearch_last(nums2, 8);
    std::cout << "bsearch_last: " <<  idx << std::endl;

    idx = bsearch_up(nums2, 8);
    std::cout << "bsearch_up: " <<  idx << std::endl;

    idx = bsearch_down(nums2, 6);
    std::cout << "bsearch_down: " <<  idx << std::endl;
}
