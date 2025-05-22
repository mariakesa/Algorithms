import statistics

def choosePivot(pivot_type, arr, low, high):
    if pivot_type == 'first':
        return low
    elif pivot_type == 'last':
        return high
    elif pivot_type == 'median':
        mid = (low + high) // 2
        candidates = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
        candidates.sort()  # sort by value
        return candidates[1][1]
        
    
def swap(arr, k, l):
    arr[l], arr[k] = arr[k], arr[l]

def partition(arr, pivot_index, low, high):
    swap(arr, low, pivot_index)  # Move pivot to front
    pivot = arr[low]
    i = low + 1
    for j in range(low + 1, high+1):
        if arr[j] < pivot:
            swap(arr, i, j)
            i += 1
    swap(arr, low, i - 1)
    return i - 1  # Return new index of the pivot

def quicksort(arr, low, high, pivot_type='first'):
    if low < high:
        pivot_index = choosePivot(pivot_type, arr, low, high)
        new_pivot = partition(arr, pivot_index, low, high)
        quicksort(arr, low, new_pivot - 1, pivot_type)
        quicksort(arr, new_pivot + 1, high, pivot_type)
        
def test():
    arr = [10, 7, 8, 9, 1, 5]
    quicksort(arr, 0, len(arr)-1)
    print(arr)

test()