def merge(left, right):
    i=0
    j=0
    sorted_array=[]
    while i<len(left) and j<len(right):
        if right[j]<=left[i]:
            sorted_array.append(right[j])
            j+=1
        else:
            sorted_array.append(left[i])
            i+=1
    sorted_array.extend(left[i:])
    sorted_array.extend(right[j:])

    return sorted_array

def merge_sort(n_arr):
    splitter=len(n_arr)//2
    if len(n_arr) <= 1:
        return n_arr
    
    left=n_arr[:splitter]
    right=n_arr[splitter:]

    sorted_left=merge_sort(left)
    sorted_right=merge_sort(right)

    return merge(sorted_left, sorted_right)
    
def test():
    n_arr=[1,5,3,2,4]
    sorted_arr=merge_sort(n_arr)
    print(sorted_arr)

test()