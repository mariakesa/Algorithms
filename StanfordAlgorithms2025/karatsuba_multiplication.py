def karatsuba(x,y):
    n_digits=len(str(x))
    m=n_digits//2
    if n_digits==1:
        return x*y
    
    left_x=x//10**(m)
    right_x=x%10**(m)
    left_y=y//10**(m)
    right_y=y%10**(m)

    first=karatsuba(left_x,left_y)
    second=karatsuba(right_x,right_y)
    third=karatsuba(left_x+right_x,left_y+right_y)-first-second

    return 10**(2*m)*first+10**(m)*third+second

def test():
    x=3141592653589793238462643383279502884197169399375105820974944592
    y=2718281828459045235360287471352662497757247093699959574966967627

    assert karatsuba(x,y)==x*y

    print('Passed!')

test()