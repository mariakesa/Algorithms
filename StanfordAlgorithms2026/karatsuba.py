first = 1234
second = 5678

def karatsuba(first, second):
    n = max(len(str(first)), len(str(second)))
    n_halved = n // 2

    if n <= 2:
        return first * second

    a = first // (10 ** n_halved)
    b = first % (10 ** n_halved)
    c = second // (10 ** n_halved)
    d = second % (10 ** n_halved)

    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    bc_ad = karatsuba(a + b, c + d) - ac - bd

    return (10 ** (2 * n_halved)) * ac + (10 ** n_halved) * bc_ad + bd


if __name__ == "__main__":
    print(karatsuba(first, second))
    print(first * second)