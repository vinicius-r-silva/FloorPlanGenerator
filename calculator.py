
import math   


def factorial(x):
    if x == 1:
        return x;

    return x*factorial(x-1);

def NConnections(n):
    exponent = (n*2) - 2;

    res = 1;
    for item in range(exponent):
        res = res * 4;

    return res;

def qtdRoomSizes(Hmin, Wmin, Hmax, Wmax, step):
    diffH = Hmax - Hmin
    diffW = Wmax - Wmin

    nH = math.ceil(diffH / step) + 1
    nW = math.ceil(diffW / step) + 1
    return nH * nW

def qtdRooms(n):
    res = 1
    for i in range(n):
        res = res * qtdRoomSizes(1, 3, 2, 5, 0.5)

    return res
     

def numberOfComb(n):
    f = factorial(n)
    c = NConnections(n)
    return f*c*pow(2, n)
    
print('numberOfComb')
print("1: ", numberOfComb(1)) 
print("2: ", numberOfComb(2)) 
print("3: ", numberOfComb(3)) 
print("4: ", numberOfComb(4)) 
print("5: ", numberOfComb(5)) 
print("6: ", numberOfComb(6)) 
print("7: ", numberOfComb(7)) 
print("8: ", numberOfComb(8)) 
print("9: ", numberOfComb(9)) 
print("10: ", numberOfComb(10)) 
print("11: ", numberOfComb(11)) 
print("12: ", numberOfComb(12)) 
print("13: ", numberOfComb(13)) 
print("14: ", numberOfComb(14)) 
print("15: ", numberOfComb(15)) 
print("\n\n")


print('qtdRooms')
print("1: ", qtdRooms(1)) 
print("2: ", qtdRooms(2)) 
print("3: ", qtdRooms(3)) 
print("4: ", qtdRooms(4)) 
print("5: ", qtdRooms(5)) 
print("6: ", qtdRooms(6)) 
print("7: ", qtdRooms(7)) 
print("8: ", qtdRooms(8)) 
print("9: ", qtdRooms(9)) 
print("10: ", qtdRooms(10)) 
print("11: ", qtdRooms(11)) 
print("12: ", qtdRooms(12)) 
print("13: ", qtdRooms(13)) 
print("14: ", qtdRooms(14)) 
print("15: ", qtdRooms(15)) 
print("\n\n")

print('numberOfComb * qtdRooms')
print("1: ", numberOfComb(1) * qtdRooms(1) )
print("2: ", numberOfComb(2) * qtdRooms(2) ) # 2 * 16 * 4
print("3: ", numberOfComb(3) * qtdRooms(3) ) # 6 * 256 * 8
print("4: ", numberOfComb(4) * qtdRooms(4) ) # 24 * 4096 * 16
print("5: ", numberOfComb(5) * qtdRooms(5) ) #
print("6: ", numberOfComb(6) * qtdRooms(6) )
print("7: ", numberOfComb(7) * qtdRooms(7) )
print("8: ", numberOfComb(8) * qtdRooms(8) )
print("9: ", numberOfComb(9) * qtdRooms(9) )
print("10: ", numberOfComb(10) * qtdRooms(10) )
print("11: ", numberOfComb(11) * qtdRooms(11) )
print("12: ", numberOfComb(12) * qtdRooms(12) )
print("13: ", numberOfComb(13) * qtdRooms(13) )
print("14: ", numberOfComb(14) * qtdRooms(14) )
print("15: ", numberOfComb(15) * qtdRooms(15) )
print("\n\n")