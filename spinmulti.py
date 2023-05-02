import sympy as sp 
import numpy as np

# calculate T*T
phi1, phi2, phi1c, phi2c, phi1p, phi1pc, l, lp = sp.symbols('p1 p2 p1c p2c p1p p1pc l b')
I = sp.symbols('i')

s1 = sp.Array([1, -l*lp*phi2, lp*phi1p, -I*lp*phi1p, l*phi1c, I*l*phi1c]) #original
s1m = sp.Array([1, l*lp*phi2, lp*phi1p, -I*lp*phi1p, -l*phi1c, -I*l*phi1c]) #only k negative

s2 = sp.Array([1, -l*lp*phi2, -lp*phi1p, I*lp*phi1p, -l*phi1c, -I*l*phi1c]) #negative
s3 = sp.Array([1, -l*lp*phi2c, l*phi1, -I*l*phi1, lp*phi1pc, I*lp*phi1pc]) #prime
s4 = sp.Array([1, -l*lp*phi2c, -l*phi1, I*l*phi1, -lp*phi1pc, -I*lp*phi1pc]) #prime+negative

kmk = sp.tensorproduct(s1,s2)
kkp = sp.tensorproduct(s3,s1)
kkpn = sp.tensorproduct(s1,s4)
mkmk = sp.tensorproduct(s1m,s2)
mkkp = sp.tensorproduct(s1m,s3)
mkkpn = sp.tensorproduct(s1m,s4)

new1 = sp.tensorproduct(s2, s2)
new2 = sp.tensorproduct(s2, s4)

for element in kmk:
    print(element)
# print('\n############################################################################\n')
# for element in kkp:
#     print(element)
# print('\n############################################################################\n')
# for element in mkkpn:
#     print(element)
# print('\n############################################################################\n')
# for element in mkmk:
#     print(element)
# print('\n############################################################################\n')
# for element in mkkp:
#     print(element)
# print('\n############################################################################\n')
# for element in mkkpn:
#     print(element)
# for element in new1:
#     print(element)
# print('\n############################################################################\n')
# for element in new2:
#     print(element)
