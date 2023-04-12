# A function to facilitate the Ahead-of-time compilation with Numba 

### pip install numba-aot-compiler


Provides a simple way to use Ahead-of-Time (AOT) compilation with Numba
https://numba.readthedocs.io/en/stable/reference/aot-compilation.html

## How does it work? 


### First, import the compnumba function:


```python
from numba_aot_compiler import compnumba
```


### Next, write a numba-compatible function, such as:


```python
import numpy as np
from numba import uint8, uint16

def search_colors(r, g, b, rgb, divider):
    res = np.zeros(b.shape, dtype=np.uint16)
    res2 = np.zeros(b.shape, dtype=np.uint16)
    endxy = np.array([0], dtype=np.uint16)
    zaehler = 0
    for i in range(r.shape[0]):
        if r[i] == rgb[0] and g[i] == rgb[1] and b[i] == rgb[2]:
            dvquot, dvrem = divmod(i, divider)
            res[zaehler] = dvquot
            res2[zaehler] = dvrem
            endxy[0] = zaehler
            zaehler = zaehler + 1
    results = np.dstack((res[: endxy[0]], res2[: endxy[0]]))
    return results
```



### Then, compile the function ...
 ... using compnumba, providing the function to be compiled (fu), the desired name of the compiled function (funcname), the name of the file to be generated (file), the folder in which the file should be saved (folder), the function signature (signature), and any other relevant parameters for the compiler:

```python
compi2 = compnumba(
    fu=search_colors,
    funcname="search_colors_fu",   
    file="search_colors_file",
    folder=r"numbatesting",
    signature=(uint8[:], uint8[:], uint8[:], uint8[:], uint16), 
    parallel=True, 
    fastmath=True,
    nogil=True,
    # you can pass *args/**kwargs for more compiler options
)
```


### Finally, import the compiled function and use it as desired:


```python
from numbatesting import search_colors_fu
import cv2
import time
import numpy as np
pic = cv2.imread(r"pexels-alex-andrews-2295744.jpg") # https://www.pexels.com/pt-br/foto/foto-da-raposa-sentada-no-chao-2295744/
rgb_ = np.array([66, 71, 69],dtype=np.uint8)
r = pic[..., 0].flatten()
g = pic[..., 1].flatten()
b = pic[..., 2].flatten()
divider =np.uint16(pic.shape[1])
qq=search_colors_fu(r,g,b,rgb_,divider)
print(qq)
```

#### Do a benchmark to compare the performance of the compiled function with other functions/methods, such as numpy: 

```python
# %timeit np.where((pic[..., 0]==66)&(pic[..., 1]==71)&(pic[..., 2]==69))
# 161 ms ± 1.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# %timeit qq=search_colors_fu(r,g,b,rgb_,divider)
# 70.4 ms ± 484 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

### More (not organized) examples

```python


import numba
import numpy as np
from numba import uint32, float64, prange, types, uint8, uint16
from numba_aot_compiler import compnumba
from numba.typed import Dict


def go_fast(a):
    no = 1
    for r in range(10):
        a[r] += no
    return a


def prange_wrong_result(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in prange(n):
        y[:] += x[i]

    return y


def test(x, a, b):
    for aa, bb in zip(a, b):
        x = x.replace(aa, bb)
    return x


def square_list(n):
    lili = numba.typed.List()
    [lili.append(x**2) for x in n]
    return lili


def foo(x):
    return [[i for i in range(n)] for n in range(x)]


float_array = float64[:]


def foxo():
    # Make dictionary
    d = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    # Fill the dictionary
    d["posx"] = np.arange(3).astype(np.float64)
    d["posy"] = np.arange(3, 6).astype(np.float64)
    return d


def g(r, g, b, rgb, res, res2, endxy, divider):
    zaehler = 0
    for i in range(r.shape[0]):
        if r[i] == rgb[0] and g[i] == rgb[1] and b[i] == rgb[2]:
            dvquot, dvrem = divmod(i, divider)
            res[zaehler] = dvquot
            res2[zaehler] = dvrem
            endxy[0] = zaehler
            zaehler = zaehler + 1


compilethose = True
if compilethose:
    compi = compnumba(
        fu=go_fast,
        funcname="gofastfu",
        file="gofastfile",
        folder=r"numbatestcomp",
        signature=(uint32[:](uint32[:])),
        parallel=True,
        fastmath=True,
    )
    # print(compi)
    # exec(compi)
    compi = compnumba(
        fu=prange_wrong_result,
        funcname="prange_wrong_resultfu",
        file="prange_wrong_resultfile",
        folder=r"numbatestcomp",
        signature=(float64[:](float64[:])),
        parallel=False,
        fastmath=True,
    )
    # print(compi)
    # a=np.array([1,2,34,4,45,5,56,67,7,87,123],dtype=np.float64)
    # exec(compi)
    # a1=prange_wrong_resultfu(a)
    # print(a1)
    compi = compnumba(
        fu=test,
        funcname="testfu",
        file="testfile",
        folder=r"numbatestcomp",
        signature=((types.unicode_type, types.unicode_type, types.unicode_type)),
        parallel=True,
        fastmath=True,
    )
    # print(compi)
    # a='hallo'
    # exec(compi)
    # a1=testfu(a,ascii_lowercase,ascii_uppercase)
    # print(a1)

    compi = compnumba(
        fu=square_list,
        funcname="square_list_fu",
        file="square_list_file",
        folder=r"numbatestcomp",
        signature=((float64[:],)),
        parallel=True,
        fastmath=True,
        nogil=True,
    )
    # print(compi)
    # a=np.arange(1,2000000)
    # py_listx = a.astype(np.float64)
    # exec(compi)
    # a1=square_list_fu(py_listx.copy())
    # print(a1)

    compi = compnumba(
        fu=foo,
        funcname="foo_fu",
        file="foo_file",
        folder=r"numbatestcomp",
        signature=((uint32,)),
        parallel=True,
        fastmath=True,
        nogil=False,
    )
    # print(compi)
    # a=np.uint32(100)
    # exec(compi)
    # a1=foo_fu(a)
    # print(a1)

    compi = compnumba(
        fu=foxo,
        funcname="foxo_fu",
        file="foxo_file",
        folder=r"numbatestcomp",
        signature=(()),
        parallel=True,
        fastmath=True,
        nogil=False,
    )
    # print(compi)
    ##a = float_arrayx = types.float64[:]
    # exec(compi)
    # a1 = foxo_fu()
    # print(a1)
```




