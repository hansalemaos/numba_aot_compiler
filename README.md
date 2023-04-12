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



