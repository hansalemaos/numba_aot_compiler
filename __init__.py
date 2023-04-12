import os
import sys
from deepcopyall import deepcopy
from touchtouch import touch

mod = sys.modules[__name__]
from ordered_set import OrderedSet


def compnumba(
    fu,
    funcname,
    file,
    folder,
    signature,
    parallel=True,
    fastmath=True,
    nogil=False,
    *args,
    **kwargs,
):
    """
    Compile a Python function with Numba and export it as a shared library.

    Args:
        fu: The Python function to be compiled.
        funcname: The name of the exported function.
        file: The name of the shared library file to be generated.
        folder: The folder where the shared library file will be saved.
        signature: The signature of the function to be compiled.
        parallel: Whether to enable parallel execution (default True).
        fastmath: Whether to enable fast math optimizations (default True).
        nogil: Whether to release the global interpreter lock (default False).
        *args: Positional arguments to be passed to the compiled function.
        **kwargs: Keyword arguments to be passed to the compiled function.

    Returns:
        A string representing the import statement for the compiled function.

    Raises:
        NumbaError: If there is an error during the compilation process.

    """
    folderpure = folder
    folder = os.path.normpath(os.path.join(os.path.dirname(sys.executable), folder))
    setattr(sys.modules[__name__], "myfux", deepcopy(fu))
    setattr(sys.modules[__name__], "myfu", fu)

    from numba.core import sigutils, typing
    from numba.pycc import CC
    import numba as nb
    from numba.pycc.compiler import ExportEntry

    modu = file if not file.lower().endswith(".pyd") else file[:-4]
    outputfolder = folder
    outputfolderinit = os.path.join(outputfolder, "__init__.py")

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    touch(outputfolderinit)
    with open(outputfolderinit, mode="r", encoding="utf-8") as f:
        data = f.read()
    data = (
        "\n".join(list(OrderedSet([q for q in data.splitlines() if q.strip()])))
        + f"\nfrom .{file} import {funcname}\n"
    )
    with open(outputfolderinit, mode="w", encoding="utf-8") as f:
        f.write(data.strip())
    cc = CC(modu)
    cc.verbose = True
    cc.output_dir = outputfolder
    cc.output_file = f"{modu}.pyd"

    fn_args, fn_retty = sigutils.normalize_signature(signature)
    sig = typing.signature(fn_retty, *fn_args)

    entry = ExportEntry(
        funcname,
        sig,
        nb.njit(
            mod.myfu, parallel=parallel, fastmath=fastmath, nogil=nogil, *args, **kwargs
        ),
    )
    cc._exported_functions[funcname] = entry
    cc.compile()
    setattr(sys.modules[__name__], "myfux", None)
    setattr(sys.modules[__name__], "myfu", None)
    return f"from {folderpure} import {funcname}"
