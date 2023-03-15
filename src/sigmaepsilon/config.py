try:
    import pypardiso

    __haspardiso__ = True
except Exception:
    __haspardiso__ = False


try:
    import sympy

    __hassympy__ = True
except Exception:
    __hassympy__ = False


try:
    import pylatex

    __haspylatex__ = True
except Exception:
    __haspylatex__ = False
