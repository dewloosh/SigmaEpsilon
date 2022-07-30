# -*- coding: utf-8 -*-
from ..config import __hassympy__, __haspylatex__

if __haspylatex__:
    from pylatex import NoEscape, Package


def append_preamble(doc):
    r"""
    Tools related to displaying math. It's a bit like the numpy of latex, must
    have stuff.
    """
    doc.packages.append(Package('amsmath'))

    r"""
    Sympy uses the 'operatorname' command frequently to print symbols.
    """
    doc.packages.append(Package('amsopn'))

    r"""
    To automatically break long equations into multiple lines.
    """
    doc.packages.append(Package('breqn'))

    r"""
    mathtools provides us with the \coloneqq command, for defining equality
    symbol ':='
    """
    doc.packages.append(Package('mathtools'))

    r"""
    Misc
    """
    doc.packages.append(Package('enumitem'))  # to customize enumerations
    doc.packages.append(Package('xcolor'))  # colors
    doc.packages.append(Package('lmodern'))  # high quality fonts
    
    return doc


def expr_to_ltx(lhs, rhs, *args, env='{equation}', sign='=',
                dfrac=False, pre=None, post=None, **kwargs):
    if dfrac:
        lhs = lhs.replace('frac', 'dfrac')
        rhs = rhs.replace('frac', 'dfrac')
    if isinstance(pre, str):
        lhs = ' '.join([pre, lhs])
    if isinstance(post, str):
        rhs = ' '.join([rhs, post])
    return NoEscape(
        r"""
        \begin{env}
            {lhs} {sign} {rhs}
        \end{env}
        """.format(env = env,
                   lhs = lhs,
                   sign = sign,
                   rhs = rhs
                   )
        )


def expr_to_ltx_breqn(lhs, rhs, *args, env='{dmath}', **kwargs):
    return expr_to_ltx(lhs, rhs, *args, env=env, **kwargs)


