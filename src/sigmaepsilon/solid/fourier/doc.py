# -*- coding: utf-8 -*-

from ..config import __hassympy__, __haspylatex__

if __hassympy__ and __haspylatex__:
    import sympy as sy
    from sympy import symbols, Function, diff, Matrix, MatMul, \
        integrate, Symbol, sin, cos, pi, simplify
    import pylatex as pltx
    from pylatex import Section, Subsection, Command, NoEscape, NewPage

from .latex import expr_to_ltx, expr_to_ltx_breqn, append_preamble
from .symtools import inv_sym_3x3


def append_mindlin_theory(doc, *args, glossary:dict=None, signs:dict=None, **kwargs):
    
    if glossary is None:
        glossary = {
            'UX' : 'u_0',
            'UY' : 'v_0',
            'UZ' : 'w_0',
            'ROTX' : '\\theta_y',
            'ROTY' : '\\theta_x',
            'ROTZ' : '\\theta_z',
        }
        
    if signs is None:
        glossary = {
            'UX' : 1,
            'UY' : 1,
            'UZ' : 1,
            'ROTX' : 1,
            'ROTY' : 1,
            'ROTZ' : 1,
        }

    __hasreqs__ = __hassympy__ and __haspylatex__
    assert __hasreqs__, "You must have `sympy` and `pylatex` for this."

    with doc.create(Section('Theoretical background')):

        # coordinates in a suitable vector space
        x, y, z = symbols('x y z', real=True)

        # kinematic variables
        UZ = Function(glossary['UZ'])(x, y)
        ROTX = Function(glossary['ROTY'])(x, y)
        ROTY = Function(glossary['ROTX'])(x, y)

        # sign modifiers for kinematic variables
        sign_w, sign_u, sign_v, = symbols('\\beta_w, \\beta_u, \\beta_v', real=True)

        # displacement field
        with doc.create(Subsection('Displacement Field and Boundary Conditions')):
            doc.append(NoEscape(
                r"""
                \noindent
                The Mindlin-Reissner kinematical model poses the following
                assumptions on the displacement field:
                """))
            doc.append(NoEscape(
                r"""
                \begin{enumerate}[label*=\protect\fbox{MR\arabic{enumi}},
                itemindent=5em]
                \item   normals to the midsurface remain normal
                \item   the plate thickness does not change during deformation
                \item   normal stress through the thickness is ignored
                \end{enumerate}
                """))

            """
            u = -z * beta_u * thx
            v = -z * beta_v * thy
            w = beta_w * w0
            """
            u = -z * ROTX
            v = -z * ROTY
            w = UZ
            doc.append(expr_to_ltx(r'u(x, y, z)', sy.latex(u)))
            doc.append(expr_to_ltx(r'v(x, y, z)', sy.latex(v)))
            doc.append(expr_to_ltx(r'w(x, y, z)', sy.latex(w)))

        # engineering strains
        with doc.create(Subsection('Strain - Displacement Equations')):
            exx = diff(u, x)
            eyy = diff(v, y)
            exy = diff(u, y) + diff(v, x)
            exz = diff(w, x) + diff(u, z)
            eyz = diff(w, y) + diff(v, z)
            doc.append(expr_to_ltx(r'\varepsilon_x(x, y, z)', sy.latex(exx)))
            doc.append(expr_to_ltx(r'\varepsilon_y(x, y, z)', sy.latex(eyy)))
            doc.append(expr_to_ltx(r'\gamma_{xy}(x, y, z)', sy.latex(exy)))
            doc.append(expr_to_ltx(r'\gamma_{xz}(x, y, z)', sy.latex(exz)))
            doc.append(expr_to_ltx(r'\gamma_{yz}(x, y, z)', sy.latex(eyz)))

            doc.append('Curvatures are defined as')
            kxx = diff(exx, z)
            kyy = diff(eyy, z)
            kxy = diff(exy, z)
            doc.append(
                expr_to_ltx(r"""\kappa_x(x, y) \coloneqq \frac{\partial
                \varepsilon_x}{\partial z}""", sy.latex(kxx)))
            doc.append(
                expr_to_ltx(r"""\kappa_y(x, y) \coloneqq \frac{\partial
                \varepsilon_y}{\partial z}""", sy.latex(kyy)))
            doc.append(
                expr_to_ltx(r"""\kappa_{xy}(x, y) \coloneqq \frac{\partial
                \gamma_{xy}}{\partial z}""", sy.latex(kxy)))

        # material model and stress resultants
        with doc.create(Subsection('Stress Resultants')):
            # strains '+' Hooke model -> stresses
            C11, C22, C12, C16, C26, C66 = C126ij = \
                symbols('C_11 C_22 C_12 C_16 C_26 C_66', real=True)
            C126 = sy.Matrix([[C11, C12, 0], [C12, C22, 0], [0, 0, C66]])
            sxx, syy, sxy = MatMul(C126, sy.Matrix([exx, eyy, exy])).doit()
            C44, C55 = C45ij = symbols('C_44 C_55', real=True)
            C45 = sy.Matrix([[C44, 0], [0, C55]])
            syz, sxz = MatMul(C45, sy.Matrix([eyz, exz])).doit()

            # integrate through the thickness
            h = Symbol('h', real=True)  # thickness
            def Int(expr): return integrate(expr, (z, -h/2, h/2))
            mx, my, mxy = M = Matrix([Int(s * z) for s in [sxx, syy, sxy]])
            vx, vy = V = Matrix([Int(s) for s in [sxz, syz]])

            D11, D22, D12, D66 = Dij = symbols(
                'D_11 D_22 D_12 D_66', real=True)
            S44, S55, = Sij = symbols('S_44 S_55', real=True)
            #
            mx = mx.simplify().expand()
            cD11 = mx.coeff(C11 * h**3 / 12)
            cD12 = mx.coeff(C12 * h**3 / 12)
            mx = D11 * cD11 + D12 * cD12
            #
            my = my.simplify().expand()
            cD22 = my.coeff(C22 * h**3 / 12)
            cD21 = my.coeff(C12 * h**3 / 12)
            my = D22 * cD22 + D12 * cD21
            #
            mxy = mxy.simplify().expand()
            cD66 = mxy.coeff(C66 * h**3 / 12)
            mxy = D66 * cD66
            #
            vx = vx.simplify().expand()
            cS55 = vx.coeff(C55 * h)
            vx = S55 * cS55
            #
            vy = vy.simplify().expand()
            cS44 = vy.coeff(C44 * h)
            vy = S44 * cS44

            # sign modifiers for internal forces
            """
            inds = ['x', 'y', 'xy', 'xz', 'yz']
            syms = " ".join(["\\alpha_{}".format('{' + i + '}') for i in inds])
            alpha = symbols(sims, real = True)
            mx *= alpha[0]
            my *= alpha[1]
            mxy *= alpha[2]
            vx *= alpha[3]
            vy *= alpha[4]
            """

            mx = mx.simplify().expand()
            my = my.simplify().expand()
            mxy = mxy.simplify().expand()
            vx = vx.simplify().expand()
            vy = vy.simplify().expand()
            doc.append(expr_to_ltx(r'm_x(x, y)', sy.latex(mx)))
            doc.append(expr_to_ltx(r'm_y(x, y)', sy.latex(my)))
            doc.append(expr_to_ltx(r'm_{xy}(x, y)', sy.latex(mxy)))
            doc.append(expr_to_ltx(r'v_x(x, y)', sy.latex(vx)))
            doc.append(expr_to_ltx(r'v_y(x, y)', sy.latex(vy)))

        # Equilibrium equations. The signs of the equations
        # is selected such, that the coefficients of the load functions on the
        # right-hand sides admit positive signs according to global axes.
        ###
        # moment around global X
        lhs_mx = simplify(-diff(mxy, x) - diff(my, y) + vy).expand()
        # moment around global Y
        lhs_my = simplify(diff(mxy, y) + diff(mx, x) - vx).expand()
        # vertical equilibrium
        lhs_fz = simplify(diff(vx, x) + diff(vy, y)).expand()
        with doc.create(Subsection('Equilibrium Equations')):
            doc.append(NoEscape(r'Equilibrium of moments around global $x$ :'))
            doc.append(expr_to_ltx_breqn(sy.latex(lhs_mx), 'm_x(x, y)'))
            doc.append(NoEscape(r'Equilibrium of moments around global $y$ :'))
            doc.append(expr_to_ltx_breqn(sy.latex(lhs_my), 'm_y(x, y)'))
            doc.append(NoEscape(r'Equilibrium of forces along global $z$ :'))
            doc.append(expr_to_ltx_breqn(sy.latex(lhs_fz), 'f_z(x, y)'))

        # Boundary Conditions and Trial Solution
        with doc.create(Subsection('Boundary Conditions and Trial Solution')):
            mn = m, n = symbols('m n', integer=True, positive=True)
            coeffs = Amn, Bmn, Cmn = symbols('A_{mn} B_{mn} C_{mn}', real=True)
            shape = Lx, Ly = symbols('L_x, L_y', real=True)
            Sm, Sn = sin(m * pi * x / Lx), sin(n * pi * y / Ly)
            Cm, Cn = cos(m * pi * x / Lx), cos(n * pi * y / Ly)
            w0_trial = Cmn * Sm * Sn
            thx_trial = Amn * Cm * Sn
            thy_trial = Bmn * Sm * Cn
            doc.append(NoEscape(
                r"""
                All of the boundary conditions are of the essential type, and
                are satisfied a priori by the proper selection of approximation
                functions. Thus, the unknown field functions are sought in the
                form
                """))
            sumltx = r'\sum_{m, n} '
            doc.append(expr_to_ltx(sy.latex(UZ), sumltx + sy.latex(w0_trial)))
            doc.append(expr_to_ltx(sy.latex(ROTX),
                       sumltx + sy.latex(thx_trial)))
            doc.append(expr_to_ltx(sy.latex(ROTY),
                       sumltx + sy.latex(thy_trial)))

            # substitute trial solution
            trial = {UZ: w0_trial, ROTX: thx_trial, ROTY: thy_trial}
            eq_mx_trial = lhs_mx.subs(trial).expand().doit() / (Sm * Cn)
            eq_mx_trial = eq_mx_trial.simplify().expand()
            eq_my_trial = lhs_my.subs(trial).expand().doit() / (Cm * Sn)
            eq_my_trial = eq_my_trial.simplify().expand()
            eq_fz_trial = lhs_fz.subs(trial).expand().doit() / (Sm * Sn)
            eq_fz_trial = eq_fz_trial.simplify().expand()
            doc.append(NoEscape(
                r"""
                \noindent
                Upon substitution of the trial solution into the vertical
                equilibrium equation we obtain the following equality
                """))
            mxltx = r"\left[" + sy.latex(eq_fz_trial.simplify()) + r"\right]" + \
                sy.latex(Sm * Sn)
            doc.append(expr_to_ltx_breqn(mxltx, 'f_z(x, y)'))
            doc.append(NoEscape(
                r"""
                \noindent
                It is clear, that if was possible to write the function
                of vertical forces $f_z(x, y)$ as a finite sum of purely sinusoidal
                functions, the trigonometric terms could be completely ruled out
                from the equation. After forming similar thoughts about the other
                two equations, the loads are sought as
                """))

            # coefficient matrix
            doc.append('Discrete equilibrium equation')
            P = sy.zeros(3, 3)
            P[0, :] = Matrix([eq_mx_trial.coeff(c) for c in coeffs]).T
            P[1, :] = Matrix([eq_my_trial.coeff(c) for c in coeffs]).T
            P[2, :] = Matrix([eq_fz_trial.coeff(c) for c in coeffs]).T
            doc.append(expr_to_ltx(r'\boldsymbol{P}', sy.latex(P), dfrac=True))
            detP, adjP = inv_sym_3x3(P, as_adj_det=True)
            detP = detP.simplify().expand()
            adjP.simplify()

        with doc.create(Subsection('Postprocessing')):
            mx = mx.subs(trial).expand().doit()
            mx = mx.simplify().expand()
            my = my.subs(trial).expand().doit()
            my = my.simplify().expand()
            mxy = mxy.subs(trial).expand().doit()
            mxy = mxy.simplify().expand()
            vx = vx.subs(trial).expand().doit()
            vx = vx.simplify().expand()
            vy = vy.subs(trial).expand().doit()
            vy = vy.simplify().expand()
            kxx = kxx.subs(trial).expand().doit()
            kxx = kxx.simplify().expand()
            kyy = kyy.subs(trial).expand().doit()
            kyy = kyy.simplify().expand()
            kxy = kxy.subs(trial).expand().doit()
            kxy = kxy.simplify().expand()

            exz = exz.subs(trial).expand().doit()
            exz = exz.simplify().expand()
            eyz = eyz.subs(trial).expand().doit()
            eyz = eyz.simplify().expand()

            doc.append(expr_to_ltx(r'm_x(x, y)', sy.latex(mx)))
            doc.append(expr_to_ltx(r'm_y(x, y)', sy.latex(my)))
            doc.append(expr_to_ltx(r'm_{xy}(x, y)', sy.latex(mxy)))
            doc.append(expr_to_ltx(r'v_x(x, y)', sy.latex(vx)))
            doc.append(expr_to_ltx(r'v_y(x, y)', sy.latex(vy)))
            doc.append(expr_to_ltx(r'\kappa_x(x, y)', sy.latex(kxx)))
            doc.append(expr_to_ltx(r'\kappa_y(x, y)', sy.latex(kyy)))
            doc.append(expr_to_ltx(r'\kappa_{xy}(x, y)', sy.latex(kxy)))
            doc.append(expr_to_ltx(r'\gamma_{xz}(x, y)', sy.latex(exz)))
            doc.append(expr_to_ltx(r'\gamma_{yz}(x, y)', sy.latex(eyz)))

        with doc.create(Subsection('Loads')):
            doc.append(NoEscape(
                r"""
                If the load functions are represented as suggested by the
                left hand-sides of the equilibrium equations, the solution of
                the system simplifies to the solution of a 3x3 algebraic equation
                system.
                """))

            # constans vertical load over a rectangular area
            q, w, h = symbols('q w h', real=True)
            xc, yc = symbols('x_c y_c', real=True)
            qmn = (4/(Lx*Ly)) * integrate(q * Sm * Sn, (x, xc - w/2, xc + w/2),
                                          (y, yc - h/2, yc + h/2))
            qmn = qmn.simplify().expand()
            doc.append(NoEscape(
                r"""
                \noindent
                \paragraph{Constans vertical load of intensity $q$ over a
                rectangular area}
                """))
            doc.append(expr_to_ltx(r'q_{mn}', sy.latex(qmn), post=','))
            # doc.append(NoEscape(r"\noindent"))
            doc.append(NoEscape(
                r"""
                where $w$ and $h$ denote the width and height, $x_c$ and $y_c$ the
                coordinates of the center point of the rectangle.
                """))

            # constans moment of intensity mx, around global axis X,
            # over a rectangular area with width w, height h and center (xc, yc)
            m_x, w, h = symbols('m_x w h', real=True)
            xc, yc = symbols('x_c y_c', real=True)
            qmn = (4/(Lx*Ly)) * integrate(m_x * Sm * Cn, (x, xc - w/2, xc + w/2),
                                          (y, yc - h/2, yc + h/2))
            qmn = qmn.simplify().expand()
            doc.append(NoEscape(
                r"""
                \noindent
                \paragraph{Constans moment around global axis $x$ of intensity
                $m_x$ over a rectangular area}
                """))
            doc.append(expr_to_ltx(r'q_{mn}', sy.latex(qmn), post=','))
            # doc.append(NoEscape(r"\noindent"))
            doc.append(NoEscape(
                r"""
                where $w$ and $h$ denote the width and height, $x_c$ and $y_c$ the
                coordinates of the center of the rectangle.
                """))

            # constant moment of intensity my, around global axis Y,
            # over a rectangular area with width w, height h and center (xc, yc)
            m_y, w, h = symbols('m_y w h', real=True)
            xc, yc = symbols('x_c y_c', real=True)
            qmn = (4/(Lx*Ly)) * integrate(m_y * Cm * Sn, (x, xc - w/2, xc + w/2),
                                          (y, yc - h/2, yc + h/2))
            qmn = qmn.simplify().expand()
            doc.append(NoEscape(
                r"""
                \noindent
                \paragraph{Constans moment around global axis $y$ of intensity
                $m_y$ over a rectangular area}
                """))
            doc.append(expr_to_ltx(r'q_{mn}', sy.latex(qmn), post=','))
            # doc.append(NoEscape(r"\noindent"))
            doc.append(NoEscape(
                r"""
                where $w$ and $h$ denote the width and height, $x_c$ and $y_c$ the
                coordinates of the center of the rectangle.
                """))
            
            return doc

    
if __name__ == '__main__':
    assert __hassympy__ and __haspylatex__
    from dewloosh.solid.fourier.latex import  append_preamble

    geometry_options = {
        "tmargin": "1.5cm",
        "lmargin": "1.5cm",
        "rmargin": "1.5cm"
    }
    
    doc = pltx.Document(geometry_options=geometry_options)

    doc = append_preamble(doc)

    title = "Calculation of rectangular, simply supported plates \
        resting on elastic foundation."
    doc.preamble.append(Command('title', title))
    doc.preamble.append(Command('author', 'Claude Louis Marie Henri Navier'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    doc.append(NoEscape(r'\tableofcontents'))
    doc.append(NewPage())
    
    doc = append_mindlin_theory(doc)
    
    doc.generate_pdf('f:\\navier2', clean_tex=False, compiler='pdfLaTeX')
