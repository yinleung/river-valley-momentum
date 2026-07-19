import sympy as sp

eta, lam, mu, c, s2 = sp.symbols('eta lambda mu c sigma2', positive=True)

# Curved valley, linear floor f=cx, quadratic river phi = mu/2 x^2, beta=0 (plain SGD).
# State z=(d, x). Derived one-step:
# d' = (1-eta*lam*(1+c^2)) d + c*eta*mu x - eta*(xi_y - c*xi_x)
# x' = eta*c*lam d + (1-eta*mu) x - eta*xi_x
lloc = lam*(1+c**2)
A = sp.Matrix([[1-eta*lloc, c*eta*mu],
               [eta*c*lam, 1-eta*mu]])
# noise vector n = -eta*(xi_y - c xi_x, xi_x); xi iid var s2
# Cov(n) = eta^2 * s2 * [[1+c^2, c],[c, 1]]
Q = eta**2*s2*sp.Matrix([[1+c**2, c],[c, 1]])

# Solve discrete Lyapunov P = A P A^T + Q via vectorization
P11, P12, P22 = sp.symbols('P11 P12 P22')
P = sp.Matrix([[P11, P12],[P12, P22]])
Eqs = sp.Matrix(A*P*A.T + Q - P)
sol = sp.solve([Eqs[0,0], Eqs[0,1], Eqs[1,1]], [P11, P12, P22], dict=True)[0]
var_d = sp.simplify(sol[P11])
print("Var(d_inf) exact (beta=0):"); sp.pprint(sp.factor(var_d))

# Reviewer's claimed formula
rev = eta*s2*(2-eta*mu) / (lam*(4 - 2*eta*lam*(1+c**2) - 2*eta*mu + eta**2*lam*mu))
print("\nreviewer - exact simplifies to:", sp.simplify(var_d - rev))

# Paper's Prop 5(b) formula at beta=0 (T_eff=1)
paper = eta*s2/(lam*(2-eta*lloc))
print("paper  - exact simplifies to:", sp.simplify(sp.factor(var_d - paper)))

# mu->0 limit of exact
print("mu->0 limit of exact:", sp.simplify(sp.limit(var_d, mu, 0)), " vs paper:", sp.simplify(paper))

# numeric spot cell
subs = {eta: sp.Rational(1,10), lam: 10, mu: sp.Rational(1,10), c: sp.Rational(1,2), s2: 4}
print("\nnumeric cell eta=.1 lam=10 mu=.1 c=.5 s2=4:")
print("  exact   =", float(var_d.subs(subs)))
print("  reviewer=", float(rev.subs(subs)))
print("  paper   =", float(paper.subs(subs)))
import sympy as sp
eta, lam, mu, c, s2 = sp.symbols('eta lambda mu c sigma2', positive=True)
lloc = lam*(1+c**2)
A = sp.Matrix([[1-eta*lloc, c*eta*mu],[eta*c*lam, 1-eta*mu]])
rev = eta*s2*(2-eta*mu) / (lam*(4 - 2*eta*lam*(1+c**2) - 2*eta*mu + eta**2*lam*mu))

def var_d(Q):
    P11,P12,P22 = sp.symbols('P11 P12 P22')
    P = sp.Matrix([[P11,P12],[P12,P22]])
    E = sp.Matrix(A*P*A.T + Q - P)
    sol = sp.solve([E[0,0],E[0,1],E[1,1]],[P11,P12,P22],dict=True)[0]
    return sp.simplify(sol[P11])

# variant A: noise only on the y-gradient: d gets -eta*xi, x unaffected
QA = eta**2*s2*sp.Matrix([[1,0],[0,0]])
# variant B: noise only on hill-gradient but entering both coords? (x update has -eta*(mu x - c lam d + 0))
# variant C: iid noise on x and y gradient, but d-noise NOT scaled by (1+c^2): Q diag
QC = eta**2*s2*sp.Matrix([[1,0],[0,1]])
for name,Q in [("y-only",QA),("diag-iid",QC)]:
    v = var_d(Q)
    print(name, "matches reviewer:", sp.simplify(v-rev)==0)
    if sp.simplify(v-rev)!=0:
        print("   diff:", sp.simplify(sp.factor(v-rev)))
import sympy as sp
eta, lam, mu, c, s2 = sp.symbols('eta lambda mu c sigma2', positive=True)
lloc = lam*(1+c**2)
A = sp.Matrix([[1-eta*lloc, c*eta*mu],[eta*c*lam, 1-eta*mu]])
rev = eta*s2*(2-eta*mu) / (lam*(4 - 2*eta*lam*(1+c**2) - 2*eta*mu + eta**2*lam*mu))
# scalar noise xi on the hill gradient (lam*d -> lam*d + xi), chain rule into x:
# d' = (1-eta*lloc) d + c*eta*mu x - eta*(1+c^2) xi ;  x' = eta*c*lam d + (1-eta*mu) x + eta*c*xi
n = sp.Matrix([-eta*(1+c**2), eta*c])
Q = s2*n*n.T
P11,P12,P22 = sp.symbols('P11 P12 P22')
P = sp.Matrix([[P11,P12],[P12,P22]])
E = sp.Matrix(A*P*A.T + Q - P)
sol = sp.solve([E[0,0],E[0,1],E[1,1]],[P11,P12,P22],dict=True)[0]
v = sp.simplify(sol[P11])
print("hill-scalar-noise matches reviewer:", sp.simplify(v-rev)==0)
print("diff:", sp.simplify(sp.factor(v-rev)))
print("mu->0 limit:", sp.simplify(sp.limit(v, mu, 0)))
