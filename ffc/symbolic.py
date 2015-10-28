import collections
import numpy
import sympy

from ffc.cpp import format


def operands_and_reconstruct(expr):
    if isinstance(expr, sympy.Expr):
        return (expr.args,
                lambda children: expr.func(*children))
    else:
        # e.g. floating-point numbers
        return (), None


class SSATransformer(object):
    def __init__(self, prefix=None):
        self._regs = {}
        self._code = collections.OrderedDict()
        self._prefix = prefix or "r"

    def _new_reg(self):
        return sympy.Symbol('%s%d' % (self._prefix, len(self._regs)))

    def __call__(self, e):
        ops, reconstruct = operands_and_reconstruct(e)
        if len(ops) == 0:
            return e
        elif e in self._regs:
            return self._regs[e]
        else:
            s = reconstruct(map(lambda e_: self(e_), ops))
            r = self._new_reg()
            self._regs[e] = r
            self._code[r] = s
            return r

    @property
    def code(self):
        return self._code.items()


def rounding(expr):
    eps = format["epsilon"]

    if isinstance(expr, (float, sympy.numbers.Float)):
        v = float(expr)
        if abs(v - round(v, 1)) < eps:
            return round(v, 1)
    elif isinstance(expr, sympy.Expr):
        if expr.args:
            return expr.func(*map(rounding, expr.args))

    return expr


def ssa_arrays(args, prefix=None):
    transformer = SSATransformer(prefix=prefix)

    refs = []
    for arg in args:
        ref = numpy.zeros_like(arg, dtype=object)
        arg_flat = arg.reshape(-1)
        ref_flat = ref.reshape(-1)
        for i, e in enumerate(arg_flat):
            ref_flat[i] = transformer(rounding(e))
        refs.append(ref)

    return transformer.code, refs


class _CPrinter(sympy.printing.StrPrinter):
    """sympy.printing.StrPrinter uses a Pythonic syntax which is invalid in C.
    This subclass replaces the printing of power with C compatible code."""

    def _print_Pow(self, expr, rational=False):
        # WARNING: Code mostly copied from sympy source code!
        from sympy.core import S
        from sympy.printing.precedence import precedence

        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that will
                # match -0.5, which we don't want.
                return "1/sqrt(%s)" % self._print(expr.base)
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '1/%s' % self.parenthesize(expr.base, PREC)

        e = self.parenthesize(expr.exp, PREC)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                e = e[1:-1]

        # Changes below this line!
        if e == "2":
            return '{0}*{0}'.format(self.parenthesize(expr.base, PREC))
        elif e == "3":
            return '{0}*{0}*{0}'.format(self.parenthesize(expr.base, PREC))
        else:
            return 'pow(%s,%s)' % (self.parenthesize(expr.base, PREC), e)


def c_print(expr):
    printer = _CPrinter(dict(order=None))
    return printer.doprint(expr)
