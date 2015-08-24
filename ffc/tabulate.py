import collections
import numpy

from pymbolic.primitives import Expression, Variable, Quotient


def operands_and_reconstruct(expr):
    if isinstance(expr, Variable):
        return (), None
    elif isinstance(expr, Quotient):
        return ((expr.numerator, expr.denominator),
                lambda children: Quotient(*children))
    elif isinstance(expr, Expression):
        return (expr.children,
                lambda children: type(expr)(tuple(children)))
    else:
        # e.g. floating-point numbers
        return (), None


def static_single_assignment(expr, regs, code, prefix=None):
    if prefix is None:
        prefix = "r"

    def create_reg():
        return Variable('%s%d' % (prefix, len(regs)))

    def recurse(e):
        ops, reconstruct = operands_and_reconstruct(e)
        if len(ops) == 0:
            return e
        elif e in regs:
            return regs[e]
        else:
            s = reconstruct(map(recurse, ops))
            r = create_reg()
            regs[e] = r
            code[r] = s
            return r

    return recurse(expr)


def tabulate(fiat_element, order, reference_coordinates, prefix=None):
    points = numpy.array([map(Variable, reference_coordinates)])
    tabulations = fiat_element.tabulate(order, points)

    regs = {}
    code = collections.OrderedDict()

    refs = {}
    for alpha, t in tabulations.iteritems():
        u = numpy.zeros(t.shape[:-1], dtype=object)
        t_flat = t.reshape(-1)
        u_flat = u.reshape(-1)
        for i, e in enumerate(t_flat):
            u_flat[i] = static_single_assignment(e, regs, code, prefix=prefix)

        refs[alpha] = u

    return refs, list(code.iteritems())
