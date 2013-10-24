numpy-einsum
============

numpy.einsum simulation in written in Python

`einsum.c.src` is the C source file from `https://github.com/numpy`.

`einsum_py.py` contains equivalent code.  Its `myeinsum()` has a similar
API to `numpy.einsum` (though it does not use the keywords like `dtype`).

`test_einsum.py` is an adaptation of the `numpy` test file, using
`myeinsum`.  Currently it skips the initial `ValueError` tests, tests using
the `(op1, [0], op2, [0,1]...)` syntax, most `dtype` and `casting` tests,
and `b.base is a` tests.

The initial purpose of this project was to understand the issue behind
https://github.com/numpy/numpy/issues/2455,
http://stackoverflow.com/questions/16591696/ellipsis-broadcasting-in-numpy-einsum
and
http://comments.gmane.org/gmane.comp.python.numeric.general/53705

If `A` is 2d, and `B` is 1d, why is the 2nd correct, but not the first?

    einsum('ij...,j->ij...',A,B)
    einsum('ij...,j...->ij...',A,B)

Thus the primary focus was on how `einsum` parses the subscripts.  But to
run as many of the test cases as possible, I added the `nditer` loop as
suggested in https://github.com/numpy/numpy/issues/3142#issuecomment-14909173
as well as the strided view mechanism.  So it is also a good exercise in
using those features.

My tentative conclusion is that the `/* Middle or None broadcasting */`
case in `prepare_op_axes` is unnecessary, raising a `no broadcasting` error
when it is not needed.  The `'...'` elipsis is being used for two purposes,
as a place holder for existing, but unspecified dimensions, and as a
'broadcasting permission'  slip.  It's this second use that is arbitrary
and unnecessary.



