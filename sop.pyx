import numpy as np
cimport numpy as np
cimport cython

def sum_of_prod(ops, op_axes, order='K', itdump=False, **kwargs):
    nop = len(ops)
    if nop==2:
        return sum_product_cy(ops, op_axes, order=order)
    ops.append(None)
    flags = ['reduce_ok','buffered', 'external_loop',
             'delay_bufalloc', 'grow_inner',
             'zerosize_ok', 'refs_ok']
    op_flags = [['readonly']]*nop + [['allocate','readwrite']]

    it = np.nditer(ops, flags, op_flags, op_axes=op_axes,
        order=order)
    it.operands[nop][...] = 0
    it.reset()
    cnt = 0
    if itdump:
        it.debug_print()
    if nop==1:
        # a sum without multiply
        for (x,w) in it:
            w[...] += x
            cnt += 1
    elif nop==2:
        for (x,y,w) in it:
            w[...] += x*y
            cnt += 1
    elif nop==3:
        for (x,y,z,w) in it:
            w[...] += x*y*z
            cnt += 1
    else:
        raise ValueError('calc for more than 3 nop not implemented')

    return it.operands[nop]

@cython.boundscheck(False)
def sum_product_cy(ops, op_axes, order='K'):
    #(arr, axis=None, out=None):
    cdef np.ndarray[double] x
    cdef np.ndarray[double] y
    cdef np.ndarray[double] w
    cdef int size
    cdef int nop

    nop = len(ops)
    ops.append(None)

    """
    axeslist = axis_to_axeslist(axis, arr.ndim)
    it = np.nditer([arr, out], flags=['reduce_ok', 'external_loop',
                                      'buffered', 'delay_bufalloc'],
                op_flags=[['readonly'], ['readwrite', 'allocate']],
                op_axes=[None, axeslist],
                op_dtypes=['float64', 'float64'])
    """

    flags = ['reduce_ok','buffered', 'external_loop',
             'delay_bufalloc', 'grow_inner',
             'zerosize_ok', 'refs_ok']
    op_flags = [['readonly']]*nop + [['allocate','readwrite']]

    it = np.nditer(ops, flags, op_flags, op_axes=op_axes, order=order)
    it.operands[nop][...] = 0
    it.reset()
    for xarr, yarr, warr in it:
        x = xarr
        y = yarr
        w = warr
        size = x.shape[0]
        for i in range(size):
           value = x[i]
           w[i] = w[i] + x[i] * y[i]
    return it.operands[nop]

"""
# ops need to be double to match (for now) dtype in code
x=np.arange(1000,dtype=np.double).reshape(10,10,10)
y=np.arange(1000,dtype=np.double).reshape(10,10,10)
einsum_py.myeinsum('ijk,ijk->ijk',x,y,debug=True).shape
# use op_axes from this
op_axes=[[1, 2, 0], [1, 2, 0], [1, 2, 0]]
ops=[x,y];w=sop.sum_product_cy(ops,op_axes)

# cython sop.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/local/include/python2.7 -o sop.so sop.c

"""