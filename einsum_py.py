import numpy as np
#import np.lib.stride_tricks.as_strided as ast
ast = np.lib.stride_tricks.as_strided

class Labels(object):
    def __init__(self, ops):
        if len(ops)>0 and hasattr(ops[0],'ndim'):
            # save data about ops, but not the ops themselves
            ndims = [op.ndim for op in ops]
            self.ndims = ndims # dimensions of the ops
            self.nop = len(ndims)
            self.shapes = [op.shape for op in ops]
            self.strides = [op.strides for op in ops]
        else:
            ndims = ops
            self.ndims = ndims # dimensions of the ops
            self.nop = len(ndims)
            self.shapes = []
            self.strides = []
        self.counts = {}
        self.num_labels = 0
        self.min_label = ord('z')
        self.max_label = ord('a')
        self.ndim_broadcast = 0

def parse_operand_subscripts(labelstr, subscripts, ndim):
    # for one operand with dimension, ndim
    idim = ndim-1
    length = len(subscripts)
    labels = [-1 for i in range(ndim)]
    left_labels, right_labels, ellipsis = False, False, False
    i = 0
    for i in range(length-1, -1, -1):
        label = subscripts[i]

        if label.isalpha():
            label = ord(label)
            if idim >= 0:
                labels[idim] = label
                idim -= 1
                if label<labelstr.min_label:
                    labelstr.min_label = label
                if label>labelstr.max_label:
                    labelstr.max_label = label
                cnt = labelstr.counts.get(label,0)
                if cnt==0:
                    labelstr.num_labels += 1
                labelstr.counts[label] = cnt + 1
                right_labels = True
            else:
                raise ValueError('too many subscripts 1')
        elif label == '.':
            if i>=2 and subscripts[i-1]=='.' and subscripts[i-2]=='.':
                ellipsis = True
                length = i-2
                break
        elif label == ' ':
            pass
        else:
            raise ValueError('invalid subscript')
    if not ellipsis and idim != -1:
        raise ValueError('ndim more than subscripts and no ellipsis')
    ndim_left = idim+1
    #print ndim_left, length, i
    idim = 0
    #print subscripts, labels, self.counts, ellipsis, chr(min_label), chr(max_label),i
    if i>0:
        for i in range(length):
            label = subscripts[i]
            if label.isalpha():
                if idim<ndim_left:
                    label = ord(label)
                    labels[idim] = label
                    idim += 1
                    if label<labelstr.min_label: labelstr.min_label = label
                    if label>labelstr.max_label: labelstr.max_label = label
                    cnt = labelstr.counts.get(label,0)
                    if cnt==0:
                        labelstr.num_labels += 1
                    labelstr.counts[label] = cnt + 1
                    left_labels = True
                else:
                    raise ValueError('too many subscripts 2')
            elif label == ' ':
                pass
            else:
                raise ValueError('not valid subscript')
    while idim < ndim_left:
        labels[idim] = 0; idim += 1
    # check for duplicates
    # so 'iii' becomes [-2,-1,105]
    for idim in range(ndim):
        label = labels[idim]
        if label>0:
            ii = idim
            while True:
                try:
                    ii = labels.index(label,ii+1)
                except ValueError:
                    break
                if ii>idim:
                    #labels[idim] = idim - ii # offset (negative)
                    labels[ii] = idim - ii

    # # apparently it does not matter which is R v L; both work
    if not ellipsis: broadcast = 'NONE'
    elif left_labels and right_labels: broadcast = 'MIDDLE'
    elif not left_labels: broadcast = 'RIGHT'  # c has !labels_left
    else: broadcast = 'LEFT'
    assert ndim==len(labels)
    return subscripts, labels, broadcast

def parse_output_subscripts(labelstr, subscripts):
    # Count the labels, making sure they're all unique and valid
    length = len(subscripts)
    nlabels = 0
    for i in range(length):
        label = subscripts[i]
        if label.isalpha():
            # check if occurs again
            if subscripts.count(label)>1:
                raise ValueError('output string contains label multiple times %s'%label)
            else:
                if labelstr.counts.get(ord(label),None):
                    nlabels += 1
                else:
                    raise ValueError('out label not in ops %s'%label)
    # number of output dimensions
    ndim = labelstr.ndim_broadcast + nlabels
    #print 'nlabels',nlabels, ndim
    labelstr = Labels([ndim]) # try new, 'dummy' structure
    # c version uses a stripped down version of operand parse to parse output
    # one that parses the same, but does not set any of the 'globals'
    # but with the above nlabels test, num labels, min, max is not affected
    # if main str is used; counts are uped though
    # error messages differ
    args = parse_operand_subscripts(labelstr, subscripts, ndim)
    #print vars(labelstr)
    return args

def count_broadcast(args):
    return max(a[1].count(0) for a in args)

def fake_outstr(labelstr):
    """
     * If there is no output signature, create one using each label
     * that appeared once, in alphabetical order
    """
    outstr = ['...']
    for label in range(labelstr.min_label, labelstr.max_label+1):
        if labelstr.counts.get(label, 0) ==1:
            outstr.append(chr(label))
    outstr = ''.join(outstr)
    return outstr

def combine_dimensions(labelstr, label_list, output_labels, ndim_output, debug=False):
    # return new shape and strides for each op
    if len(labelstr.shapes)==0: # len(ops_list)==0:
        return None
    if labelstr.nop==1:
        # if out is None
        # try remapping the axes to the output to return
        # a view instead of a copy.
        # how do I check for the out kwarg?
        ret = get_single_op_view(labelstr.ndims[0], label_list[0],
            labelstr.shapes[0], labelstr.strides[0], output_labels, ndim_output, debug=debug)
        if ret is None:
            # return ret
            pass # could not return a view
        else:
            labelstr.view = ret
            # for now return new shape and strides
            return True
    combine_list = []
    for iop in range(labelstr.nop):
        # op_shape, op_strides = ops_list[iop]
        op_shape = labelstr.shapes[iop]
        op_strides = labelstr.strides[iop]
        ndim = labelstr.ndims[iop]
        labels = label_list[iop][1]
        combine = any(l<0 for l in labels)
        if combine:
            # get_combined_dims_view(op_in[iop], iop, labels)
            if debug: print 'try to combine, iop %s'%iop
            # get_combined_dims_view()
            new_strides = [0 for l in range(ndim)]
            new_shape = [0 for l in range(ndim)]
            icombinemap = [0 for l in range(ndim)]
            icombine = 0
            # Copy the dimensions and strides, except when collapsing
            for idim in range(ndim):
                label = labels[idim]
                if label<0:
                    combineoffset = label
                    label = labels[idim+label]
                else:
                    combineoffset = 0
                    if icombine != idim:
                        labels[icombine] = labels[idim]
                    icombinemap[idim] = icombine
                # If the label is 0, it's an unlabeled broadcast dimension
                if label==0:
                    new_shape[icombine] = op_shape[idim]
                    new_strides[icombine] = op_strides[idim]
                else:
                    # Update the combined axis dimensions and strides
                    i = idim + combineoffset
                    if combineoffset<0 and new_shape[i]!=op_shape[idim]:
                        raise ValueError("dimensions in operand %d for collapsing ")
                    i = icombinemap[i]
                    new_shape[i] = op_shape[idim]
                    new_strides[i] += op_strides[idim]
                # If the label didn't say to combine axes, increment dest i
                if combineoffset==0:
                    icombine += 1
            # compressed number of dimensions
            ndim = icombine
            ret = (new_shape[:ndim], new_strides[:ndim])
            # C returns new array with these
            combine_list.append(ret)
        else:
            # no combining needed; C op[iop]=op_in[iop]
            combine_list.append(None)
    if all(x is None for x in combine_list):
        return None
    labelstr.combine_list = combine_list
    return True

def get_single_op_view(ndim, label_tpl, op_shape, op_strides, output_labels, ndim_output, debug=False):
    # labelstr.ndims[0], label_list[0], [labelstr.shapes[0], labelstr.strides[0]], output_labels, ndim_output, debug=debug
    # get a view for a single op
    # how does this differ from combine_dimensions operations for one op?

    subscripts, labels, broadcast = label_tpl
    new_shape = [0 for i in range(ndim_output)]
    new_strides = [0 for i in range(ndim_output)]
    #op_shape, op_strides = op
    ibroadcast = 0
    fail = False
    for idim in range(ndim):
        label = labels[idim]
        if debug: print 'idimlp',labels, idim, label
        if label<0:
            # parse_operand_subscripts puts neg number offset in place of repeats

            label = labels[idim+label]
            if debug: print 'new label',label
        if label==0:
            # unlabeled broadcast dimension
            # next output label thats a broadcast dim
            while ibroadcast<ndim_output:

                if output_labels[ibroadcast]==0:
                    #fail = True
                    break
                ibroadcast+=1
            if ibroadcast == ndim_output:
                raise ValueError("output had too few broadcast dimensions")

            new_shape[ibroadcast] = op_shape[idim]
            new_strides[ibroadcast] = op_strides[idim]
            if debug: print 'ibrd',ibroadcast, idim, new_shape, new_strides
            ibroadcast+=1
        else:
            # find position for the dimension in the output
            try:
                ilabel = output_labels.index(label)
            except ValueError:
                # If it's not found, reduction -> can't return a view
                fail = True
                break
            # Update the dimensions and strides of the output
            if new_shape[ilabel] != 0 and new_shape[ilabel] != op_shape[idim]:
                raise ValueError("dimensions in operand %d for collapsing "+\
                        "index '%c' don't match (%d != %d)")
            if debug: print 'ilabel',ilabel,idim,label
            new_shape[ilabel] = op_shape[idim]
            new_strides[ilabel] += op_strides[idim]
    # If we processed all the input axes, return a view
    if fail:
        if debug:
            print 'get view fail'
            print new_shape, new_strides
        return None
        # C returns 1, and ret=NULL; for errors returns 0
    else:
        # return parameters to generate new view
        if debug:
            print 'new view', op_shape, op_strides, new_shape, new_strides
        return new_shape, new_strides

def iterlabels(labelstr, output_labels, ndim_output):
    """
     * Set up the labels for the iterator (output + combined labels).
     * Can just share the output_labels memory, because iter_labels
     * is output_labels with some more labels appended.
    """
    iter_labels = output_labels[:]
    ndim_iter = ndim_output
    for label in range(labelstr.min_label, labelstr.max_label+1):
        if labelstr.counts.get(label, 0) > 0 and label not in output_labels:
            if ndim_iter > 128:
                raise ValueError('too many subscripts in einsum')
            iter_labels.append(label)
            ndim_iter += 1
    # may have 0 in iter_labels
    return iter_labels

def prepare_op_axes_original(labelstr, args, ndim, iter_labels, ndim_iter):
    subscripts, labels, broadcast = args
    axes = []
    if broadcast == 'RIGHT':
        ibroadcast = ndim-1
        #for i in range(ndim_iter-1, -1, -1):
        #    label = iter_labels[i]
        for label in reversed(iter_labels):
            if label==0:
                while ibroadcast >=0 and labels[ibroadcast] !=0:
                    ibroadcast -= 1
                if ibroadcast<0:
                    axes.insert(0,-1)
                else:
                    axes.insert(0,ibroadcast)
                    ibroadcast -= 1
            else:
                try:
                    match = labels.index(label)
                    axes.insert(0,match)
                except ValueError:
                    axes.insert(0,-1)

    elif broadcast == 'LEFT':
        ibroadcast = 0
        #for i in range(ndim_iter):
        #    label = iter_labels[i]
        for label in iter_labels:
            if label==0:
                while ibroadcast < ndim and labels[ibroadcast] !=0:
                    ibroadcast += 1
                if ibroadcast>=ndim:
                    axes.append(-1)
                else:
                    axes.append(ibroadcast)
                    ibroadcast += 1
            else:
                try:
                    match = labels.index(label)
                    axes.append(match)
                except ValueError:
                    axes.append(-1)


    elif broadcast in ['MIDDLE', 'NONE']:
        ibroadcast = 0
        #for i in range(ndim_iter):
        #    label = iter_labels[i]
        for label in iter_labels:
            if label==0:
                while ibroadcast < ndim and labels[ibroadcast] !=0:
                    ibroadcast += 1
                if ibroadcast>=ndim:
                    if False: # empty 'broadcast'
                        axes.append(-1)
                        ibroadcast += 1
                    else:
                        # default error msg
                        raise ValueError('cant middle broadcast %s %s'%(ndim, ibroadcast))
                else:
                    axes.append(ibroadcast)
                    ibroadcast += 1
            else:
                try:
                    match = labels.index(label)
                    axes.append(match)
                except ValueError:
                    axes.append(-1)

    else:
        raise ValueError( 'unknown broadcast')
    return axes

def prepare_op_axes_right(labelstr, args, ndim, iter_labels, ndim_iter):
    subscripts, labels, broadcast = args
    axes = []

    # right; adds auto broadcast on left where it belongs
    # broadcast on right has to be explicit
    ibroadcast = ndim-1
    for label in reversed(iter_labels):
        if label==0:
            while ibroadcast >=0 and labels[ibroadcast] !=0:
                ibroadcast -= 1
            if ibroadcast<0:
                axes.insert(0,-1)
            else:
                axes.insert(0,ibroadcast)
                ibroadcast -= 1
        else:
            try:
                match = labels.index(label)
                axes.insert(0,match)
            except ValueError:
                axes.insert(0,-1)
    return axes

def prepare_op_axes_left(labelstr, args, ndim, iter_labels, ndim_iter):
    subscripts, labels, broadcast = args
    axes = []

    ibroadcast = 0
    #for i in range(ndim_iter):
    #    label = iter_labels[i]
    for label in iter_labels:
        if label==0:
            while ibroadcast < ndim and labels[ibroadcast] !=0:
                ibroadcast += 1
            if ibroadcast>=ndim:
                axes.append(-1)
            else:
                axes.append(ibroadcast)
                ibroadcast += 1
        else:
            try:
                match = labels.index(label)
                axes.append(match)
            except ValueError:
                axes.append(-1)
    return axes

prepare_op_axes = prepare_op_axes_right


def prepare_out_axes(labelstr, ndim_output, ndim_iter):
    axes = range(ndim_output) + [-1]*(ndim_iter-ndim_output)
    return axes

def parse_subscripts(subscripts, labelstr, debug=True, **kwargs):
    #
    ndims = labelstr.ndims
    opstr = subscripts.split('->')
    if len(opstr)>1:
        opstr, outstr = opstr
    else:
        opstr = opstr[0]
        outstr = None
    opstr = opstr.split(',')
    label_list = []
    for subscripts, ndim in zip(opstr, ndims):
        args = parse_operand_subscripts(labelstr, subscripts, ndim)
        label_list.append(args)
    labelstr.ndim_broadcast = count_broadcast(label_list)
    if outstr is None:
        outstr = fake_outstr(labelstr)
    #print fake_outstr(labelstr), outstr
    argout = parse_output_subscripts(labelstr, outstr)
    output_labels = argout[1]
    ndim_output = len(output_labels)
    # if out is not None, cf its dim with ndim_output
    if debug:
        print subscripts
        print vars(labelstr)
        print label_list
        print argout
    # print parse_output_subscripts(labelstr, fake_outstr(labelstr))
    ret = combine_dimensions(labelstr, label_list, output_labels, ndim_output, debug=debug)
    if ret is not None:
        if hasattr(labelstr,'view'):
            if debug: print 'view', labelstr.view
            #print vars(labelstr)
            return labelstr, []
        if hasattr(labelstr,'combine_list'):
            if debug:
                print 'combined dims', labelstr.combine_list
                #print vars(labelstr)
                print label_list

    iter_labels = iterlabels(labelstr, output_labels, ndim_output)
    ndim_iter = len(iter_labels)
    op_axes = []
    for args, ndim in zip(label_list, ndims):
        args = prepare_op_axes(labelstr, args, ndim, iter_labels, ndim_iter)
        op_axes.append(args)
    args = prepare_out_axes(labelstr, ndim_output, ndim_iter)
    op_axes.append(args)
    if debug:
        def foo(a):
            if ord('a')<=a<=ord('z'):
                a = chr(a)
            return '%s'%a
        print 'iter labels: %s,%r'%(iter_labels, ''.join(foo(a) for a in iter_labels),)
        print 'op_axes', op_axes
    #op_a
    xes = [a[1:] for a in op_axes]
    return labelstr, op_axes

def sum_of_prod(ops, op_axes, order='K', itdump=False, funs=[], **kwargs):
    nop = len(ops)
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
        if funs:
            for (x,w) in it:
                funs[0](x, w)
                cnt += 1
        else:
            # a sum without multiply
            for (x,w) in it:
                w[...] += x
                cnt += 1
    elif nop==2:
        if funs:
            for (x,y,w) in it:
                #np.add(np.multiply(x, y), w, out=w)
                funs[0](funs[1](x,y), w, w)
                cnt += 1
        else:
            for (x,y,w) in it:
                w[...] += x*y
                cnt += 1
    elif nop==3:
        if funs:
            raise ValueError('generalized funs not implemented for nop 3')
        for (x,y,z,w) in it:
            w[...] += x*y*z
            cnt += 1
    else:
        raise ValueError('calc for more than 3 nop not implemented')

    if itdump:
        print 'cnt',cnt, x.shape
    return it.operands[nop]


def myeinsum(subscripts, *ops, **kwargs):
    # dropin preplacement for np.einsum (more or less)
    kwargs.setdefault('debug', False)
    ops = [np.array(x) for x in ops]
    # ndims = [x.ndim for x in ops]
    # kwargs.setdefault('ops_list', [(op.shape, op.strides) for op in ops])
    label_str = Labels(ops)
    #print vars(label_str)
    label_str, op_axes = parse_subscripts(subscripts, label_str, **kwargs)
    if hasattr(label_str, 'view'):
        view = label_str.view
        x = ast(ops[0], shape=view[0], strides=view[1])
        print 'view', ops[0].shape, '=>',x.shape
        # einsum preserves the base
    elif hasattr(label_str, 'combine_list'):
        clist = label_str.combine_list
        ops1 = [op for op in ops]
        for i in range(len(ops)):
            if clist[i] is not None:
                new_shape, new_strides = clist[i]
                ops1[i] = ast(ops[i], shape=new_shape, strides=new_strides)
                print 'combined',ops[i].shape,'=>',ops1[i].shape
        x = sum_of_prod(ops1, op_axes, **kwargs)
    else:
        x = sum_of_prod(ops, op_axes, **kwargs)
    if kwargs.has_key('out'):
        kwargs['out'][...] = x
    return x

if __name__ == '__main__':

    if False:
        trials = [
            ('ij...,...kj->i...k', [2,2]),
            ('ij...,...kj->...ik', [2,2]),
            ('ij...,...kj, kk,...i->i...j',[2,2,2,1]),
            ('ij...,...jk->i...k',[2,3]),
            ('ij,jk->ik',[2,2]),
            ('ik,kj->ij',[2,2]),
            ('...k,kj->...j',[2,2]),
            ('...j,j...->...',[2,2]),
        ]
        trials1 = [
            ('ik,kj->ij', [2,2]),
            ('ik...,k...->i...', [2,2]),
            ('ik...,...kj->i...j',[2,2]),
            ('ik,k...->i...',[2,2]),
        ]
        for s,n in trials:
            print s
            parse_subscripts(s, Labels(n))
            print ''

        print '\n-----------------'
        A = np.arange(12).reshape((4,3))
        B = np.arange(6).reshape((3,2))
        # np.einsum('ik,k...->i...', A, B)
        # this error
        print np.einsum('ik,kj->ij', A, B)  #ok
        np.einsum('ik...,k...->i...', A, B) #ok
        np.einsum('ik...,...kj->i...j', A, B)
        print 'add(multiply())'
        funs = [np.add, np.multiply]
        print sum_of_prod([A,B],[[0,1,-1],[-1,0,1], [0,-1,1]], funs=funs)

        label_str, op_axes = parse_subscripts('ik,kj->ij', Labels([A.ndim,B.ndim]))
        print op_axes
        # [[0, -1, 1], [-1, 1, 0], [0, 1, -1]]  fine
        # map (4,newaxis,3)(newaxis,3,2)->(4,2,newaxis)
        print sum_of_prod([A,B],op_axes)

        label_str, op_axes = parse_subscripts('ik...,k...->i...', Labels([A.ndim,B.ndim]))
        print op_axes
        # correct with left/right switch
        print sum_of_prod([A,B],op_axes)

        label_str, op_axes = parse_subscripts('ik...,...kj->i...j', Labels([A.ndim,B.ndim]))
        print op_axes
        # [[0, -1, 1], [0, 1, -1], [0, 1, -1]]
        print sum_of_prod([A,B],op_axes)
        # correct with l/r switch

        label_str, op_axes = parse_subscripts('ik,k...->i...', Labels([A.ndim,B.ndim]))
        print op_axes
        print sum_of_prod([A,B],op_axes)

        # 'ik,k...->i...' np.einsum error
        # 'ik,kj->ij'  #ok
        # 'ik...,k...->i...' #ok
        # 'ik...,...kj->i...j'
        # print myeinsum('ik,kj->ij', A, B)
        print 'add(multiply())'
        funs = [np.add, np.multiply]
        print myeinsum('ik,kj->ij', A, B, funs=funs)
        print 'maximum(add())'
        funs = [np.maximum, np.add]
        print myeinsum('ik,kj->ij', A, B, funs=funs)


        # - how to account for l/r switch
        # - how get last case work

    if True:

        print '\nprefactor, dipoles'
        N, M, O = 160, 160, 128
        N,M,O = 16,16,12
        prefactor = np.random.random((1, 1, 1, M, N, O))
        dipoles = np.random.random((M, N, O, 3))
        subscripts = '...lmn,...lmno->...o'
        subscripts = 'abclmn,lmno->abco'
        print subscripts
        x = np.einsum(subscripts, prefactor, dipoles)
        print x.shape
        label_str, op_axes = parse_subscripts(subscripts, Labels([prefactor.ndim,dipoles.ndim]))
        #print op_axes
        x = sum_of_prod([prefactor,dipoles],op_axes)
        print x.shape

        print ''
        subscripts = '...bclmn,...lmno->...bco'
        subscripts = '...lmn,...lmno->...o'
        subscripts = '...lmn,lmno->...o'
        #subscripts = 'lmn,lmno->...o'
        print subscripts
        try:
            x = np.einsum(subscripts, prefactor, dipoles)
            print x.shape
            print 'einsum correct with %r'%subscripts
        except ValueError:
            print 'einsum error with %r'%subscripts
            pass
        """
        label_str, op_axes = parse_subscripts(subscripts, [prefactor.ndim,dipoles.ndim])
        #print op_axes
        x = sum_of_prod([prefactor,dipoles],op_axes)
        """
        x = myeinsum(subscripts, prefactor, dipoles)
        print x.shape

    if True:

        print
        dtype='int32'; n =4
        a = np.arange(3*n, dtype=dtype).reshape(3,n)
        b = np.arange(2*3*n, dtype=dtype).reshape(2,3,n)
        c = myeinsum("..., ...", a, b,debug=True)
        print c.shape
        # shorter prepare_op_axes does not work with this case
        # 'right', iterate on reversed labels does fine
        # axes: [[-1, 0, 1], [0, 1, 2], [0, 1, 2]]
        print
        a = np.arange(2*3).reshape(2,3)
        b = np.arange(2*3*4).reshape(2,3,4)
        try:
            c = np.einsum("ij, ij...->ij...", a, b) # cnnot extend
            print c.shape, 'with %r'%"ij, ij...->ij..."
        except ValueError:
            print 'einsum error with %r'%"ij, ij...->ij..."
        #c = myeinsum("..., ...", a, b,debug=True) # tries to add dim at start
        c = myeinsum("ij, ij...->ij...", a, b,debug=True)
        print c.shape

        print '\n19388152'
        dims=[2,3,4,5];
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        print myeinsum('ijkl,k->ijl',a,v).shape
        print myeinsum('ijkl,k',a,v).shape
        print myeinsum('...kl,k',a,v).shape  # np.einsum xxx
        print myeinsum('...kl,k...',a,v).shape
        print np.einsum('ijkl,k->ijl',a,v).shape
        print np.einsum('...kl,...k',a,v).shape
        print np.einsum('...kl,k...',a,v).shape
        try:
            print np.einsum('...kl,k',a,v).shape
        except ValueError:
            print 'np.einsum ValueError with %r'%'...kl,k'
        # does not matter which side the ... is on
        # myeinsum does not have the broadcast error objection

    if True:

        print '\nviews'
        a = np.arange(9).reshape(3,3)
        b = myeinsum("ii->i", a,debug=True)
        print a
        print 'got', b
        print 'expect diagonal',[a[i,i] for i in range(3)]
        ##assert_(b.base is a)
        ##assert_equal(b, [a[i,i] for i in range(3)])

        a = np.arange(27).reshape(3,3,3)
        print '\niii->i',np.einsum('iii->i',a) # [0 13 26]
        print myeinsum('iii->i', a,debug=True) # error []

        print '\n..ii->...i',np.einsum('...ii->...i',a)
        print myeinsum('...ii->...i', a,debug=True) # err
        # not returning a view

        print '\nii...->...i',np.einsum('ii...->...i',a)
        print myeinsum('ii...->...i', a,debug=True) # err

        print '\njii->ij',np.einsum('jii->ij',a)
        print myeinsum('jii->ij', a,debug=True)

        a = np.arange(24).reshape(2,3,4)
        print '\nijk->jik',np.einsum('ijk->jik',a)
        print myeinsum('ijk->jik', a,debug=False) # ok

        #parse_subscripts('iii', [3], debug=True)
        #parse_subscripts('iki',[3], debug=True)

    if True:
        print '\n combine dims'
        print "'ii,i', expect 3.0"
        print np.einsum('ii,i',np.eye(3),np.ones((3)))
        assert 3.0==myeinsum('ii,i',np.eye(3),np.ones((3)),debug=True)

        print "\n'i->' expect 3"
        print np.einsum('i->...',np.arange(3))
        assert 3== myeinsum('i->',np.arange(3),debug=True)

        print "\n'ii', expect 3"
        print np.einsum('ii', np.eye(3))
        assert 3==myeinsum('ii',np.eye(3))


        # http://scipy-lectures.github.io/advanced/advanced_numpy/
        print "\n'ijij', expect 7800"
        x = np.arange(5*5*5*5).reshape(5,5,5,5)
        s = 0
        for i in xrange(5):
            for j in xrange(5):
                s += x[j,i,j,i]
        print s # = 7800
        print np.einsum('ijij',x)
        #by striding, and using sum() on the result.
        x.strides # (500, 100, 20, 4)

        y=ast(x,shape=(5,5),strides=(520,104))
        s2 = y.sum()
        assert s == s2
        print myeinsum('ijij',x,debug=True)

    if True:
        prepare_op_axes = prepare_op_axes_right
        # this case gives error if BROADCAST_LEFT is used instead of RIGHT
        # (3,1) (2,3,1)
        # target:
        # op_axes [[-1, 0, 1], [0, 1, 2], [0, 1, 2]]
        # makes (newaxis,3,1)
        # wrong with 'right'
        # op_axes [[0, 1, -1], [0, 1, 2], [0, 1, 2]]
        # makes (3,1,newaxis)
        n = 1; dtype = np.int32
        a = np.arange(3*n, dtype=dtype).reshape(3,n)
        b = np.arange(2*3*n, dtype=dtype).reshape(2,3,n)
        astr = '..., ...'
        print myeinsum(astr, a, b,debug=True)
        print np.multiply(a, b)
        print np.einsum(astr, a, b)
        print np.einsum(astr, a, b).shape

        print '\n'
        prepare_op_axes = prepare_op_axes_right
        a = np.arange(3*2).reshape(2,3)
        astr = '..., ...'   # both 'right' broadcast type; fails both with original and right
        #astr = 'ij, ijk ->ijk'
        #astr = 'ij, ij...->ij...' # fails 'original'
        #astr = 'ij...,ijk->ijk'  # explicit broadcast
        #astr = '...ij,ijk->ijk' # also ok
        # op_axes [[-1, 0, 1], [0, 1, 2], [0, 1, 2]] right
        # sumofprod error, fail broadcast
        # op_axes [[0, 1, -1], [0, 1, 2], [0, 1, 2]] left, correct
        print myeinsum(astr, a, b,debug=True) # cannot broadcast with 'right'; ok with left
        print np.multiply(a[...,None], b)
        print np.einsum(astr, a, b)