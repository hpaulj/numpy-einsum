import numpy as np
#import np.lib.stride_tricks.as_strided as ast
ast = np.lib.stride_tricks.as_strided

class Labels(object):
    def __init__(self, ndims):
        self.ndims = ndims
        self.nop = len(ndims)
        self.counts = {}
        self.num_labels = 0
        self.min_label = ord('z')
        self.max_label = ord('a')
        self.ndim_broadcast = 0

def parse_operand_subscripts(labelstr, astr, ndim):
    # for one operand with dimension, ndim
    idim = ndim-1
    length = len(astr)
    labels = [-1 for i in range(ndim)]
    left_labels, right_labels, ellipsis = False, False, False
    i = 0
    for i in range(length-1, -1, -1):
        label = astr[i]

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
            if i>=2 and astr[i-1]=='.' and astr[i-2]=='.':
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
    #print astr, labels, self.counts, ellipsis, chr(min_label), chr(max_label),i
    if i>0:
        for i in range(length):
            label = astr[i]
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
    return astr, labels, broadcast

def parse_output_subscripts(labelstr, astr):
    # Count the labels, making sure they're all unique and valid
    length = len(astr)
    nlabels = 0
    for i in range(length):
        label = astr[i]
        if label.isalpha():
            # check if occurs again
            if astr.count(label)>1:
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
    args = parse_operand_subscripts(labelstr, astr, ndim)
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

def combine_dimensions(labelstr, label_list, ops_list, output_labels, ndim_output, debug=False):
    # this creates new shape and strides for each op
    # does this act on label_list labels, or ops (new view) or both?
    # get_combined_dims_view
    # op[iop] = op_in[iop]
    # op[nop] = out
    if len(ops_list)==0:
        return None
    if labelstr.nop==1:
        # if out is None
        # try remapping the axes to the output to return
        # a view instead of a copy.
        # how do I check for the out kwarg?
        ret = get_single_op_view(labelstr.ndims[0], label_list[0], ops_list[0], output_labels, ndim_output, debug=debug)
        if ret is None:
            # return ret
            pass # could not return a view
        else:
            labelstr.view = ret
            # for now return new shape and strides
            return True
    combine_list = []
    for iop in range(labelstr.nop):
        op_shape, op_strides = ops_list[iop]
        ndim = labelstr.ndims[iop]
        labels = label_list[iop][1]
        combine = any(l<0 for l in labels)
        if combine:
            # get_combined_dims_view(op_in[iop], iop, labels)
            if debug: print 'try to combine, iop %s'%iop
            # get_combined_dims_view()
            new_strides = [0 for l in range(ndim)]
            new_dims = [0 for l in range(ndim)]
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
                    new_dims[icombine] = op_shape[idim]
                    new_strides[icombine] = op_strides[idim]
                else:
                    # Update the combined axis dimensions and strides
                    i = idim + combineoffset
                    if combineoffset<0 and new_dims[i]!=op_shape[idim]:
                        raise ValueError("dimensions in operand %d for collapsing ")
                    i = icombinemap[i]
                    new_dims[i] = op_shape[idim]
                    new_strides[i] += op_strides[idim]
                # If the label didn't say to combine axes, increment dest i
                if combineoffset==0:
                    icombine += 1
            # compressed number of dimensions
            ndim = icombine
            ret = (new_dims[:ndim], new_strides[:ndim])
            # C returns new array with these
            combine_list.append(ret)
        else:
            # no combining needed; C op[iop]=op_in[iop]
            combine_list.append(None)
    if all(x is None for x in combine_list):
        return None
    labelstr.combine_list = combine_list
    return True

def get_single_op_view(ndim, label_tpl, op, output_labels, ndim_output, debug=False):
    # get a view for a single op

    astr, labels, broadcast = label_tpl
    new_dims = [0 for i in range(ndim_output)]
    new_strides = [0 for i in range(ndim_output)]
    op_shape, op_strides = op
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

            new_dims[ibroadcast] = op_shape[idim]
            new_strides[ibroadcast] = op_strides[idim]
            if debug: print 'ibrd',ibroadcast, idim, new_dims, new_strides
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
            if new_dims[ilabel] != 0 and new_dims[ilabel] != op_shape[idim]:
                raise ValueError("dimensions in operand %d for collapsing "+\
                        "index '%c' don't match (%d != %d)")
            if debug: print 'ilabel',ilabel,idim,label
            new_dims[ilabel] = op_shape[idim]
            new_strides[ilabel] += op_strides[idim]
    # If we processed all the input axes, return a view
    if fail:
        if debug:
            print 'get view fail'
            print new_dims, new_strides
        return None
        # C returns 1, and ret=NULL; for errors returns 0
    else:
        # return parameters to generate new view
        if debug:
            print 'new view', op_shape, op_strides, new_dims, new_strides
        return new_dims, new_strides

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

def prepare_op_axes(labelstr, args, ndim, iter_labels, ndim_iter):
    astr, labels, broadcast = args
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
                    if True: # empty 'broadcast'
                        axes.append(-1)
                        ibroadcast += 1
                    else:
                        # when shouldn't we do broadcast?
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

def prepare_op_axes(labelstr, args, ndim, iter_labels, ndim_iter):
    astr, labels, broadcast = args
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

def prepare_out_axes(labelstr, ndim_output, ndim_iter):
    axes = range(ndim_output) + [-1]*(ndim_iter-ndim_output)
    return axes

def parse_subscripts(subscripts, ndims, debug=True, ops_list=[], **kwargs):
    #
    labelstr = Labels(ndims)
    opstr = subscripts.split('->')
    if len(opstr)>1:
        opstr, outstr = opstr
    else:
        opstr = opstr[0]
        outstr = None
    opstr = opstr.split(',')
    label_list = []
    for astr, ndim in zip(opstr,ndims):
        args = parse_operand_subscripts(labelstr, astr, ndim)
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
    ret = combine_dimensions(labelstr, label_list,ops_list,output_labels, ndim_output, debug=debug)
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

def mysum(ops, op_axes, order='K', debug=False, funs=[]):
      nops = len(ops)
      ops.append(None)
      flags = ['reduce_ok','buffered', 'external_loop',
             'delay_bufalloc', 'grow_inner',
             'zerosize_ok', 'refs_ok']
      op_flags = [['readonly']]*nops + [['allocate','readwrite']]

      it = np.nditer(ops, flags, op_flags, op_axes=op_axes,
            order=order)
      it.operands[nops][...] = 0
      it.reset()
      cnt = 0
      if debug:
            it.debug_print()
      if nops==1:
            if funs:
                # sum of squares or the like
                for (x,w) in it:
                    funs[0](funs[1](x,x), w, w)
                    # w[...] += x*x
                    cnt += 1
            else:
                # a sum without multiply?
                for (x,w) in it:
                    w[...] += x
                    cnt += 1
      elif nops==2:
            if funs:
                  assert nops==2
                  for (x,y,w) in it:
                        #np.add(np.multiply(x, y), w, out=w)
                        funs[0](funs[1](x,y), w, w)
                        cnt += 1
            else:
                  assert nops==2
                  for (x,y,w) in it:
                        w[...] += x*y
                        cnt += 1
      elif nops==3:
            for (x,y,z,w) in it:
                  w[...] += x*y*z
                  cnt += 1
      else:
            raise Error

      if debug:
            print 'cnt',cnt, x.shape
      return it.operands[nops]


def myeinsum(astr, *ops, **kwargs):
    # debug = kwargs.get('debug', False)
    kwargs.setdefault('debug',False)
    ops = [np.array(x) for x in ops]
    dims = [x.ndim for x in ops]
    kwargs.setdefault('ops_list',[(op.shape, op.strides) for op in ops])
    label_str, op_axes = parse_subscripts(astr, dims, **kwargs)
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
        # op_axes need to be adjusted to reflect new shape
        # mysum for nops=1 needs to do sum of right axis
        x = mysum(ops1, op_axes)
    else:
        x = mysum(ops, op_axes)
    if kwargs.has_key('out'):
        kwargs['out'][...] = x
    return x

if __name__ == '__main__':

    if True:
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
            parse_subscripts(s,n)
            print ''

        #from numpy_hub2455 import mysum

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
        print mysum([A,B],[[0,1,-1],[-1,0,1], [0,-1,1]], funs=funs)

        label_str, op_axes = parse_subscripts('ik,kj->ij', [A.ndim,B.ndim])
        print op_axes
        # [[0, -1, 1], [-1, 1, 0], [0, 1, -1]]  fine
        # map (4,newaxis,3)(newaxis,3,2)->(4,2,newaxis)
        print mysum([A,B],op_axes)

        label_str, op_axes = parse_subscripts('ik...,k...->i...', [A.ndim,B.ndim])
        print op_axes
        # correct with left/right switch
        print mysum([A,B],op_axes)

        label_str, op_axes = parse_subscripts('ik...,...kj->i...j', [A.ndim,B.ndim])
        print op_axes
        # [[0, -1, 1], [0, 1, -1], [0, 1, -1]]
        print mysum([A,B],op_axes)
        # correct with l/r switch

        label_str, op_axes = parse_subscripts('ik,k...->i...', [A.ndim,B.ndim])
        print op_axes
        print mysum([A,B],op_axes)

        # - how to account for l/r switch
        # - how get last case work

    if True:
        N, M, O = 160, 160, 128
        N,M,O = 16,16,12
        prefactor = np.random.random((1, 1, 1, M, N, O))
        dipoles = np.random.random((M, N, O, 3))
        astr = '...lmn,...lmno->...o'
        astr = 'abclmn,lmno->abco'
        print astr
        x = np.einsum(astr, prefactor, dipoles)
        print x.shape
        label_str, op_axes = parse_subscripts(astr, [prefactor.ndim,dipoles.ndim])
        #print op_axes
        x = mysum([prefactor,dipoles],op_axes)
        print x.shape

        print ''
        astr = '...bclmn,...lmno->...bco'
        astr = '...lmn,...lmno->...o'
        astr = '...lmn,lmno->...o'
        #astr = 'lmn,lmno->...o'
        print astr
        try:
            x = np.einsum(astr, prefactor, dipoles)
            print x.shape
        except ValueError:
            print 'einsum error'
            pass
        """
        label_str, op_axes = parse_subscripts(astr, [prefactor.ndim,dipoles.ndim])
        #print op_axes
        x = mysum([prefactor,dipoles],op_axes)
        """
        x = myeinsum(astr, prefactor, dipoles)
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
        #c = np.einsum("ij, ij...->ij...", a, b) # cnnot extend
        #print c.shape
        #c = myeinsum("..., ...", a, b,debug=True) # tries to add dim at start
        c = myeinsum("ij, ij...->ij...", a, b,debug=True)
        print c.shape

        print '\n19388152'
        dims=[2,3,4,5];
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        print myeinsum('ijkl,k->ijl',a,v).shape
        print myeinsum('...kl,k',a,v).shape  # np.einsum xxx
        print myeinsum('...kl,k...',a,v).shape
        print np.einsum('ijkl,k->ijl',a,v).shape
        print np.einsum('...kl,...k',a,v).shape
        print np.einsum('...kl,k...',a,v).shape
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

