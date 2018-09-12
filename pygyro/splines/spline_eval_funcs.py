
def eval1d( self, x, kts, coeffs, deg, der=0 ):
    tck = (kts, coeffs, deg)
    return splev( x, tck, der )

@jit(nopython=True,cache=True,nogil=True)
def eval2d( x1, x2, kts1, kts2, coeffs, deg1, deg2, der1=0, der2=0 ):

    tck = (kts1, kts2, coeffs, deg1, deg2)
    return bisplev( x1, x2, tck, der1, der2 )
#--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self.ldim

        bases = []
        index = []

        for (x, space) in zip( eta, self.spaces ):

            knots  = space.knots
            degree = space.degree
            span   =  find_span( knots, degree, x )
            basis  = basis_funs( knots, degree, x, span )

            bases.append( basis )
            index.append( slice( span-degree, span+1 ) )

        # Get contiguous copy of the spline coefficients required for evaluation
        index  = tuple( index )
        coeffs = field.coeffs[index].copy()

        # Evaluation of multi-dimensional spline
        # TODO: optimize

        # Option 1: contract indices one by one and store intermediate results
        #   - Pros: small number of Python iterations = ldim
        #   - Cons: we create ldim-1 temporary objects of decreasing size
        #
        res = coeffs
        for basis in bases[::-1]:
            res = np.dot( res, basis )

#        # Option 2: cycle over each element of 'coeffs' (touched only once)
#        #   - Pros: no temporary objects are created
#        #   - Cons: large number of Python iterations = number of elements in 'coeffs'
#        #
#        res = 0.0
#        for idx,c in np.ndenumerate( coeffs ):
#            ndbasis = np.prod( [b[i] for i,b in zip( idx, bases )] )
#            res    += c * ndbasis

        return res
