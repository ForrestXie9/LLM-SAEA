def srgtsRBFSetOptions(P=None, T=None, FIT_Fn=None, FIT_LossFn=None, RBF_type=None, RBF_c=None, RBF_usePolyPart=None):
    srgtOPT = {}
    srgtOPT['SRGT'] = 'RBF'

    if P is None:
        srgtOPT['P'] = []
        srgtOPT['T'] = []
        srgtOPT['FIT_Fn'] = None
        srgtOPT['RBF_type'] = None
        srgtOPT['RBF_c'] = None
        srgtOPT['RBF_usePolyPart'] = None
    elif T is None:
        raise ValueError("Both P and T must be provided.")
    elif FIT_Fn is None:
        srgtOPT['P'] = P
        srgtOPT['T'] = T
        srgtOPT['FIT_Fn'] = 'my_rbfbuild'

        srgtOPT['RBF_type'] = 'MQ'
        srgtOPT['RBF_c'] = 2
        srgtOPT['RBF_usePolyPart'] = 0
    else:
        srgtOPT['P'] = P
        srgtOPT['T'] = T
        srgtOPT['FIT_Fn'] = FIT_Fn

        srgtOPT['RBF_type'] = RBF_type
        srgtOPT['RBF_c'] = RBF_c
        srgtOPT['RBF_usePolyPart'] = RBF_usePolyPart

    return srgtOPT
