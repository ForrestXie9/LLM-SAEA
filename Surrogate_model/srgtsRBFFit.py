from Surrogate_model.srgtsFitCreateState import srgtsFitCreateState
from Surrogate_model.my_refbuild import my_rbfbuild

def srgtsRBFFit(srgtOPT):
    srgtSRGT = {}
    # srgtSTT = {}

    # FIT_Fn_str = srgtOPT['FIT_Fn'].__name__


    srgtSRGT['P'] = srgtOPT['P']
    srgtSTT = srgtsFitCreateState(srgtOPT)
    srgtSRGT['RBF_Model'] = my_rbfbuild(srgtOPT['P'], srgtOPT['T'],
                                         srgtOPT['RBF_type'], srgtOPT['RBF_c'],
                                         srgtOPT['RBF_usePolyPart'], 0)


    return srgtSRGT, srgtSTT
