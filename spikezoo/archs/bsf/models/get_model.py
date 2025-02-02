from .bsf.bsf import BSF

def get_model(args):
    if args.arch.upper() == 'BSF'.upper():
        model = BSF()

    return model