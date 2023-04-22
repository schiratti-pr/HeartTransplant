from .training_modules import NLSTTrainingModule
from .reconstruction import NLSTReconstructModule

training_modules = {
    'NLST': NLSTTrainingModule,
    'NLST_reconstruct': NLSTReconstructModule
}

__all__ = ['training_modules', 'reconstruction']
