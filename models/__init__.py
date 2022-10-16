from .training_modules import NLSTTrainingModule

training_modules = {
    'NLST': NLSTTrainingModule,
}

__all__ = ['training_modules']
