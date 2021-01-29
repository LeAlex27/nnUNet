from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper


class ceLossTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = RobustCrossEntropyLoss()