from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.dcase2025_multi_branch.dcase2025_multi_branch import DCASE2025MultiBranch

class Models:
    ModelsDic = {
        "DCASE2023T2-AE":DCASE2023T2AE,
        "DCASE2025MultiBranch":DCASE2025MultiBranch
    }

    def __init__(self,models_str):
        self.net = Models.ModelsDic[models_str]

    def show_list(self):
        return Models.ModelsDic.keys()
