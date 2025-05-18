import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, ae: nn.Module, flow: nn.Module, student: nn.Module=None,
                 w_ae=0.5, w_flow=0.5):
        super().__init__()
        self.ae = ae
        self.flow = flow
        self.student = student
        self.w_ae = w_ae
        self.w_flow = w_flow

    def forward(self, x):
        # x: [B,1,n_mels,T]
        # AE
        x_rec, z = self.ae(x)
        ae_score = torch.mean((x - x_rec)**2, dim=[1,2,3])
        # flow on latent
        flow_logprob = self.flow(z)
        flow_score = -flow_logprob  # higher means more anomalous
        # normalized fusion (assumes pre-normalized by user)
        ens_score = self.w_ae * ae_score + self.w_flow * flow_score
        # student (if available)
        student_score = None
        if self.student:
            student_score, _ = self.student(x)
        return {
            'ae': ae_score,
            'flow': flow_score,
            'ensemble': ens_score,
            'student': student_score
        }