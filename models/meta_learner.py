import learn2learn as l2l

class MetaLearner:
    """
    MAML wrapper for first-shot adaptation
    """
    def __init__(self, model, lr_inner=1e-2, ways=1, shots=5):
        self.maml = l2l.algorithms.MAML(
            model, lr=lr_inner, first_order=True
        )
    def adapt(self, losses, inner_steps=1):
        """
        Perform inner-loop adaptation on a small support set 
        """
        adapted_models = []
        for loss in losses:
            learner = self.maml.clone()
            for _ in range(inner_steps):
                learner.adapt(loss)
            adapted_models.append(learner)
        return adapted_models
