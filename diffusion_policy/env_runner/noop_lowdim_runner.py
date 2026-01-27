from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class NoopLowdimRunner(BaseLowdimRunner):
    def __init__(self, **kwargs):
        pass

    def run(self, policy):
        # Provide a default metric so TopKCheckpointManager can run.
        return {"test_mean_score": 0.0}
