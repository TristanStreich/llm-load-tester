import replicate
from replicate.prediction import Prediction
from load_tester.replicate_base_target import ReplicateBaseTargetType


class DeploymentType(ReplicateBaseTargetType):
    canonical_name = "deployment"

    async def get_prediciton(self, inputs) -> Prediction:
        model_id = self.experiment.target
        deployment = await replicate.deployments.async_get(model_id)
        prediction = await deployment.predictions.async_create(
            input=inputs, stream=True
        )
        return prediction