import replicate
from replicate.prediction import Prediction
from load_tester.replicate_base_target import ReplicateBaseTargetType

class ModelType(ReplicateBaseTargetType):
    canonical_name = "official-model"

    async def get_prediciton(self, inputs) -> Prediction:
        model_id = self.experiment.target
        prediction = await replicate.models.predictions.async_create(model_id, input=inputs, stream=True)
        return prediction