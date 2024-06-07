import replicate
from replicate.prediction import Prediction
from load_tester.replicate_base_target import ReplicateBaseTargetType



class ModelType(ReplicateBaseTargetType):
    canonical_name = "model"

    async def get_prediciton(self, inputs) -> Prediction:
        model_id = self.experiment.target
        if ":" not in model_id:
            latest_version = await replicate.models.async_get(model_id)
            version = latest_version.latest_version.id
            model_id = f"{model_id}:{version}"
        version = model_id.split(":")[-1]
        prediction = await replicate.predictions.async_create(version=version, input=inputs, stream=True)
        return prediction