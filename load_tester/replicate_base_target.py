from abc import abstractmethod
import replicate.prediction
import replicate
import asyncio
from datetime import datetime, timezone
import replicate


class ReplicateBaseTargetType:
    canonical_name: str
    
    def __init__(self, experiment):
        self.experiment = experiment

    @abstractmethod
    async def get_prediciton(self, inputs) -> replicate.prediction.Prediction:
        pass

    async def make_request(self, prompt: str, max_number_tokens: int):
        inputs = {
            "prompt": prompt,
            "min_new_tokens": max_number_tokens,
            "max_new_tokens": max_number_tokens,
        }

        prediction = await self.get_prediciton(inputs)

        start_time = datetime.now(timezone.utc).replace(tzinfo=None)

        ttft_client = None
        while True:
            try:
                async for event in prediction.async_stream():
                    if ttft_client is None and str(event).strip():
                        first_token_time = datetime.now(timezone.utc).replace(
                            tzinfo=None
                        )  # Use datetime
                        ttft_client = (first_token_time - start_time).total_seconds()
                        break

                await prediction.async_wait()
            except Exception as e:
                if hasattr(prediction, "id"):
                    print(f"Failed to get prediction: {prediction.id}")
                print(e)
                print("sleeping and retrying...")
                await asyncio.sleep(1)  # Retry to connect to pred??
                continue
                # raise e
            break

        # response = await self.poll_replicate_request(client, response, headers)
        response = prediction.dict()
        if response["status"] == "failed":
            print(f"Prediction {response['id']} failed")

        completed_at = response["completed_at"]

        end_time = parse_cog_time(completed_at)

        _ = response["output"] if isinstance(response, dict) else ""
        delta = (end_time - start_time).total_seconds()

        self.experiment.append_ttft(ttft_client)
        self.experiment.append_start_time(start_time)
        self.experiment.increment_requests_made()
        self.experiment.append_returned_request(response)
        self.experiment.increment_requests_started()
        self.experiment.append_latency(delta)
        self.experiment.append_start_end_times(start_time, end_time)
        self.experiment.append_sstp(self.experiment.n_output_tokens / delta)



def parse_cog_time(x):
    # Remove trailing Z if present
    x = x.rstrip("Z")
    if "." in x:
        # Split the timestamp into date and microseconds
        date_part, microseconds_part = x.split(".")
        # Truncate or round the microseconds part to 6 digits
        microseconds_part = (microseconds_part + "0" * 6)[:6]
        # Reassemble the timestamp
        x = date_part + "." + microseconds_part
    return datetime.fromisoformat(x)