# LLM Perf Tester



## Installation
To install simply run
```shell
pipx install https://github.com/TristanStreich/llm-load-tester/archive/main.zip
```
To verify that the installation worked run
```shell
perf_test --help
```

## Usage

* `--target`: should be the name of a replicate model or deployment. I.E: `meta/meta-llama-3-8b-instruct` or `replicate-internal/llama-2-13b-chat-int8-1xa100-80gb-triton:b127007ce910cd8e291f3a7da2405a7cc2562fa63ac2e8f21905b2e0b5edb10d`
* `--target_type`: set to the type of model. Currently only `model`, `official-model`, or `deployemnt` but see [Adding New Model Types](#adding-new-model-types) for how to add more.
* `--unit`: choose between rps or batch size mode
* `--rate`: sets either batch size ore rps depending on the `--unit` set
* `--n_io_tokens`: input_tokens:output_tokens currently input tokens are random ascii characters
* `--duration`: duration of each test in seconds

For each combination of `rate` and `n_io_tokens` a different test is run.


### Example

The following runs a test against the offical model `meta/meta-llama-3-8b-instruct`. This runs with a batch size of 4 and 128 random input tokens and 128 output tokens for 1 second.

```shell
perf_test \
    --target meta/meta-llama-3-8b-instruct \
    --target_type official-model \
    --unit batch \
    --rate 4 \
    --n_io_tokens 128:128 \
    --duration 1
```

## Outputs

Tests outputs are written into a dir `perf-results`. Each invocation of the cli will create a dir called `{timestamp}-{target}-{unit}-{rates_str}-{n_io_tokens}-{duration}`. Then each combination of duration io_tokens and rate will create subdir with a series of graphs for the test.

## Development

This is set up with [poetry](https://python-poetry.org/). Once the repo is cloned run
```shell
poetry install
poetry shell
```
which enters a interactive terminal in a virtual env with all the necessary dependencies installed and the cli in the path.

To configure vscode to use the venv, you can use `Python: Set Interpreter` command and set it to the output of `poetry env info --path`


## Adding New Model Types


Currently this test supports replicate official models, models, and deployments. All of them are implemented as base classes of `ReplicateBaseTargetType` in the `target_types` dir. To add a new implemention of the target type with a different service discovery mechanism, just add a new file in `target_types` with a class that implements `ReplicateBaseTargetType`. This must have a field `canonical_name` which is what will be passed into the cli `--target_type` to invoke it.


Outputs will be attached to the passed in experiment object like so:


```python
class MyNewType(ReplicateBaseTargetType):
    canonical_name: "my_new_type"
    
    def __init__(self, experiment):
        self.experiment = experiment

    
    async def make_request(self, prompt: str, max_number_tokens: int):
        self.experiment.append_ttft(ttft_client)
        self.experiment.append_start_time(start_time)
        self.experiment.increment_requests_made()
        self.experiment.append_returned_request(response)
        self.experiment.increment_requests_started()
        self.experiment.append_latency(delta)
        self.experiment.append_start_end_times(start_time, end_time)
        self.experiment.append_sstp(self.experiment.n_output_tokens / delta)
```


Which you could then invoke with `perf_test --target_type my_new_type`