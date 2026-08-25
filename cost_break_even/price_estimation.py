import math
from dataclasses import replace
from demo.demo_config import DemoScenario, Model
from demo.token_count_estimation import num_tokens
from demo.x_values import get_plot_x_ticks

OUTPUT = "output"
INPUT = "input"
LLAMA_8B = "Llama-8B"
BERT_80M = "bert-80M"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_PRO = "gemini-2.5-pro"
GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
CLAUDE_SONNET_5 = "claude-sonnet-5"
CLAUDE_OPUS_5 = "claude-opus-5"
GPT_4_1 = "gpt-4.1"
GPT_4_1_MINI = "gpt-4.1_mini"
GPT_4_1_NANO = "gpt-4.1_nano"
GPT_5_5 = "gpt-5.5"
GPT_5_4_MINI = "gpt-5.4_mini"
LLAMA_70B_TURBO = "Llama-70B-turbo"
LLAMA_405B_TURBO = "Llama-405B-turbo"

INPUT_TOKENS = "input_tokens"
OUTPUT_TOKENS = "output_tokens"
NUM_REQUESTS = "num_requests"
MODEL_ID = "model_id"

MODEL_PRICING_PER_1M = {  # prices in $/1M tokens
    # https://api.together.ai/models/togethercomputer/m2-bert-80M-32k-retrieval
    Model.BERT_80M: {INPUT: 0.01, OUTPUT: 0.01},
    # Ref
    Model.LLAMA_8B: {INPUT: 0.20, OUTPUT: 0.20},
    # https://platform.openai.com/docs/pricing
    # - gpt 5.5: 5.00\$ (input), 30.00\$ (output)
    Model.GPT_5_5: {INPUT: 5.00, OUTPUT: 30.00},
    # - gpt 4.1: 2.00\$ (input), 8.00\$ (output), batch prices half
    Model.GPT_4_1: {INPUT: 2, OUTPUT: 8},
    # - gpt 5.4-mini: 0.75\$ (input), 4.50\$ (output)
    Model.GPT_5_4_MINI: {INPUT: 0.75, OUTPUT: 4.50},
    # - gpt 4.1-mini: 0.40\$ (input), 1.60\$ (output), batch prices half
    Model.GPT_4_1_MINI: {INPUT: 0.4, OUTPUT: 1.6},
    # - gpt 4.1-nano: 0.10\$ (input), 0.40\$ (output), batch prices half
    Model.GPT_4_1_NANO: {INPUT: 0.1, OUTPUT: 0.4},
    # https://cloud.google.com/vertex-ai/generative-ai/pricing
    # - gemini 2.5 pro (<=200k context): 1.25\$ (input), 10.00\$ (output)
    Model.GEMINI_2_5_PRO: {INPUT: 1.25, OUTPUT: 10.00},
    # - gemini 3 flash preview: 0.50\$ (input), 3.00\$ (output)
    Model.GEMINI_3_FLASH_PREVIEW: {INPUT: 0.50, OUTPUT: 3.00},
    # - gemini 2.5 flash: 0.30\$ (input), 2.50\$ (output)
    Model.GEMINI_2_5_FLASH: {INPUT: 0.30, OUTPUT: 2.50},
    # - gemini 2.5 flash-lite: 0.10\$ (input), 0.40\$ (output)
    Model.GEMINI_2_5_FLASH_LITE: {INPUT: 0.10, OUTPUT: 0.40},
    # https://www.anthropic.com/pricing#api
    # - claude opus 5: 5.00\$ (input), 25.00\$ (output)
    Model.CLAUDE_OPUS_5: {INPUT: 5.00, OUTPUT: 25.00},
    # - claude sonnet 5: 3.00\$ (input), 15.00\$ (output)
    Model.CLAUDE_SONNET_5: {INPUT: 3.00, OUTPUT: 15.00},
    # - claude haiku 4.5: 1.00\$ (input), 5.00\$ (output)
    Model.CLAUDE_HAIKU_4_5: {INPUT: 1.00, OUTPUT: 5.00},
    # https://api.together.ai/models/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
    Model.LLAMA_405B_TURBO: {INPUT: 3.50, OUTPUT: 3.50},
    # https://api.together.ai/models/meta-llama/Llama-3.3-70B-Instruct-Turbo
    Model.LLAMA_70B_TURBO: {INPUT: 0.88, OUTPUT: 0.88},
}

# Custom models added at runtime (string-keyed, injected by the UI server)
CUSTOM_MODEL_PRICING: dict[str, dict] = {}


def _get_pricing(model_id) -> dict:
    """Lookup pricing for a Model enum OR a custom model string."""
    # Direct enum key
    if model_id in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[model_id]
    # String matching an enum's .value
    for k, v in MODEL_PRICING_PER_1M.items():
        if hasattr(k, "value") and k.value == model_id:
            return v
    # Custom model registered at runtime
    if isinstance(model_id, str) and model_id in CUSTOM_MODEL_PRICING:
        return CUSTOM_MODEL_PRICING[model_id]
    raise KeyError(f"No pricing found for model: {model_id!r}")


def demo_price_estimation(config: DemoScenario):
    large_model_costs = single_model_price(config.models.large_model, config)


def single_model_price(model_id, config: DemoScenario):
    price_per_request = single_model_price_per_request(model_id, config)
    price = price_per_request * config.requests.expected_requests

    return price


def single_model_price_per_request(model_id, config, wrapped=False):
    pricing = _get_pricing(model_id)
    inp_token_price = pricing[INPUT] / 10 ** 6
    out_token_price = pricing[OUTPUT] / 10 ** 6

    input_tokens = num_tokens(config.tokens.input) + num_tokens(config.tokens.prompt)
    if wrapped:
        input_tokens += num_tokens(config.tokens.wrapper_prompt)

    if wrapped:
        output_tokens = num_tokens(config.tokens.wrapped_output)
    else:
        output_tokens = num_tokens(config.tokens.output)

    price_per_request = inp_token_price * input_tokens + out_token_price * output_tokens

    return price_per_request


def poodle_price(config: DemoScenario):
    large_model_price_per_request_wrapped = \
        single_model_price_per_request(config.models.large_model, config, wrapped=True)
    large_model_price_per_request_non_wrapped = \
        single_model_price_per_request(config.models.large_model, config, wrapped=False)
    small_model_price_per_request = \
        single_model_price_per_request(config.models.small_model, config, wrapped=False)

    # first n request that go to lagre model
    large_model_requests = min(config.requests.expected_requests, config.requests.switch_after_n_items)
    large_model_requests_wrapped = config.tokens.wrapped_requests_percent * large_model_requests
    large_model_requests_non_wrapped = large_model_requests - large_model_requests_wrapped
    first_n_cost = (large_model_requests_non_wrapped * large_model_price_per_request_non_wrapped +
                    large_model_requests_wrapped * large_model_price_per_request_wrapped)

    # next requests to small model
    small_model_requests = max(config.requests.expected_requests - config.requests.switch_after_n_items, 0)
    small_model_cost = small_model_requests * small_model_price_per_request

    # every now and then send request to large model to check for data drift
    monitoring_requests = config.validation.validation_requests_percent * small_model_requests
    monitoring_cost = monitoring_requests * large_model_price_per_request_non_wrapped


    # TODO get better numbers here, probably cost will go lower
    # for now assume GPU time of 3 hours for model search and fine-tuning, have to validate
    # cost per hour ~1.2$/h -> ~4$
    model_search_and_dev_cost = config.dev.model_dev_costs

    return first_n_cost + small_model_cost + monitoring_cost + model_search_and_dev_cost


def analytical_break_even(config: DemoScenario):
    """Return the exact break-even request count N₀ where poodle becomes cheaper,
    or None if poodle never becomes cheaper.

    Derived by solving C_baseline(N) = C_poodle(N) analytically for N > N*:

        N₀ = [ N* · (A·(1-α-β) + α·B - S) + C_dev ] / (A·(1-β) - S)

    where A = large model cost/req (non-wrapped), B = large model cost/req (wrapped),
    S = small model cost/req, α = wrapped_requests_percent, β = validation_requests_percent.
    """
    N_star = config.requests.switch_after_n_items
    alpha  = config.tokens.wrapped_requests_percent
    beta   = config.validation.validation_requests_percent
    C_dev  = config.dev.model_dev_costs

    A = single_model_price_per_request(config.models.large_model, config, wrapped=False)
    B = single_model_price_per_request(config.models.large_model, config, wrapped=True)
    S = single_model_price_per_request(config.models.small_model, config, wrapped=False)

    denom = A * (1 - beta) - S
    if denom <= 0:
        return None  # small model is not cheaper per request — poodle never wins

    numer = N_star * (A * (1 - alpha - beta) + alpha * B - S) + C_dev
    N_0 = numer / denom

    if N_0 <= 0:
        return 0  # already cheaper from the first request

    return math.ceil(N_0)


def compare_single_model_and_poodle(config: DemoScenario):
    request_values = get_plot_x_ticks(config.requests.expected_requests)
    base_prices = {}
    poodle_prices = {}
    poodle_savings = {}
    for request in request_values:
        cfg = replace(config, requests=replace(config.requests, expected_requests=request))
        base_prices[request] = single_model_price(cfg.models.large_model, cfg)
        poodle_prices[request] = poodle_price(cfg)
        poodle_savings[request] = base_prices[request] - poodle_prices[request]

    return base_prices, poodle_prices, poodle_savings

def poodle_min_switch_time(
    time_for_model_dev,      # seconds — time to develop/validate the small model
    llm_throughput,          # items/s — throughput of the large LLM
    small_model_throughput,  # items/s — throughput of the small model after switch
    total_requests,          # int     — total expected requests in the scenario
    switch_req,              # int     — number of requests at which Poodle switches
):
    """Estimate the earliest time (seconds) at which Poodle can switch to the small model.
    if we get numbers here, we know we have valid numbers/measurements.

    Returns seconds, or None if inputs are insufficient.
    """

    # as long as we develop the surrogate model -> handle requests using the LLM
    # also as long as the user prevents the switch -> handle requests using the LLM
    items_processed_by_llm = max((time_for_model_dev * llm_throughput), switch_req)
    # the remaining requests are handled by the surrogate model
    items_processed_by_surrogate = total_requests - items_processed_by_llm

    llm_time = items_processed_by_llm/llm_throughput
    surrogate_time = items_processed_by_surrogate/small_model_throughput

    return llm_time + surrogate_time

if __name__ == '__main__':
    example_scenario = DemoScenario.get_example_scenario()
    base_prices, poodle_prices, poodle_savings = compare_single_model_and_poodle(example_scenario)

    print("Base prices:", base_prices)
    print("Poodle prices:", poodle_prices)
    print("Poodle savings:", poodle_savings)