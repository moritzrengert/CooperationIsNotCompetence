from src.mechanisms.base import Mechanism


class NoMechanism(Mechanism):
    name = "none"

    def augment_observation(self, observation, context):
        return observation

    def augment_prompt(self, prompt, observation):
        return prompt

    def resolve(self, decisions, context):
        return {
            "implemented_choices": [d.choice for d in decisions],
            "contributors": None,
        }


class FischerBaselineMechanism(NoMechanism):
    # Backward-compatible alias for older configs that used mechanism: fischer_2004.
    name = "fischer_2004"
