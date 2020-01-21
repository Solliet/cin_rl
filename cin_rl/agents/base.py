import abc


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self):
        pass


class ValueFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, states):
        raise NotImplementedError()


class PolicyGradient(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def action_probabilities(self, states):
        raise NotImplementedError()

class ActorCritic(PolicyGradient, ValueFunction):
    pass

