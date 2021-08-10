from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Algorithm(models.Model):
    name = models.CharField("algorithm's name", max_length=32)
    can_continuous_actions = models.BooleanField("Compatible with continuous actions")
    
    def __str__(self):
        return self.name

class Agent(models.Model):
    POLICIES = [
        ('MlpPolicy', 'MlpPolicy: states that are lists of floats'),
        ('CnnPolicy', 'CnnPolicy: states that are images or 2D arrays'),
    ]

    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="user that owns this agent")
    algo = models.ForeignKey(Algorithm, null=True, on_delete=models.SET_NULL, verbose_name="RL algorithm executed by this agent")
    policy = models.CharField("Policy class", max_length=32, choices=POLICIES)
    action_space = models.TextField('Action space JSON')
    observation_space = models.TextField('Observation space JSON')
    
    def __str__(self):
        return str(self.id) + ': ' + ('(none)' if self.algo is None else self.algo.name) + ' agent of ' + self.owner.username

class EpisodeReturn(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, verbose_name="Agent")
    datetime = models.DateTimeField(auto_now_add=True, verbose_name="Date-time of the episode")
    ret = models.FloatField(verbose_name="Return obtained during that episode")

    def __str__(self):
        return str(self.agent) + " episode return " + str(self.ret)

class Parameter(models.Model):
    class ParamType(models.IntegerChoices):
        BOOL = 1
        INT = 2
        FLOAT = 3
        STR = 4
    
    name = models.CharField("name given to the parameter", max_length=32)
    algo = models.ForeignKey(Algorithm, on_delete=models.CASCADE, verbose_name="Algorithms that has this parameter")
    t = models.IntegerField(choices=ParamType.choices, verbose_name="Type of the parameter")
    
    value_bool = models.BooleanField(null=True)
    value_int = models.IntegerField(null=True)
    value_float = models.FloatField(null=True)
    value_str = models.CharField(max_length=64, null=True)

    def __str__(self):
        return self.name + ' of type ' + self.get_t_display() + ' for algorithm ' + self.algo.name

class ParameterValue(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, verbose_name="Agent for which this parameter has a value")
    param = models.ForeignKey(Parameter, on_delete=models.CASCADE, verbose_name="Parameter set to the value")
    value_bool = models.BooleanField(null=True)
    value_int = models.IntegerField(null=True)
    value_float = models.FloatField(null=True)
    value_str = models.CharField(max_length=64, null=True)
    
    def __str__(self):
        return str(self.agent) + ' param ' + self.param.name + '=' + str(self.value_bool) + ' ' + str(self.value_int) + ' ' + str(self.value_float) + ' ' + self.value_str
