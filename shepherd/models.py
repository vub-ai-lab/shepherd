from django.db import models
from django.utils.html import format_html, mark_safe
from django.contrib.auth.models import User

import uuid

class Algorithm(models.Model):
    name = models.CharField("algorithm's name", max_length=32, help_text="Name of the Reinforcement Learning algorithm (for instance, Proximal Policy Optimization (PPO)).")
    can_continuous_actions = models.BooleanField("Compatible with continuous actions", help_text="Not all RL algorithms are compatible with continuous actions (a continuous action is a real number); this field indiscates if this specific RL algorithm can handle continuous actions, True or False. In contrast, we consider that all RL algorithms available via Shepherd are compatible with discrete actions by default.")

    def __str__(self):
        return self.name

class Agent(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="user that owns this agent", help_text="The owner of an agent is the user/client of Shepherd.")
    algo = models.ForeignKey(Algorithm, null=True, on_delete=models.SET_NULL, verbose_name="RL algorithm executed by this agent", help_text="Reinforcement Learning algorithm that the agent is running (for instance, PPO, SAC, BDPI, etc).")

    action_space = models.TextField('Action space JSON', help_text="")
    observation_space = models.TextField('Observation space JSON', help_text="")

    creation_time = models.DateTimeField('Creation of the agent', auto_now_add=True)
    last_activity_time = models.DateTimeField('Date of last usage of the agent', auto_now_add=True)
    max_percent_cpu_usage = models.FloatField('Maximum CPU usage in percentage points (100.0 = 1 full CPU used every one-minute interval)', default=100.0)

    def learning_curve(self):
        img_tag = '<img id=\"learning_curve\" src="/shepherd/send_curve/?agent_id=%s&">' % self.id
        refresh = """
<script>
window.onload = function() {
    var image = document.getElementById("learning_curve");

    function updateImage() {
        image.src = image.src.split("&")[0] + "&" + new Date().getTime();
    }

    setInterval(updateImage, 1000);
}
</script>
"""

        return mark_safe(img_tag + refresh)

    def latest_agent_model_zip_file(self):
        return format_html(
            '<a href="/shepherd/generate_zip/?agent_id=%s">Download ZIP (if it exists)</a>' % self.id
        )

    def __str__(self):
        return str(self.id) + ': ' + ('(none)' if self.algo is None else self.algo.name) + ' agent of ' + self.owner.username

class EpisodeReturn(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, verbose_name="Agent")
    datetime = models.DateTimeField(auto_now_add=True, verbose_name="Date-time of the episode")
    ret = models.FloatField(verbose_name="Return obtained during that episode")

    def __str__(self):
        return str(self.agent) + " episode return " + str(self.ret)

def make_api_key():
    """ Generate a random API key using the UUID module. API keys look like "4bf9eeca-d1e6-45a2-997a-0d45273a8cd8"
    """
    return str(uuid.uuid4())

class APIKey(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, verbose_name="Agent identified by this API key")
    key = models.CharField('API Key string', max_length=36, default=make_api_key(), unique=True)

    def __str__(self):
        return self.key + " for agent " + str(self.agent)

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
