from django.db import models
from django.utils.html import format_html, mark_safe
from django.contrib.auth.models import User

import uuid

class Algorithm(models.Model):
    name = models.CharField("algorithm's name", max_length=32, help_text="Name of the Reinforcement Learning algorithm (for instance, Proximal Policy Optimization (PPO)).")
    can_continuous_actions = models.BooleanField("Compatible with continuous actions", help_text="Not all RL algorithms are compatible with continuous actions (a continuous action is a real number); this field indiscates if this specific RL algorithm can handle continuous actions, True or False. In contrast, we consider that all RL algorithms available via Shepherd are compatible with discrete actions by default.")

    def __str__(self):
        return self.name

class Parameter(models.Model):
    class ParamType(models.IntegerChoices):
        BOOL = 1
        INT = 2
        FLOAT = 3
        STR = 4

    name = models.CharField("name given to the parameter", max_length=32)
    algo = models.ForeignKey(Algorithm, on_delete=models.CASCADE, verbose_name="Algorithm that has this parameter")
    t = models.IntegerField(choices=ParamType.choices, verbose_name="Type of the parameter")

    default_value = models.CharField(max_length=64, null=True)

    def __str__(self):
        return self.algo.name + ': ' + self.name + ' (' + self.get_t_display() + ')'


class Agent(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="user that owns this agent", help_text="The owner of an agent is the user/client of Shepherd.")
    algo = models.ForeignKey(Algorithm, null=True, on_delete=models.SET_NULL, verbose_name="RL algorithm executed by this agent", help_text="Reinforcement Learning algorithm that the agent is running (for instance, PPO, SAC, BDPI, etc).")
    name = models.CharField("Descriptive name given to the agent", max_length=128)
    enable_learning = models.BooleanField("Enable learning", help_text="True for normal behavior. If False, the agent performs its task (it selects actions) but ignores the rewards it gets and does not update its policy. Useful in production of stable behavior is required. Don't forget to RESTART the agent if you change this parameter.", default=True)

    action_space = models.TextField('Action space JSON', help_text="Syntax: either an integer like \"4\", indicating a discrete number of actions (integers from 0 to 3); a list of 3 elements (shape, low, high) like \"[[28], 0.0, 1.0]\" indicating a continuous action made of a list of 28 floating-point numbers between 0 and 1; or a dictionary (see Observation Space JSON, as it is only used for observations)")
    observation_space = models.TextField('Observation space JSON', help_text="Same syntax as Action Space JSON. It is also possible to have a dictionary of keys to integers or lists (as described above), for environments that produce readings from multiple sensors, such as 16 distance sensors and an 80x80 color camera: {\"distance\": [[16], 0.0, 1.0], \"camera\": [[80, 80, 3], 0, 255]}.")

    creation_time = models.DateTimeField('Creation of the agent', auto_now_add=True)
    last_activity_time = models.DateTimeField('Date of last usage of the agent', auto_now_add=True)
    max_percent_cpu_usage = models.FloatField('Maximum CPU usage in percentage points', default=100.0, help_text="100.0 = 1 full CPU used every one-minute interval")

    parameters = models.ManyToManyField(Parameter, through='ParameterValue', help_text="Parameters used to configure the learning algorithm used by the agent")

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

    def special_actions(self):
        id = str(self.id)

        return format_html(
            f"""
            <a href="/shepherd/generate_zip/?agent_id={id}">Download ZIP (if it exists)</a> &bull;
            <a href="/shepherd/delete_zip/?agent_id={id}">Delete ZIP</a> &bull;
            <a href="/shepherd/delete_curve/?agent_id={id}">Delete learning curve</a> &bull;
            <a href="/shepherd/export_curve_CSV/?agent_id={id}">Export curve CSV</a> &bull;
            <a href="/shepherd/kill_processes/?agent_id={id}">Restart agent</a> (will invalidate session keys)
            """
        )

    def __str__(self):
        return f"{self.id} of {self.owner.username}: {self.name}"

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

class ParameterValue(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, verbose_name="Agent for which this parameter has a value")
    param = models.ForeignKey(Parameter, on_delete=models.CASCADE, verbose_name="Parameter")
    value = models.CharField(max_length=64, null=True)

    def __str__(self):
        return str(self.agent) + ' param ' + self.param.name + '=' + self.value
