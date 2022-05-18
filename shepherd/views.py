# This file is part of Shepherd.
#
# Shepherd is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Shepherd is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with Shepherd. If not, see <https://www.gnu.org/licenses/>.

from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User # import the database classes
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect, Http404
from django.conf import settings
from django.utils import timezone

import json
import numpy as np
import torch as th
import gym
import psutil
import shutil
import csv

import ast
import sys
import datetime
import time
import io
import os
import traceback
import uuid
import multiprocessing

from .models import *
from .worker_process import get_latest_model, spawn_worker_process

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROCESS_POOLS = {}
SESSIONS = {}

def str_to_json(space):
    try:
        return int(space)
    except ValueError:
        return ast.literal_eval(space)

def action_to_json(a):
    if a is None:
        return None
    elif isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return int(a)

def json_to_observation(json):
    if isinstance(json, int):
        return json
    elif isinstance(json, list):
        return np.array(json)
    elif isinstance(json, dict):
        return {k: json_to_observation(v) for k, v in json.items()}
    else:
        raise Exception("Observation obtained from the environment must be an integer, a list of floats or a dictionary of integers or lists of floats. Got " + repr(json))

def get_param_values_from_database(agent):
    """
    Retrieve parameter value for that agent, with that algorithm, either set by the user or the default value of the parameter
    """
    param_values = ParameterValue.objects.select_related('param').filter(agent=agent)

    arguments = {}

    for pv in param_values:
        key = pv.param.name
        value = pv.value

        # Cast the value
        casts = {
            Parameter.ParamType.BOOL: bool,
            Parameter.ParamType.INT: int,
            Parameter.ParamType.FLOAT: float,
            Parameter.ParamType.STR: str,
        }

        value = casts[pv.param.t](value)

        arguments[key] = value

    return arguments

class AgentProcessPool:
    """ List of Process instances for a particular agent_id. Used to produce advice
    """

    def __init__(self, agent):
        self.processes = []
        self.agent = agent

        self.last_time_quota_checked = time.monotonic()
        self.meets_quota = True

        self.load_spaces_from_db()

    def load_spaces_from_db(self):
        self.agent.refresh_from_db()

        self.observation_space = str_to_json(self.agent.observation_space)
        self.action_space = str_to_json(self.agent.action_space)

    def meets_cputime_quota(self):
        if (time.monotonic() - self.last_time_quota_checked) > 60.:
            # Checked quota more than one minute ago, re-check
            total_percent = 0.0

            for p in self.processes:
                total_percent += p["psutil"].cpu_percent()

            self.agent.refresh_from_db()
            self.meets_quota = (total_percent < self.agent.max_percent_cpu_usage)
            self.last_time_quota_checked = time.monotonic()

        return self.meets_quota

    def allocate_process(self):
        """ Return a process for a new episode. It is either an existing available
            process, or a new process.
        """
        for p in self.processes:
            if p["available"]:
                p["available"] = False
                return p

        # Make a new process, none of the existing ones was available
        to_process = multiprocessing.Queue()
        from_process = multiprocessing.Queue()

        p = spawn_worker_process(
            to_process,
            from_process,
            self.action_space,
            self.observation_space,
            self.agent.algo.name,
            get_param_values_from_database(self.agent),
            str(self.agent.owner) + '_' + self.agent.algo.name + '_' + str(self.agent.id),
            not self.agent.enable_learning
        )

        self.processes.append({
            "process": p,
            "psutil": psutil.Process(p.pid),
            "available": False,
            "to_process": to_process,
            "from_process": from_process,
            "owner_session": None,
            "last_activity_time": time.monotonic(),
            "start_time": time.monotonic(),
            "id": len(self.processes)
        })

        p = self.processes[-1]
        p["psutil"].cpu_percent()  # First read of the CPU counters for quota

        return p

    def get_process_for_session(self, id, session_key):
        """ Get the process for a session. It is processes[id], except in the
            case that this process has already been re-used by another session, in
            which case a new process has to be allocated for this session.
        """
        if id >= len(self.processes):
            # Processes got cleared, need to create a new one
            p = self.allocate_process()
        else:
            p = self.processes[id]

        if (p["owner_session"] is not None) and (p["owner_session"] != session_key):
            # The thread has been given to another session, get another one
            p = self.allocate_process()

        p["available"] = False
        p["owner_session"] = session_key

        return p

    def cleanup_processes(self):
        """ Explore the processes and remove those that have not seen any recent activity.
        """
        i = 0
        current_time = time.monotonic()

        while i < len(self.processes):
            p = self.processes[i]

            if (current_time - p["last_activity_time"]) > 30*60:
                # No activity since 30 minutes
                p["to_process"].put({"type": "stop"})
                p["process"].join()

                del self.processes[i]
            else:
                i += 1

    def kill_processes(self):
        """ Kill all the processes in this pool, useful when the Agent object has changed in the database.
        """
        for p in self.processes:
            p["to_process"].put({"type": "stop"})

        for p in self.processes:
            p["process"].join()

        self.processes.clear()

        # Reload the agent
        self.load_spaces_from_db()

    def produce_advice_from_other_processes(self, obs, current_process):
        """ Ask all the threads to produce an advice vector for observation obs
        """
        if isinstance(self.action_space, int):
            num_actions = self.action_space
        else:
            num_actions = None

        # Advise current_process with every process that has a last_activity_time
        # later than the start_time of current_process
        processes = []

        for p in self.processes:
            if (p is not current_process) and (p["last_activity_time"] > current_process["start_time"]):
                processes.append(p)

        if len(processes) == 0:
            # No advice available, return some default (neutral) advice
            if num_actions is None:
                advice = self.get_advice_from_process(current_process, obs) # TODO: Use ourselves as advice. This works for mean (mean combined with mean leads to mean), but not std (std combined with std leads to std / 2*sqrt(std))
            else:
                advice = np.ones((num_actions,), dtype=np.float32) / num_actions

            return advice

        # Average the received advices (this will average the probability distributions for discrete actions, or the mean/variances for continuous actions)
        # TODO: For continuous actions, the averaging has no mathematical meaning (and does not increase variance, for instance). We need to find something else to do.
        advices = [self.get_advice_from_process(p, obs) for p in processes]
        advice = np.mean(advices, axis=0)

        return advice[0]

    def get_advice_from_process(self, p, obs):
        """ Send a request to the process for advice, and return it
        """
        p["to_process"].put({"type": "advice", "obs": obs})
        return p["from_process"].get()["advice"]

    def get_action_from_process(self, p, obs, reward, done, info):
        """ Send an experience to a process and get an action from it
        """
        # Update the last activity time of the agent
        self.agent.last_activity_time = timezone.now()
        self.agent.save()

        # Make a dictionary observation
        if not isinstance(obs, dict):
            # Single-observation environment, force it to be a dictionary to match ShepherdEnv
            obs = {"obs": obs}   # See shepherdEnv.py for keys

        # Add advice to the state
        obs["advice"] = self.produce_advice_from_other_processes(obs, p)

        # Send to the process and receive the action
        p["last_activity_time"] = time.monotonic()
        p["to_process"].put({"type": "action", "obs_tuple": (obs, reward, done, info)})
        response = p["from_process"].get()

        assert done == ("return" in response)

        if "return" in response:
            # The episode finished. Log it, then mark the process as available
            log_entry = EpisodeReturn()
            log_entry.agent = self.agent
            log_entry.ret = response["return"]
            log_entry.save()

            # Reset cumulative_reward for the next episode
            p["available"] = True
            p["owner_session"] = None

        return response["action"]

def shepherd_wrap(view):
    """ This is what allows a Shepherd server to answer JSON queries from Javascript pages on different servers.
    """
    def inner(request):
        print('Got request', request.body)

        try:
            response = view(request)
        except Exception as e:
            stacktrace = traceback.format_exc()
            response = JsonResponse({'ok': False, 'error': str(e), 'traceback': stacktrace})

        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Credentials"] = "true"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Max-Age"] = "1000"
        response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
        response["X-Frame-Options"] = "SAMEORIGIN"

        return response

    return inner

@csrf_exempt
@shepherd_wrap
def login_user(request):
    global PROCESS_POOLS, SESSIONS

    data = json.loads(request.body)

    # Look for an API key
    try:
        key = APIKey.objects.select_related('agent').get(key=data['apikey'])
        agent = key.agent
    except AttributeError:
        raise Exception("JSON query data must contain an apikey element")
    except APIKey.DoesNotExist:
        raise Exception("API Key not found in the database")

    # Create a pool for the agent_id if necessary
    agent_id = agent.id

    if agent_id not in PROCESS_POOLS:
        PROCESS_POOLS[agent_id] = AgentProcessPool(agent)

    pool = PROCESS_POOLS[agent_id]

    # Allocate a thread for the client
    p = pool.allocate_process()
    pid = p["id"]

    session_key = str(uuid.uuid4())
    session = {
        'process_id': pid,
        'agent_id': agent_id
    }
    SESSIONS[session_key] = session

    return JsonResponse({'ok': True, 'session_key': session_key})

@csrf_exempt
@shepherd_wrap
def env(request):
    data = json.loads(request.body)

    # Load the session from the session_key in the request
    if 'session_key' not in data:
        raise Exception('No session_key in the request. Please use /login_user/ to get a session key from an API key')

    session_key = data['session_key']

    if session_key not in SESSIONS:
        raise Exception('Session key not in the sessions known to this instance of Shepherd. The server may have restarted since you got your session key. Please use /login_user/ again to get a fresh session key')

    session = SESSIONS[session_key]

    # Find the worker process
    pid = session['process_id']
    agent_id = session['agent_id']

    try:
        pool = PROCESS_POOLS[agent_id]
        p = pool.get_process_for_session(pid, data['session_key'])
        pid = p["id"]

        session['process_id'] = pid
    except KeyError:
        raise Exception("Could not find thread in the current threads pool")

    # Time quota management
    if not pool.meets_cputime_quota():
        raise Exception("CPU time quota exceeded. Please re-try this request in a few moments")

    # Ask the process for an action
    action = pool.get_action_from_process(
        p,
        json_to_observation(data['obs']),
        data['reward'],
        data['done'],
        data.get('info', {})
    )

    # Cleanup the processes
    pool.cleanup_processes()

    return JsonResponse({'action': action_to_json(action)})

def get_agent_for_request(request):
    agent_id = request.GET['agent_id']

    try:
        agent = Agent.objects.get(id=agent_id)

        if agent.owner_id != request.user.id and not request.user.is_superuser:
            raise Http404("No such agent for this user")
    except Agent.DoesNotExist:
        raise Http404("Unknown agent")

    return agent

# http://localhost:5000/shepherd/send_curve/?agent_id=1
@login_required
def send_curve(request):
    """ Show the agent's learning curve on the admin site.
    """
    # Find agent
    agent = get_agent_for_request(request)

    # Select returns for the agent
    ep_returns = EpisodeReturn.objects.filter(agent=agent)
    returns = [] # actual floats

    for ret in ep_returns:
        returns.append(ret.ret)

    # save plot in an in-memory file
    plot = plt.figure()

    plt.plot(returns)
    plt.xlabel('Episode number')
    plt.ylabel('Cumulative reward')

    buf = io.BytesIO()
    plot.savefig(buf, format='svg')
    buf.seek(0)
    plt.close(plot)

    return HttpResponse(buf, content_type="image/svg+xml")

@login_required
def generate_zip(request):
    """ Make the latest agent's model zip file available for download on the amdin site.
    """
    # Find agent
    agent = get_agent_for_request(request)

    # Directory where the zip file is
    savename = str(agent.owner) + '_' + agent.algo.name + '_' + str(agent.id)
    zipname = get_latest_model(savename)

    if zipname is None:
        raise Http404("No log yet for this agent")

    # Return the zip data
    with open(zipname, 'rb') as f:
        return HttpResponse(f, headers={'Content-Type': 'application/zip', 'Content-Disposition': 'attachment; filename="' + savename + '.zip"'})

def go_back_to_admin_with_message(request, message):
    if message is not None:
        messages.add_message(request, messages.SUCCESS, message)

    return HttpResponseRedirect("/admin/shepherd/agent/%s/change/" % request.GET['agent_id'])

@login_required
def delete_zip(request):
    """ Delete all model zip files associated to the user's agent. It's a button on the amdin site.
    """
    # Find agent
    agent = get_agent_for_request(request)

    # Directory where the zip file is
    savename = str(agent.owner) + '_' + agent.algo.name + '_' + str(agent.id)

    try:
        shutil.rmtree(savename) # deletes the whole directory
        messages.add_message(request, messages.SUCCESS, "ZIP files removed.")
    except FileNotFoundError:
        messages.add_message(request, messages.INFO, "No ZIP files for this agent yet, so none deleted.")

    return go_back_to_admin_with_message(request, None)

@login_required
def delete_curve(request):
    """ Delete all episode returns for this agent.
    """
    # Find agent
    agent = get_agent_for_request(request)

    # Delete all episode return records for the agent
    EpisodeReturn.objects.filter(agent=agent).delete()

    return go_back_to_admin_with_message(request,  "Learning curve deleted.")

@login_required
def export_curve_CSV(request):
    """ Export curve in CSV file to be downloaded on the admin site.
    """
    # Find agent
    agent = get_agent_for_request(request)

    # Select returns for the agent
    ep_returns = EpisodeReturn.objects.filter(agent=agent)
    returns = [] # actual floats

    for ret in ep_returns:
        returns.append(ret.ret)

        # Create the HttpResponse object with the appropriate CSV header.
    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename="curve.csv"'},
    )

    writer = csv.writer(response)
    writer.writerow(['episode', 'reward'])
    for i in range(len(returns)):
        writer.writerow([str(i), str(returns[i])])

    return response

@login_required
def kill_processes(request):
    """ Kill all the processes for this agent
    """
    agent = get_agent_for_request(request)
    agent_id = agent.id

    try:
        pool = PROCESS_POOLS[agent_id]
        pool.kill_processes()
    except KeyError:
        pass

    return go_back_to_admin_with_message(request, "Agent re-started. Any client will need to obtain a new session key.")
