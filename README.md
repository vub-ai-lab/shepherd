# Distributed Reinforcement Learning over HTTP

Shepherd is a web application that allows to do [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) in the field, with the agent being remotely accessed through the web. [Documentation](https://helepli-shepherd-docs.readthedocs.io/en/latest/)

Use-cases:

- An Arduino with a Wifi dongle offloads learning and prediction to a remote Shepherd server
- A web-page runs entirely client side (in Javascript) and performs Shepherd XHR requests to obtain actions, such as in [this example](https://bored-inthecity.be/)
- A MATLAB simulation (even a graphical one with Simulink) performs HTTP requests to interact with a Shepherd server, written in Python
- Several programmable logic controllers, that perform the same task, communicate with a Shepherd server. They even get to learn from each other experiences transparently!

The main property of Shepherd is that replicas of a single environment can all learn together and benefit from each other. For instance:

- The programmable logic controllers mentioned in Use-cases.
- A single smartphone application that runs concurrently on thousands of smartphones. Each phone user interacting with the application will lead to a better experience to all the other users, almost instantaneously.
- Audioguides in a museum that all learn together what interests people.
- Smart thermostats, electric car batteries, solar panels, smart power plugs, etc, that all individually perform a task such as optimizing energy usage or maximizing comfort, and need to learn from each other.

**Interested and want to benefit from a learning agent in your application/robot/machine?** The Vrije Universiteit Brussel Artificial Intelligence lab offers consulting by PhDs, experts in Reinforcement Learning and aware of the latest research, to make your project happen. Depending on your needs, we can tell you whether Reinforcement Learning will solve your problem (we are not pushing it at any cost), formalize your dream as a Reinforcement Learning task, implement the software you may need on your side to interact with Shepherd, and deploy a Shepherd server for you. We also gladly answer questions and emails. Shepherd is [Free and Open Source Software](#license). The emails and questions are free too.

1. For quick technical questions, please drop an email to `denis.steckelmacher@vub.be` (he will try to answer the same day or a few days later)
2. For closer collaboration, or to schedule a meeting about how we can help you with your potential Reinforcement Learning use-case, plase contact our business developer, `gill.balcaen@vub.be`

[![VUB AI Lab](https://ai.vub.ac.be/wp-content/uploads/2019/08/VUB-AI_RGB-1-800x223.png)](https://ai.vub.ac.be/)

# Installing

Shepherd is a Django application, written in Python (tested on Python 3.10). Shepherd does not need to be installed, but can be set up with

```
git clone https://github.com/vub-ai-lab/shepherd
cd shepherd
pip3 install -r requirements.txt

python3 manage.py migrate
python3 manage.py createsuperuser
python3 manage.py runserver
```

This runs Shepherd locally, listening on `localhost:8000`.

# Usage - Admin side

The amin interface of Shepherd allows the super user (created above with `createsuperuser`) to create and manage:

- Shepherd users: people who have access to the admin interface and can configure agents, observe how they learn, save checkpoints, ...
- Shepherd agents: configurations of Reinforcement Learning agents, such as what algorithm is to be used, what observations they receive, what actions they produce, etc. There is one Shepherd agent per "application", for instance one Shepherd agent for the collection of audioguides in the example above. Each Shepherd user can be the owner of several Shepherd agents.
- Shepherd algorithms and parameters, that describe algorithms available on Shepherd. These tables are not pre-populated yet. The version of stable_baselines3 that Shepherd ships with has `PPO`, `SAC`, `DDPG` and `TD3`.
- API keys, that will allow remote access to the Shepherd agents. A Shepherd agent may have several API keys (one for the iOS and one for the Android app for instance).

It is available at `localhost:8000/admin/`.

# Usage - Environment side

In Reinforcement Learning, the component that produces observations and executes actions, such as a robot arm, a web-page, a smartphone application, a thermostat or an IoT device, is called the **environment**. The environment is a device that executes code, in any programming language, that sends observations to Shepherd and receives actions back. The communication is done over HTTP (on port 8000 with the default settings), using JSON.

## Obtain a session key

The environment needs to know the IP address of the Shepherd server, its port, and the API key to use to interact with a specific Shepherd agent. With this information, a session key can be obtained by performing a `POST` request on `/shepherd/login_user/`, with, as data, the string representation of this JSON object:

```
{"apikey": my_api_key}
```

The API key is a string.

This request leads to a response that is a JSON object with two fields:

- `ok`, set to `true` when everything is fine (otherwise, `error` contains a detailed error message, such as when the API key is invalid)
- `session_key`, a string to use later to communicate with Shepherd.

## Obtain an action

The environment can then produce observations (sensor readings, user activity, camera images, ...) and send them to Shepherd with a `POST` request on `/shepherd/env/`, with as data the string representation of the following JSON object:

```
{
    "session_key": session_key,
    "obs": list of floating-point values,
    "reward": float,
    "done": boolean,
    "info": {}
}
```

The response is a JSON object with the following attributes:

- `action`, either an integer (the number of the action to perform) or a list of floating-point values, depending on how the Shepherd agent is configured.
- `error`, in case there is an error, a detailed error message.

`shepherd/make_requests.py` in this repository contains an example Shepherd client written in Python.

# License

Shepherd is licensed under the GNU Affero General Public License (see `agpl-3.0.txt` and `COPYING`). This means that:

- You can download, use and modify Shepherd freely.
- You can redistribute Shepherd, modified or not, for free or in exchange of money, provided that:
  - You redistribute Shepherd under the same license, the GNU Affero GPL
  - You keep our copyright notices and mention our complete name, the VU Brussels AI Lab, applied AI and consulting department, ai.vub.ac.be
- The specific part of the GNU *Affero* GPL is that, if you modify Shepherd (improve, extend or any other change), you have to make your changes public, easily accessible (such as on Github), also under the GNU Affero GPL, even if you don't redistribute Shepherd. This means that if you modify Shepherd and offer it to customers through a web portal (they, themselves, do not receive a copy of Shepherd), your changes still have to be made public.

This license is to ensure that everyone, companies, users, developers, agencies, consultants, etc, can all have access to a single public version of Shepherd, that offers the best of Reinforcement Learning to the world.

Note that `stable_baselines3` is a [separate Open Source project](https://github.com/DLR-RM/stable-baselines3) that is distributed under the MIT license, a much more permissive license that allows almost-unrestricted modification, use, redistribution and sublicensing.
