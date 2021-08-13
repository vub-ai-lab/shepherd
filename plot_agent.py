#!/usr/bin/env python3
import os
import sys

import django

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
    django.setup()

    from shepherd.models import EpisodeReturn

    # Select returns for the agent
    returns = EpisodeReturn.objects.filter(agent_id=int(sys.argv[1]))

    for ret in returns:
        print(ret.ret, ret.datetime)


if __name__ == '__main__':
    main()
