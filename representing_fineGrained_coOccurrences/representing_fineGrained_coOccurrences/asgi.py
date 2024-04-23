"""
ASGI config for Fraud_Detection_in_Online_Payment.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'representing_fineGrained_coOccurrences.settings')

application = get_asgi_application()
