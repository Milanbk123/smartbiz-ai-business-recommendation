"""
Custom authentication backend for SmartBiz AI.
Allows authenticating using Email and Password.
"""

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model

User = get_user_model()


class EmailBackend(ModelBackend):
    """Authenticate using Email address and Password."""

    def authenticate(self, request, username=None, password=None, email=None, **kwargs):
        login_email = email or username
        if not login_email or not password:
            return None

        login_email = str(login_email).strip().lower()

        try:
            user = User.objects.get(email__iexact=login_email)
        except User.DoesNotExist:
            return None
        except User.MultipleObjectsReturned:
            user = User.objects.filter(email__iexact=login_email).first()

        if user and user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
