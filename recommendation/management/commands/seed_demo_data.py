"""
Management command disabled.
Synthetic demo dataset generation is permanently disabled in this project.
All data is derived exclusively from dataset/online+retail/Online Retail.xlsx.
"""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Disabled command (Synthetic demo dataset generation is disabled).'

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.WARNING("Synthetic demo generation is disabled. This project uses dataset/online+retail/Online Retail.xlsx.")
        )
