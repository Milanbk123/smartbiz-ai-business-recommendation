"""
Django management command to train the ML pipeline on Online Retail.xlsx dataset.
"""

from django.core.management.base import BaseCommand
from ml.pipeline import train_pipeline


class Command(BaseCommand):
    help = 'Trains the K-Means clustering & recommendation pipeline using dataset/online+retail/Online Retail.xlsx'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reprocessing of raw Excel file',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting ML training pipeline on Online Retail.xlsx..."))
        force = options.get('force', False)
        metadata = train_pipeline(force_reprocess=force)

        self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Pipeline training complete!"))
        self.stdout.write(f"Cleaned Transactions: {metadata['cleaned_rows']:,}")
        self.stdout.write(f"Unique Customers    : {metadata['num_customers']:,}")
        self.stdout.write(f"Selected Clusters K : {metadata['selected_k']}")
        self.stdout.write(f"Silhouette Score    : {metadata['best_silhouette_score']}")
