"""
Django automated test suite for SmartBiz AI recommendation system.
Tests authentication, business customer creation, purchase transactions,
RFM calculations, ML prediction, recommendation engine, form validations, and security access.
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta

from recommendation.models import (
    BusinessProfile, BusinessCustomer, PurchaseTransaction, RecommendationHistory
)
from recommendation.forms import (
    RegistrationForm, EmailLoginForm, CustomerLookupForm,
    AddBusinessCustomerForm, RecordPurchaseForm
)
from ml.recommender import (
    recommend_for_customer, predict_segment_and_recommend,
    recommend_for_business_customer
)


class AuthenticationTests(TestCase):
    """Test business registration, email login, logout, and access control."""

    def setUp(self):
        self.client = Client()
        self.register_url = reverse('register')
        self.login_url = reverse('login')
        self.logout_url = reverse('logout')
        self.dashboard_url = reverse('dashboard')

    def test_business_registration_success(self):
        payload = {
            'business_name': 'Aura Fashion Boutique',
            'admin_name': 'Anita Roy',
            'email': 'anita@aurafashion.com',
            'password': 'SecurePassword123!'
        }
        response = self.client.post(self.register_url, payload)
        self.assertRedirects(response, self.dashboard_url)

        # Verify User created
        user = User.objects.filter(email='anita@aurafashion.com').first()
        self.assertIsNotNone(user)
        # Verify password is NOT plain text
        self.assertNotEqual(user.password, 'SecurePassword123!')
        self.assertTrue(user.check_password('SecurePassword123!'))

        # Verify Business Profile created
        profile = getattr(user, 'business_profile', None)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.business_name, 'Aura Fashion Boutique')
        self.assertEqual(profile.admin_name, 'Anita Roy')

    def test_registration_duplicate_email_fails(self):
        User.objects.create_user(
            username='existing@example.com',
            email='existing@example.com',
            password='password123'
        )
        payload = {
            'business_name': 'Duplicate Business',
            'admin_name': 'Copy Admin',
            'email': 'EXISTING@EXAMPLE.COM',
            'password': 'password123'
        }
        response = self.client.post(self.register_url, payload)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(
            response.context['form'], 'email',
            "An account with this email address already exists. Please sign in."
        )

    def test_email_login_success_and_logout(self):
        user = User.objects.create_user(
            username='owner@store.com',
            email='owner@store.com',
            password='mysecretpassword'
        )
        BusinessProfile.objects.update_or_create(
            user=user,
            defaults={'business_name': 'Store LLC', 'admin_name': 'Owner Admin'}
        )

        # Login
        response = self.client.post(self.login_url, {
            'email': 'owner@store.com',
            'password': 'mysecretpassword'
        })
        self.assertRedirects(response, self.dashboard_url)
        self.assertTrue('_auth_user_id' in self.client.session)

        # Access dashboard
        dash_resp = self.client.get(self.dashboard_url)
        self.assertEqual(dash_resp.status_code, 200)
        self.assertContains(dash_resp, 'Store LLC')

        # Logout
        logout_resp = self.client.get(self.logout_url)
        self.assertRedirects(logout_resp, self.login_url)
        self.assertFalse('_auth_user_id' in self.client.session)

    def test_login_invalid_password(self):
        User.objects.create_user(
            username='user@test.com',
            email='user@test.com',
            password='correctpassword'
        )
        response = self.client.post(self.login_url, {
            'email': 'user@test.com',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, 200)

    def test_protected_pages_redirect_unauthenticated_user(self):
        protected_urls = [
            reverse('dashboard'),
            reverse('prediction'),
            reverse('customers'),
            reverse('add_customer'),
            reverse('record_purchase'),
            reverse('history'),
            reverse('analytics'),
            reverse('model_info'),
            reverse('about'),
        ]
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302, f"URL {url} did not enforce auth redirect.")
            self.assertTrue('/login/' in response.url)


class LiveCustomerAndPurchaseTests(TestCase):
    """Test BusinessCustomer creation, PurchaseTransaction recording, and RFM calculations."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='admin@shop.com',
            email='admin@shop.com',
            password='password123'
        )
        self.profile = BusinessProfile.objects.get(user=self.user)
        self.profile.business_name = 'Test Shop'
        self.profile.save()

        self.client = Client()
        self.client.login(username='admin@shop.com', password='password123')

    def test_create_business_customer(self):
        cust = BusinessCustomer.objects.create(
            business_profile=self.profile,
            name='Rajesh Sharma',
            email='rajesh@example.com'
        )
        self.assertTrue(cust.customer_id.isdigit())
        self.assertEqual(len(cust.customer_id), 5)
        self.assertEqual(cust.name, 'Rajesh Sharma')

    def test_record_multiple_purchases_and_verify_rfm(self):
        cust = BusinessCustomer.objects.create(
            business_profile=self.profile,
            name='Priya Patel',
            email='priya@example.com'
        )

        now = timezone.now()
        p1 = PurchaseTransaction.objects.create(
            customer=cust,
            stock_code='85123A',
            product_description='WHITE HANGING HEART T-LIGHT HOLDER',
            quantity=5,
            unit_price=2.55,
            purchase_date=now - timedelta(days=10)
        )
        p2 = PurchaseTransaction.objects.create(
            customer=cust,
            stock_code='22423',
            product_description='REGENCY CAKESTAND 3 TIER',
            quantity=2,
            unit_price=12.75,
            purchase_date=now - timedelta(days=2)
        )

        # Verify amounts
        self.assertEqual(p1.total_amount, 12.75)
        self.assertEqual(p2.total_amount, 25.50)

        # Run RFM & recommendation calculation
        res = recommend_for_business_customer(cust, top_n=5)

        self.assertEqual(res['frequency'], 2)
        self.assertEqual(res['monetary'], 38.25)
        self.assertEqual(res['total_quantity'], 7)
        self.assertEqual(res['unique_products'], 2)
        self.assertEqual(res['recency'], 2)
        self.assertEqual(res['avg_order_value'], 19.12)
        self.assertFalse(res['cold_start'])
        self.assertTrue(len(res['recommendations']) > 0)

        # Ensure purchased stock codes (85123A, 22423) are excluded from recommendations
        rec_stock_codes = [r['stock_code'] for r in res['recommendations']]
        self.assertNotIn('85123A', rec_stock_codes)
        self.assertNotIn('22423', rec_stock_codes)

    def test_cold_start_new_customer(self):
        cust = BusinessCustomer.objects.create(
            business_profile=self.profile,
            name='New Visitor',
            email='visitor@example.com'
        )

        res = recommend_for_business_customer(cust, top_n=5)
        self.assertTrue(res['cold_start'])
        self.assertEqual(res['frequency'], 0)
        self.assertEqual(res['monetary'], 0.0)
        self.assertEqual(len(res['recommendations']), 5)
        self.assertEqual(res['segment'], 'New Customer — No Purchase History')

    def test_form_validation_negative_quantity_and_price(self):
        cust = BusinessCustomer.objects.create(
            business_profile=self.profile,
            name='Test Validations'
        )

        # Negative quantity form test
        invalid_qty_form = RecordPurchaseForm(
            data={
                'customer': cust.id,
                'stock_code': '85123A',
                'product_description': 'Heart T-Light Holder',
                'quantity': -2,
                'unit_price': 5.00
            },
            business_profile=self.profile
        )
        self.assertFalse(invalid_qty_form.is_valid())
        self.assertIn('quantity', invalid_qty_form.errors)

        # Negative unit price form test
        invalid_price_form = RecordPurchaseForm(
            data={
                'customer': cust.id,
                'stock_code': '85123A',
                'product_description': 'Heart T-Light Holder',
                'quantity': 2,
                'unit_price': -10.00
            },
            business_profile=self.profile
        )
        self.assertFalse(invalid_price_form.is_valid())
        self.assertIn('unit_price', invalid_price_form.errors)


class MLRecommenderTests(TestCase):
    """Test recommendation engine, K-Means cluster inference, and customer dataset lookup."""

    def test_historical_customer_17850_lookup(self):
        res = recommend_for_customer(17850, top_n=5)
        self.assertNotIn('error', res)
        self.assertEqual(res['customer_id'], 17850)
        self.assertEqual(res['segment'], 'High-Value Customers')
        self.assertEqual(len(res['recommendations']), 5)

        # Check products are valid dicts with descriptions
        for prod in res['recommendations']:
            self.assertIn('stock_code', prod)
            self.assertIn('description', prod)
            self.assertIn('unit_price', prod)
            self.assertTrue(len(prod['description']) > 0)

    def test_nonexistent_customer_lookup(self):
        res = recommend_for_customer(999999, top_n=5)
        self.assertIn('error', res)

    def test_adhoc_rfm_prediction(self):
        res = predict_segment_and_recommend(recency=10, frequency=15, monetary=3000, top_n=5)
        self.assertIn('predicted_segment', res)
        self.assertEqual(res['predicted_segment'], 'High-Value Customers')
        self.assertEqual(len(res['recommendations']), 5)


class PageResponseTests(TestCase):
    """Verify all project URLs render without 404/500 template or static errors."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='admin@test.com',
            email='admin@test.com',
            password='password123'
        )
        BusinessProfile.objects.update_or_create(
            user=self.user,
            defaults={'business_name': 'Test Corp', 'admin_name': 'Admin User'}
        )
        self.client = Client()
        self.client.login(username='admin@test.com', password='password123')

        # Create a sample customer & purchase for detail page testing
        self.cust = BusinessCustomer.objects.create(
            business_profile=self.user.business_profile,
            name='Sam Sample'
        )
        PurchaseTransaction.objects.create(
            customer=self.cust,
            stock_code='85123A',
            product_description='Sample Item',
            quantity=1,
            unit_price=5.00
        )

    def test_all_pages_load_200(self):
        urls = [
            reverse('dashboard'),
            reverse('prediction'),
            reverse('customers'),
            reverse('add_customer'),
            reverse('record_purchase'),
            reverse('history'),
            reverse('analytics'),
            reverse('model_info'),
            reverse('about'),
            reverse('customer_detail', kwargs={'customer_id': self.cust.customer_id}),
            reverse('customer_detail', kwargs={'customer_id': 17850}),
        ]
        for url in urls:
            resp = self.client.get(url)
            self.assertEqual(resp.status_code, 200, f"Page {url} failed with status {resp.status_code}")

    def test_api_product_search(self):
        url = reverse('api_products') + '?q=heart'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('products', data)
        self.assertTrue(len(data['products']) > 0)

    def test_prediction_page_lookup_customer_id_1_and_new_ids(self):
        # Test lookup for Customer ID 1 (matches live BusinessCustomer id=1)
        resp1 = self.client.post(reverse('prediction'), {'form_type': 'lookup', 'customer_id': '1', 'top_n': 5})
        self.assertEqual(resp1.status_code, 200)
        self.assertContains(resp1, 'Recommended Products')
        self.assertContains(resp1, self.cust.customer_id)

        # Test lookup for completely new / unrecorded Customer ID 99999
        resp2 = self.client.post(reverse('prediction'), {'form_type': 'lookup', 'customer_id': '99999', 'top_n': 5})
        self.assertEqual(resp2.status_code, 200)
        self.assertContains(resp2, 'New Customer')
        self.assertContains(resp2, 'Recommended Products')
