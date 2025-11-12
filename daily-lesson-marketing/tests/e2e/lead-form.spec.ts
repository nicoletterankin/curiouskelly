import { test, expect } from '@playwright/test';

test.describe('Lead Form', () => {
  test('should submit form successfully', async ({ page }) => {
    await page.goto('/');
    await page.fill('#first_name', 'John');
    await page.fill('#last_name', 'Doe');
    await page.fill('#email', 'john.doe@example.com');
    await page.fill('#phone', '+1234567890');
    await page.selectOption('#country', 'US');
    
    // Wait for Turnstile to load (or mock it)
    await page.waitForTimeout(2000);
    
    await page.click('#submit-btn');
    
    // Should redirect to thank-you page
    await expect(page).toHaveURL(/\/thank-you/);
  });

  test('should show validation errors', async ({ page }) => {
    await page.goto('/');
    await page.click('#submit-btn');
    
    const error = await page.locator('.error-message').first();
    await expect(error).toBeVisible();
  });
});

test.describe('Consent Manager', () => {
  test('should block marketing tags until consent', async ({ page }) => {
    await page.goto('/');
    
    // Check that GTM is not loaded
    const gtmScript = await page.locator('script[src*="googletagmanager"]');
    await expect(gtmScript).toHaveCount(0);
    
    // Accept consent
    await page.click('#consent-accept-all');
    
    // Wait for scripts to load
    await page.waitForTimeout(1000);
    
    // Check that GTM is now loaded (if configured)
    // This will vary based on env vars
  });
});

test.describe('Language Switching', () => {
  test('should switch to Spanish', async ({ page }) => {
    await page.goto('/');
    await page.click('#lang-toggle');
    await page.click('a[href="/es-es/"]');
    
    await expect(page).toHaveURL(/\/es-es/);
    const headline = await page.locator('h1');
    await expect(headline).toContainText('Domina el Inglés');
  });
});

test.describe('Accessibility', () => {
  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/');
    
    const form = await page.locator('#lead-form');
    await expect(form).toBeVisible();
    
    const countdown = await page.locator('.hero-countdown [role="timer"]');
    await expect(countdown).toHaveAttribute('aria-live', 'polite');
  });

  test('should support keyboard navigation', async ({ page }) => {
    await page.goto('/');
    
    await page.keyboard.press('Tab');
    const focused = await page.evaluate(() => document.activeElement?.tagName);
    expect(focused).toBe('A');
  });
});










