import { test, expect } from '@playwright/test';

test.describe('Curious Kelly marketing site', () => {
  test('submits lead form and navigates to thank-you page', async ({ page }) => {
    await page.route('**/api/lead', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', requestId: 'test' })
      });
    });

    await page.goto('/');

    const consentBanner = page.locator('[data-consent-banner]');
    if (await consentBanner.isVisible()) {
      await page.getByRole('button', { name: /accept/i }).click();
    }

    await page.getByLabel(/First name/i).fill('Test');
    await page.getByLabel(/Last name/i).fill('User');
    await page.getByLabel(/Email/i).fill('test.user@example.com');
    await page.getByLabel(/Mobile number/i).fill('+1 202 555 0101');
    await page.getByLabel(/Country/i).selectOption('US');
    await page.getByLabel(/State/i).selectOption('CA');

    await page.getByRole('button', { name: /Submit interest/i }).click();

    await expect(page).toHaveURL(/thank-you/);
    await expect(page.getByText(/You’re officially on Kelly’s radar/i)).toBeVisible();
  });

  test('shows locale switcher banner and changes locale', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'languages', {
        get: () => ['es-ES']
      });
    });

    await page.goto('/');

    const banner = page.locator('[data-locale-banner]');
    await banner.waitFor({ state: 'visible' });
    await page.getByRole('button', { name: /Switch language/i }).click();
    await expect(page).toHaveURL(/\/es-es\//);
  });
});

