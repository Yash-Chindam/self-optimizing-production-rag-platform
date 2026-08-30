import { expect, test } from '@playwright/test';

test('user receives an evidence-backed answer', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Answers that show their evidence.' })).toBeVisible();
  await page.getByLabel('Ask the ACME employee handbook').fill('How do I request vacation?');
  await page.getByRole('button', { name: 'Find evidence' }).click();

  await expect(page.locator('#status')).toHaveText('answered');
  await expect(page.locator('#answer')).toContainText('five working days');
  await expect(page.locator('#citations a')).toHaveText('[C1] ACME Employee Handbook');
});

test('user sees a clear abstention when no evidence exists', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Ask the ACME employee handbook').fill('When is the next lunar eclipse?');
  await page.getByRole('button', { name: 'Find evidence' }).click();

  await expect(page.locator('#status')).toHaveText('insufficient evidence');
  await expect(page.locator('#answer')).toContainText('could not find authorized evidence');
  await expect(page.locator('#citations a')).toHaveCount(0);
});
