import { test, expect } from "@playwright/test";

test.describe("Join flow health check", () => {
  test.skip(!process.env.API_BASE_URL, "API_BASE_URL is required for live join test");

  test("health endpoint responds", async ({ request }) => {
    const response = await request.get("/health");
    expect(response.ok()).toBeTruthy();
    const payload = await response.json();
    expect(payload).toHaveProperty("status", "ok");
  });
});

