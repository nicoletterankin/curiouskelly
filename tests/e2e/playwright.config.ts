import { defineConfig, devices } from "@playwright/test";

const apiBaseUrl = process.env.API_BASE_URL ?? "http://localhost:4000";

export default defineConfig({
  testDir: "./",
  timeout: 60_000,
  retries: 1,
  reporter: [["list"]],
  use: {
    baseURL: apiBaseUrl,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});

