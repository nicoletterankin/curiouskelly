import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    globals: true,
    setupFiles: [],
    alias: {
      "@acme/config": path.resolve(__dirname, "../packages/config/src"),
      "@acme/logger": path.resolve(__dirname, "../packages/logger/src"),
      "@acme/database": path.resolve(__dirname, "../packages/database/src"),
      "@acme/service-auth": path.resolve(__dirname, "../services/auth/src"),
      "@acme/service-entitlements": path.resolve(__dirname, "../services/entitlements/src"),
      "@acme/service-lessons": path.resolve(__dirname, "../services/lessons/src"),
      "@acme/service-schedule": path.resolve(__dirname, "../services/schedule/src"),
      "@acme/service-classroom": path.resolve(__dirname, "../services/classroom/src"),
      "@acme/service-payments": path.resolve(__dirname, "../services/payments/src"),
      "@acme/service-notifications": path.resolve(__dirname, "../services/notifications/src"),
      "@acme/service-search": path.resolve(__dirname, "../services/search/src"),
      "@acme/service-telemetry": path.resolve(__dirname, "../services/telemetry/src"),
      "@acme/service-ops": path.resolve(__dirname, "../services/ops/src"),
      "@acme/types": path.resolve(__dirname, "../packages/types/src")
    },
    coverage: {
      reporter: ["text", "lcov"],
      exclude: ["**/node_modules/**", "**/dist/**"]
    }
  }
});

