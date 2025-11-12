import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.API_BASE_URL || "http://localhost:4000";

export const options = {
  stages: [
    { duration: "30s", target: 100 },
    { duration: "1m", target: 100 },
    { duration: "30s", target: 0 },
  ],
};

export default function () {
  const res = http.get(`${baseUrl}/health`);
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
  sleep(1);
}

