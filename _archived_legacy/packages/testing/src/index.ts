import { GenericContainer, StartedTestContainer, Wait } from "testcontainers";

interface TestEnvironment {
  postgres: StartedTestContainer;
  redis: StartedTestContainer;
  meilisearch: StartedTestContainer;
}

const startPostgres = () =>
  new GenericContainer("postgres:16")
    .withEnvironment({
      POSTGRES_DB: "app",
      POSTGRES_USER: "postgres",
      POSTGRES_PASSWORD: "postgres",
    })
    .withExposedPorts(5432)
    .withWaitStrategy(Wait.forLogMessage("database system is ready to accept connections"))
    .start();

const startRedis = () =>
  new GenericContainer("redis:7")
    .withExposedPorts(6379)
    .withWaitStrategy(Wait.forLogMessage("Ready to accept connections"))
    .start();

const startMeilisearch = () =>
  new GenericContainer("getmeili/meilisearch:v1.8")
    .withEnvironment({
      MEILI_NO_ANALYTICS: "true",
      MEILI_ENV: "development",
    })
    .withExposedPorts(7700)
    .withWaitStrategy(Wait.forHttp("/health").forPort(7700))
    .start();

export const startTestEnvironment = async (): Promise<TestEnvironment> => {
  const [postgres, redis, meilisearch] = await Promise.all([startPostgres(), startRedis(), startMeilisearch()]);
  return { postgres, redis, meilisearch };
};

export const stopTestEnvironment = async ({ postgres, redis, meilisearch }: TestEnvironment) => {
  await Promise.all([postgres.stop(), redis.stop(), meilisearch.stop()]);
};

export type { TestEnvironment };

