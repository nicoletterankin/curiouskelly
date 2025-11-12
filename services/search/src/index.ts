import { MeiliSearch } from "meilisearch";
import { env } from "@acme/config";
import { logger } from "@acme/logger";

const client = new MeiliSearch({
  host: env.MEILISEARCH_HOST,
  apiKey: env.MEILISEARCH_API_KEY,
});

export class SearchService {
  private readonly index = client.index("lessons");

  async search(query: string) {
    const result = await this.index.search(query, {
      limit: 10,
      attributesToHighlight: ["title", "summary"],
    });
    return result.hits;
  }

  async upsert(documents: Record<string, unknown>[]) {
    const task = await this.index.addDocuments(documents);
    logger.info({ taskId: task.taskUid }, "SearchService.upsert");
  }
}

export const searchService = new SearchService();

