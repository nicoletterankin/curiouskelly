const fs = require('fs/promises');
const { watch } = require('fs');
const path = require('path');
const EventEmitter = require('events');
const buildManifest = require('./buildManifest');
const validator = require('./validator');

class ReinmakerManifestService extends EventEmitter {
  constructor() {
    super();

    this.rootDir = path.join(__dirname, '../../../config/reinmaker');
    this.questsDir = path.join(this.rootDir, 'quests');
    this.manifestPath = path.join(this.rootDir, 'manifest.json');

    this.manifestCache = null;
    this.cacheTimestamp = null;
    this.watchHandlers = [];
    this.refreshTimer = null;

    this.readyPromise = this.initialize();
  }

  isFeatureEnabled() {
    return process.env.FEATURES_REINMAKER === '1';
  }

  async initialize() {
    try {
      await validator.loadSchemas();
      await this.refreshCache({ rebuildIfMissing: true });
      if (process.env.NODE_ENV === 'development') {
        this.setupWatchers();
      }
    } catch (error) {
      console.error('[ReinmakerManifestService] Initialization failed:', error);
      throw error;
    }
  }

  async waitUntilReady() {
    return this.readyPromise;
  }

  teardownWatchers() {
    this.watchHandlers.forEach((close) => close());
    this.watchHandlers = [];
  }

  scheduleRefresh(reason = 'file-change') {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    this.refreshTimer = setTimeout(async () => {
      try {
        await this.refreshCache({ rebuildIfMissing: true });
        console.log(`[ReinmakerManifestService] Manifest refreshed due to ${reason}.`);
      } catch (error) {
        console.error('[ReinmakerManifestService] Failed to refresh manifest:', error);
      }
    }, 200);
  }

  setupWatchers() {
    const targets = ['tribes', 'quests', 'locales'];

    targets.forEach((folder) => {
      const dirPath = path.join(this.rootDir, folder);
      try {
        const watcher = watch(dirPath, { persistent: false }, (eventType, filename) => {
          if (filename && filename.endsWith('.json')) {
            console.log(
              `[ReinmakerManifestService] Detected ${eventType} for ${folder}/${filename}; scheduling refresh.`
            );
            this.scheduleRefresh(`${folder}:${filename}`);
          }
        });

        this.watchHandlers.push(() => watcher.close());
      } catch (error) {
        console.warn(
          `[ReinmakerManifestService] Unable to watch directory ${dirPath}: ${error.message}`
        );
      }
    });
  }

  async refreshCache({ rebuildIfMissing = false } = {}) {
    let manifest;

    try {
      const raw = await fs.readFile(this.manifestPath, 'utf-8');
      manifest = JSON.parse(raw);
    } catch (error) {
      if (error.code === 'ENOENT' || rebuildIfMissing) {
        manifest = await buildManifest({ writeFile: true });
      } else {
        throw error;
      }
    }

    this.manifestCache = manifest;
    this.cacheTimestamp = new Date();
    this.emit('manifest:updated', manifest);
    return manifest;
  }

  async getManifest() {
    await this.waitUntilReady();

    if (!this.manifestCache) {
      await this.refreshCache({ rebuildIfMissing: true });
    }

    return this.manifestCache;
  }

  async getQuestById(questId) {
    await this.waitUntilReady();

    const questFile = path.join(this.questsDir, `${questId}.json`);

    try {
      const raw = await fs.readFile(questFile, 'utf-8');
      const quest = JSON.parse(raw);
      const { valid, errors } = validator.validate('quest.schema.json', quest);
      if (!valid) {
        throw new Error(
          `Quest ${questId} failed schema validation: ${JSON.stringify(errors, null, 2)}`
        );
      }
      return quest;
    } catch (error) {
      if (error.code === 'ENOENT') {
        const notFound = new Error(`Quest ${questId} not found.`);
        notFound.statusCode = 404;
        throw notFound;
      }
      throw error;
    }
  }
}

const manifestService = new ReinmakerManifestService();

module.exports = manifestService;












