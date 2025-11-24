const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const validator = require('./validator');

const REINMAKER_ROOT = path.join(__dirname, '../../../config/reinmaker');
const TRIBES_DIR = path.join(REINMAKER_ROOT, 'tribes');
const QUESTS_DIR = path.join(REINMAKER_ROOT, 'quests');
const LOCALES_DIR = path.join(REINMAKER_ROOT, 'locales');
const MANIFEST_PATH = path.join(REINMAKER_ROOT, 'manifest.json');

const FEATURED_ROTATION = ['Light', 'Stone', 'Metal', 'Code', 'Air', 'Water', 'Fire'];

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(raw);
}

function checksum(value) {
  return crypto.createHash('sha1').update(value).digest('hex');
}

function uniquePush(set, value) {
  if (value) {
    set.add(value);
  }
}

async function collectTribes() {
  const files = await fs.readdir(TRIBES_DIR);
  const tribes = [];

  await validator.loadSchemas();

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const filePath = path.join(TRIBES_DIR, file);
    const data = await readJson(filePath);
    const { valid, errors } = validator.validate('tribePack.schema.json', data);
    if (!valid) {
      throw new Error(
        `Reinmaker tribe pack validation failed for ${file}: ${JSON.stringify(errors, null, 2)}`
      );
    }

    tribes.push({
      id: data.id,
      tribe: data.tribe,
      lensId: data.lensId,
      color: data.color,
      icon: data.icon,
      featuredQuoteKey: data.featuredQuoteKey,
      ageDifficulty: data.ageDifficulty,
      tiers: data.tiers,
      finaleStoneId: data.finaleStoneId,
      sourcePath: path.relative(REINMAKER_ROOT, filePath)
    });
  }

  tribes.sort((a, b) => a.tribe.localeCompare(b.tribe));
  return tribes;
}

async function collectQuests() {
  const files = await fs.readdir(QUESTS_DIR);
  const quests = [];

  await validator.loadSchemas();

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const filePath = path.join(QUESTS_DIR, file);
    const data = await readJson(filePath);
    await validator.loadSchemas();
    const { valid, errors } = validator.validate('quest.schema.json', data);
    if (!valid) {
      throw new Error(
        `Reinmaker quest validation failed for ${file}: ${JSON.stringify(errors, null, 2)}`
      );
    }

    const summary = {
      id: data.id,
      tribe: data.tribe,
      tier: data.tier,
      kind: data.kind,
      ageBuckets: data.ageBuckets,
      lessonRef: data.lessonRef || null,
      estimatedDurationMin: data.estimatedDurationMin || null,
      localizationKey: data.localizationKey || null,
      rewards: data.rewards,
      captions: data.captions || {},
      sourcePath: path.relative(REINMAKER_ROOT, filePath)
    };

    if (data.audio) {
      summary.audio = data.audio;
    }

    quests.push(summary);
  }

  quests.sort((a, b) => a.id.localeCompare(b.id));
  return quests;
}

async function collectLocales() {
  const files = await fs.readdir(LOCALES_DIR);
  const locales = {};

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const locale = path.basename(file, '.json');
    const filePath = path.join(LOCALES_DIR, file);
    const raw = await fs.readFile(filePath, 'utf-8');
    const strings = JSON.parse(raw);
    locales[locale] = {
      path: path.relative(REINMAKER_ROOT, filePath),
      hash: checksum(raw),
      keys: Object.keys(strings)
    };
  }

  return locales;
}

function collectAssets({ tribes, quests }) {
  const icons = new Set();
  const captions = new Set();
  const audio = new Set();

  tribes.forEach((tribe) => {
    uniquePush(icons, tribe.icon);
  });

  quests.forEach((quest) => {
    if (quest.captions) {
      Object.values(quest.captions).forEach((captionPath) => uniquePush(captions, captionPath));
    }

    if (quest.audio) {
      Object.values(quest.audio).forEach((ageMap) => {
        Object.values(ageMap).forEach((audioPath) => uniquePush(audio, audioPath));
      });
    }
  });

  return {
    icons: Array.from(icons).sort(),
    captions: Array.from(captions).sort(),
    audio: Array.from(audio).sort()
  };
}

async function buildManifest({ writeFile = true } = {}) {
  await validator.loadSchemas();

  const tribes = await collectTribes();
  const quests = await collectQuests();
  const locales = await collectLocales();

  const assets = collectAssets({ tribes, quests });

  const contentSignature = checksum(
    JSON.stringify({ tribes, quests, locales, assets, featuredRotation: FEATURED_ROTATION })
  );

  const manifest = {
    version: 'rmk.manifest.v1',
    generatedAt: new Date().toISOString(),
    featuredRotation: FEATURED_ROTATION,
    locales,
    tribes,
    quests,
    assets,
    contentHash: contentSignature
  };

  if (writeFile) {
    await fs.writeFile(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
    console.log(`Reinmaker manifest written to ${MANIFEST_PATH}`);
  }

  return manifest;
}

if (require.main === module) {
  buildManifest()
    .then(() => {
      console.log('Reinmaker manifest build complete.');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Failed to build Reinmaker manifest:', error);
      process.exit(1);
    });
}

module.exports = buildManifest;

