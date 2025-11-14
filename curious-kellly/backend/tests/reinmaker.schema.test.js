const path = require('path');
const fs = require('fs/promises');
const assert = require('assert');
const validator = require('../src/services/reinmaker/validator');
const buildManifest = require('../src/services/reinmaker/buildManifest');

(async () => {
  try {
    process.env.FEATURES_REINMAKER = process.env.FEATURES_REINMAKER || '1';

    await validator.loadSchemas();

    const rootDir = path.join(__dirname, '../config/reinmaker');
    const tribesDir = path.join(rootDir, 'tribes');
    const questsDir = path.join(rootDir, 'quests');

    const tribeFiles = (await fs.readdir(tribesDir)).filter((file) => file.endsWith('.json'));
    assert(tribeFiles.length > 0, 'No tribe definition files found.');

    for (const file of tribeFiles) {
      const filePath = path.join(tribesDir, file);
      const raw = await fs.readFile(filePath, 'utf-8');
      const json = JSON.parse(raw);
      const { valid, errors } = validator.validate('tribePack.schema.json', json);
      assert.strictEqual(
        valid,
        true,
        `Tribe schema validation failed for ${file}: ${JSON.stringify(errors, null, 2)}`
      );
    }

    const questFiles = (await fs.readdir(questsDir)).filter((file) => file.endsWith('.json'));
    assert(questFiles.length > 0, 'No quest definition files found.');

    for (const file of questFiles) {
      const filePath = path.join(questsDir, file);
      const raw = await fs.readFile(filePath, 'utf-8');
      const json = JSON.parse(raw);
      const { valid, errors } = validator.validate('quest.schema.json', json);
      assert.strictEqual(
        valid,
        true,
        `Quest schema validation failed for ${file}: ${JSON.stringify(errors, null, 2)}`
      );
    }

    const manifest = await buildManifest({ writeFile: false });
    assert.ok(manifest);
    assert.strictEqual(manifest.version, 'rmk.manifest.v1');
    assert(Array.isArray(manifest.tribes), 'Manifest tribes array missing.');
    assert(Array.isArray(manifest.quests), 'Manifest quests array missing.');

    console.log('✅ Reinmaker schema validation tests passed.');
    process.exit(0);
  } catch (error) {
    console.error('❌ Reinmaker schema validation tests failed:', error);
    process.exit(1);
  }
})();













