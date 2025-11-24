/**
 * Reinmaker Validator Service
 * Uses AJV to load and validate all Reinmaker-related schemas.
 */

const Ajv = require('ajv');
const fs = require('fs').promises;
const path = require('path');

class ReinmakerValidator {
  constructor() {
    this.ajv = new Ajv({ allErrors: true });
    this.schemasDir = path.join(__dirname, '../../../config/reinmaker/schemas');
    this.isLoaded = false;
    this.loadPromise = null;
  }

  /**
   * Loads all schemas from the schemas directory and compiles them.
   */
  async loadSchemas() {
    if (this.isLoaded) {
      return;
    }

    if (this.loadPromise) {
      return this.loadPromise;
    }

    this.loadPromise = (async () => {
      try {
        const schemaFiles = await fs.readdir(this.schemasDir);
        const lessonSchemaPath = path.join(__dirname, '../../../config/lesson-dna-schema.json');

        // Add the existing lesson DNA schema to allow $ref
        const lessonSchemaContent = await fs.readFile(lessonSchemaPath, 'utf-8');
        const lessonSchema = JSON.parse(lessonSchemaContent);
        this.ajv.addSchema(lessonSchema, 'lesson-dna-schema.json');

        for (const file of schemaFiles) {
          if (file.endsWith('.schema.json')) {
            const schemaPath = path.join(this.schemasDir, file);
            const schemaContent = await fs.readFile(schemaPath, 'utf-8');
            const schema = JSON.parse(schemaContent);
            this.ajv.addSchema(schema, file);
          }
        }

        this.isLoaded = true;
        console.log('Reinmaker schemas loaded and compiled successfully.');
      } catch (error) {
        console.error('Failed to load Reinmaker schemas:', error);
        throw new Error('Could not initialize Reinmaker validator.');
      }
    })();

    return this.loadPromise;
  }

  /**
   * Validates a data object against a specific schema.
   * @param {string} schemaName - The file name of the schema to use for validation (e.g., 'tribePack.schema.json').
   * @param {object} data - The data object to validate.
   * @returns {{valid: boolean, errors: object[]|null}} - The validation result.
   */
  validate(schemaName, data) {
    if (!this.isLoaded) {
      throw new Error('Reinmaker schemas not loaded yet. Call loadSchemas() first.');
    }
    const validate = this.ajv.getSchema(schemaName);
    if (!validate) {
      throw new Error(`Schema ${schemaName} not found.`);
    }

    const valid = validate(data);
    return {
      valid,
      errors: valid ? null : validate.errors,
    };
  }

  /**
   * A middleware for validating request bodies for a given schema
   */
  validateMiddleware(schemaName) {
    return (req, res, next) => {
        const { valid, errors } = this.validate(schemaName, req.body);
        if(!valid) {
            return res.status(400).json({
                status: 'error',
                message: 'Schema validation failed',
                errors: errors.map(err => ({
                    path: err.instancePath,
                    message: err.message,
                    params: err.params
                }))
            });
        }
        next();
    }
  }
}

const validator = new ReinmakerValidator();
// Initialize schemas on module load.
validator.loadSchemas().catch((error) => {
  console.error('[ReinmakerValidator] Initial schema load failed:', error.message);
});

module.exports = validator;

if (require.main === module) {
  (async () => {
    try {
      await validator.loadSchemas();

      const rootDir = path.join(__dirname, '../../../config/reinmaker');
      const tribesDir = path.join(rootDir, 'tribes');
      const questsDir = path.join(rootDir, 'quests');

      const tribeFiles = await fs.readdir(tribesDir);
      for (const file of tribeFiles) {
        if (!file.endsWith('.json')) continue;
        const filePath = path.join(tribesDir, file);
        const raw = await fs.readFile(filePath, 'utf-8');
        const json = JSON.parse(raw);
        const { valid, errors } = validator.validate('tribePack.schema.json', json);
        if (!valid) {
          throw new Error(`Tribe validation failed for ${file}: ${JSON.stringify(errors, null, 2)}`);
        }
      }

      const questFiles = await fs.readdir(questsDir);
      for (const file of questFiles) {
        if (!file.endsWith('.json')) continue;
        const filePath = path.join(questsDir, file);
        const raw = await fs.readFile(filePath, 'utf-8');
        const json = JSON.parse(raw);
        const { valid, errors } = validator.validate('quest.schema.json', json);
        if (!valid) {
          throw new Error(`Quest validation failed for ${file}: ${JSON.stringify(errors, null, 2)}`);
        }
      }

      console.log('✅ Reinmaker content validation succeeded.');
      process.exit(0);
    } catch (error) {
      console.error('❌ Reinmaker content validation failed:', error);
      process.exit(1);
    }
  })();
}
