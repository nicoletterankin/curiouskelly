const express = require('express');
const manifestService = require('../../services/reinmaker/manifest.service');

const router = express.Router();

router.get('/', async (req, res, next) => {
  try {
    if (!manifestService.isFeatureEnabled()) {
      return res.status(404).json({ status: 'not_found' });
    }

    const manifest = await manifestService.getManifest();
    res.json({ status: 'ok', manifest, generatedAt: manifest.generatedAt });
  } catch (error) {
    next(error);
  }
});

module.exports = router;












