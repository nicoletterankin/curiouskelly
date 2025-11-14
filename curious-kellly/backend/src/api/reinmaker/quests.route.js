const express = require('express');
const manifestService = require('../../services/reinmaker/manifest.service');

const router = express.Router();

router.get('/:id', async (req, res, next) => {
  try {
    if (!manifestService.isFeatureEnabled()) {
      return res.status(404).json({ status: 'not_found' });
    }

    const questId = req.params.id;
    const quest = await manifestService.getQuestById(questId);

    res.json({ status: 'ok', quest, updatedAt: quest.updatedAt || null });
  } catch (error) {
    if (error.statusCode === 404) {
      return res.status(404).json({ status: 'not_found', message: error.message });
    }
    next(error);
  }
});

module.exports = router;













