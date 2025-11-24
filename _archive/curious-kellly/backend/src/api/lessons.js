const express = require('express');
const router = express.Router();
const { query } = require('../services/database');
const fs = require('fs').promises;
const path = require('path');

/**
 * GET /api/lessons/calendar
 * Get full 365-day calendar
 */
router.get('/calendar', async (req, res) => {
  try {
    const calendarPath = path.join(__dirname, '../../../../lessons/365_day_calendar.json');
    const calendarData = await fs.readFile(calendarPath, 'utf8');
    const calendar = JSON.parse(calendarData);

    res.json(calendar);
  } catch (error) {
    console.error('Calendar fetch error:', error);
    res.status(500).json({ error: 'Failed to load calendar' });
  }
});

/**
 * GET /api/lessons/day/:day
 * Get lesson by day number (1-365)
 */
router.get('/day/:day', async (req, res) => {
  try {
    const { day } = req.params;
    const dayNum = parseInt(day);

    if (isNaN(dayNum) || dayNum < 1 || dayNum > 365) {
      return res.status(400).json({ error: 'Invalid day number (must be 1-365)' });
    }

    // Load calendar to find lesson ID
    const calendarPath = path.join(__dirname, '../../../../lessons/365_day_calendar.json');
    const calendarData = await fs.readFile(calendarPath, 'utf8');
    const calendar = JSON.parse(calendarData);

    const lessonInfo = calendar.lessons[dayNum - 1];

    if (!lessonInfo) {
      return res.status(404).json({ error: 'Lesson not found' });
    }

    // If has DNA file, try to load full DNA
    if (lessonInfo.has_dna && lessonInfo.dna_file) {
      try {
        const dnaPath = path.join(__dirname, `../../../../lessons/${lessonInfo.dna_file}-dna.json`);
        const dnaData = await fs.readFile(dnaPath, 'utf8');
        const dna = JSON.parse(dnaData);

        res.json({
          ...lessonInfo,
          fullDNA: dna
        });
        return;
      } catch (dnaError) {
        console.warn(`DNA file not found for ${lessonInfo.dna_file}, returning basic info`);
      }
    }

    // Return basic lesson info
    res.json(lessonInfo);
  } catch (error) {
    console.error('Lesson fetch error:', error);
    res.status(500).json({ error: 'Failed to load lesson' });
  }
});

/**
 * POST /api/lessons/complete
 * Mark lesson as completed for user
 */
router.post('/complete', async (req, res) => {
  try {
    const { userId, lessonDay, lessonId, durationSeconds, ageVariant } = req.body;

    // Validate required fields
    if (!userId || !lessonDay || !lessonId) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Check if already completed today
    const existing = await query(
      `SELECT * FROM lesson_completions 
       WHERE user_id = $1 AND lesson_day = $2 AND DATE(completed_at) = CURRENT_DATE`,
      [userId, lessonDay]
    );

    if (existing.rows.length > 0) {
      return res.status(400).json({ error: 'Lesson already completed today' });
    }

    // Record completion
    await query(
      `INSERT INTO lesson_completions 
       (user_id, lesson_day, lesson_id, duration_seconds, age_variant) 
       VALUES ($1, $2, $3, $4, $5)`,
      [userId, lessonDay, lessonId, durationSeconds || null, ageVariant || null]
    );

    // Update user stats
    await query(
      `UPDATE users 
       SET lessons_completed = lessons_completed + 1,
           last_lesson_at = NOW()
       WHERE id = $1`,
      [userId]
    );

    // Calculate and update streak
    const streakResult = await updateUserStreak(userId);

    res.json({
      success: true,
      lessonsCompleted: streakResult.lessonsCompleted,
      currentStreak: streakResult.currentStreak,
      longestStreak: streakResult.longestStreak
    });
  } catch (error) {
    console.error('Lesson completion error:', error);
    res.status(500).json({ error: 'Failed to record lesson completion' });
  }
});

/**
 * GET /api/lessons/user/:userId/progress
 * Get user's lesson progress
 */
router.get('/user/:userId/progress', async (req, res) => {
  try {
    const { userId } = req.params;

    // Get user stats
    const userResult = await query(
      'SELECT * FROM users WHERE id = $1',
      [userId]
    );

    if (userResult.rows.length === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const user = userResult.rows[0];

    // Get completed lessons
    const completionsResult = await query(
      `SELECT lesson_day, lesson_id, completed_at, duration_seconds 
       FROM lesson_completions 
       WHERE user_id = $1 
       ORDER BY completed_at DESC`,
      [userId]
    );

    res.json({
      lessonsCompleted: user.lessons_completed,
      currentStreak: user.current_streak,
      longestStreak: user.longest_streak,
      lastLessonAt: user.last_lesson_at,
      completedLessons: completionsResult.rows
    });
  } catch (error) {
    console.error('Progress fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch progress' });
  }
});

/**
 * Helper function to update user streak
 */
async function updateUserStreak(userId) {
  // Get last lesson date
  const lastLesson = await query(
    `SELECT MAX(completed_at) as last_completed 
     FROM lesson_completions 
     WHERE user_id = $1`,
    [userId]
  );

  const lastCompleted = lastLesson.rows[0].last_completed;
  
  if (!lastCompleted) {
    // First lesson
    await query(
      'UPDATE users SET current_streak = 1, longest_streak = 1 WHERE id = $1',
      [userId]
    );
    return { currentStreak: 1, longestStreak: 1, lessonsCompleted: 1 };
  }

  // Check if completed yesterday (streak continues)
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const lastCompletedDate = new Date(lastCompleted);
  
  const isConsecutive = (
    lastCompletedDate.toDateString() === yesterday.toDateString() ||
    lastCompletedDate.toDateString() === new Date().toDateString()
  );

  if (isConsecutive) {
    // Increment streak
    const result = await query(
      `UPDATE users 
       SET current_streak = current_streak + 1,
           longest_streak = GREATEST(longest_streak, current_streak + 1)
       WHERE id = $1 
       RETURNING current_streak, longest_streak, lessons_completed`,
      [userId]
    );
    return result.rows[0];
  } else {
    // Streak broken, reset to 1
    const result = await query(
      `UPDATE users 
       SET current_streak = 1 
       WHERE id = $1 
       RETURNING current_streak, longest_streak, lessons_completed`,
      [userId]
    );
    return result.rows[0];
  }
}

module.exports = router;
