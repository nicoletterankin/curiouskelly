const express = require('express');
const router = express.Router();
const { query } = require('../services/database');

/**
 * POST /api/users/create
 * Create new user account
 */
router.post('/create', async (req, res) => {
  try {
    const { email, name, age, plan } = req.body;

    // Validate required fields
    if (!email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    // Check if user already exists
    const existingUser = await query(
      'SELECT * FROM users WHERE email = $1',
      [email]
    );

    if (existingUser.rows.length > 0) {
      return res.status(400).json({ error: 'User already exists' });
    }

    // Create user
    const result = await query(
      `INSERT INTO users 
       (email, name, age, plan, subscription_status) 
       VALUES ($1, $2, $3, $4, 'active') 
       RETURNING *`,
      [email, name || null, age || null, plan || 'personal']
    );

    const user = result.rows[0];

    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        age: user.age,
        plan: user.plan
      }
    });
  } catch (error) {
    console.error('User creation error:', error);
    res.status(500).json({ error: 'Failed to create user' });
  }
});

/**
 * GET /api/users/:id
 * Get user by ID
 */
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const result = await query(
      'SELECT * FROM users WHERE id = $1',
      [id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const user = result.rows[0];

    res.json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        age: user.age,
        plan: user.plan,
        currentStreak: user.current_streak,
        longestStreak: user.longest_streak,
        lessonsCompleted: user.lessons_completed,
        createdAt: user.created_at
      }
    });
  } catch (error) {
    console.error('User fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch user' });
  }
});

/**
 * PUT /api/users/:id
 * Update user profile
 */
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { name, age } = req.body;

    const result = await query(
      'UPDATE users SET name = COALESCE($1, name), age = COALESCE($2, age) WHERE id = $3 RETURNING *',
      [name, age, id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'User not found' });
    }

    const user = result.rows[0];

    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        age: user.age
      }
    });
  } catch (error) {
    console.error('User update error:', error);
    res.status(500).json({ error: 'Failed to update user' });
  }
});

module.exports = router;






