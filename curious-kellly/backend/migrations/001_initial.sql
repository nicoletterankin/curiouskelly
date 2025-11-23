-- Curious Kelly Database Schema
-- PostgreSQL migration

-- Create gifts table
CREATE TABLE IF NOT EXISTS gifts (
  id SERIAL PRIMARY KEY,
  code VARCHAR(50) UNIQUE NOT NULL,
  gifter_email VARCHAR(255) NOT NULL,
  gifter_name VARCHAR(255),
  recipient_email VARCHAR(255) NOT NULL,
  gift_message TEXT,
  purchase_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  delivery_date TIMESTAMP NOT NULL,
  redeemed BOOLEAN DEFAULT FALSE,
  redeemed_at TIMESTAMP,
  redeemed_by_user_id INTEGER,
  stripe_session_id VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  age INTEGER,
  plan VARCHAR(50) NOT NULL, -- 'personal', 'family', 'gift'
  subscription_status VARCHAR(50) DEFAULT 'active', -- 'active', 'canceled', 'past_due'
  stripe_customer_id VARCHAR(255),
  stripe_subscription_id VARCHAR(255),
  gift_code_used VARCHAR(50),
  current_streak INTEGER DEFAULT 0,
  longest_streak INTEGER DEFAULT 0,
  lessons_completed INTEGER DEFAULT 0,
  last_lesson_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create lesson_completions table
CREATE TABLE IF NOT EXISTS lesson_completions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  lesson_day INTEGER NOT NULL,
  lesson_id VARCHAR(255) NOT NULL,
  completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  duration_seconds INTEGER,
  age_variant VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_gifts_code ON gifts(code);
CREATE INDEX IF NOT EXISTS idx_gifts_recipient ON gifts(recipient_email);
CREATE INDEX IF NOT EXISTS idx_gifts_redeemed ON gifts(redeemed);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_stripe_customer ON users(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_completions_user ON lesson_completions(user_id);
CREATE INDEX IF NOT EXISTS idx_completions_day ON lesson_completions(lesson_day);
CREATE INDEX IF NOT EXISTS idx_completions_date ON lesson_completions(completed_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add updated_at triggers
CREATE TRIGGER update_gifts_updated_at BEFORE UPDATE ON gifts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert test data (for development only)
-- Uncomment below for local testing

/*
INSERT INTO users (email, name, age, plan) VALUES 
  ('test@example.com', 'Test User', 25, 'personal'),
  ('family@example.com', 'Family User', 40, 'family');

INSERT INTO gifts (code, gifter_email, gifter_name, recipient_email, gift_message, delivery_date) VALUES
  ('CK-TEST1-TEST1', 'gifter@example.com', 'John Doe', 'recipient@example.com', 'Merry Christmas!', '2025-12-25 06:00:00');
*/

-- Success message
SELECT 'Database schema created successfully!' AS status;






