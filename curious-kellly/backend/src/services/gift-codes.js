const { v4: uuidv4 } = require('uuid');

/**
 * Generate unique gift code
 * Format: CK-XXXXX-XXXXX (e.g., CK-A7B2C-9D4E1)
 */
function generateGiftCode() {
  const part1 = generateRandomString(5);
  const part2 = generateRandomString(5);
  return `CK-${part1}-${part2}`.toUpperCase();
}

/**
 * Generate random alphanumeric string
 */
function generateRandomString(length) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

/**
 * Validate gift code format
 */
function isValidGiftCodeFormat(code) {
  const pattern = /^CK-[A-Z0-9]{5}-[A-Z0-9]{5}$/;
  return pattern.test(code);
}

/**
 * Generate UUID for internal use
 */
function generateUUID() {
  return uuidv4();
}

module.exports = {
  generateGiftCode,
  isValidGiftCodeFormat,
  generateUUID
};






