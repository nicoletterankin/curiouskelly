import { describe, it, expect } from 'vitest';
import { validateEmail, validatePhone, validateName, validateLeadForm, sanitizeFormData } from '../../src/lib/validation';
import { calculateCountdown } from '../../src/lib/countdown';

describe('validation', () => {
  it('should validate email correctly', () => {
    expect(validateEmail('test@example.com')).toBe(true);
    expect(validateEmail('invalid')).toBe(false);
    expect(validateEmail('')).toBe(false);
  });

  it('should validate phone correctly', () => {
    expect(validatePhone('+1234567890')).toBe(true);
    expect(validatePhone('1234567890')).toBe(true);
    expect(validatePhone('invalid')).toBe(false);
  });

  it('should validate name correctly', () => {
    expect(validateName('John')).toBe(true);
    expect(validateName('A')).toBe(false);
    expect(validateName('')).toBe(false);
  });

  it('should validate full form', () => {
    const t = {
      form: {
        errors: {
          required: 'Required',
          email: 'Invalid email',
          phone: 'Invalid phone',
          minLength: 'Too short',
          maxLength: 'Too long'
        }
      }
    };

    const valid = validateLeadForm({
      first_name: 'John',
      last_name: 'Doe',
      email: 'john@example.com',
      phone: '+1234567890',
      country: 'US'
    }, t);

    expect(valid.valid).toBe(true);

    const invalid = validateLeadForm({
      first_name: '',
      email: 'invalid'
    }, t);

    expect(invalid.valid).toBe(false);
    expect(Object.keys(invalid.errors).length).toBeGreaterThan(0);
  });

  it('should sanitize form data', () => {
    const sanitized = sanitizeFormData({
      first_name: '  John  ',
      email: '  JOHN@EXAMPLE.COM  ',
      phone: '(123) 456-7890'
    });

    expect(sanitized.first_name).toBe('John');
    expect(sanitized.email).toBe('john@example.com');
    expect(sanitized.phone).toBe('1234567890');
  });
});

describe('countdown', () => {
  it('should calculate countdown correctly', () => {
    const future = new Date();
    future.setDate(future.getDate() + 1);
    
    const state = calculateCountdown(future.toISOString());
    expect(state.expired).toBe(false);
    expect(state.days).toBeGreaterThanOrEqual(0);
  });

  it('should detect expired countdown', () => {
    const past = new Date('2020-01-01');
    const state = calculateCountdown(past.toISOString());
    expect(state.expired).toBe(true);
  });
});










