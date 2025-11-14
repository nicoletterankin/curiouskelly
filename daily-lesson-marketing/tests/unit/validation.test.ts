import { describe, it, expect } from 'vitest';
import { validateLead, hasErrors, validateLeadForm, sanitizeFormData } from '../../src/lib/validation';
import { calculateCountdown } from '../../src/lib/countdown';
import { enUS } from '../../src/lib/i18n/en-us';

const copy = enUS.leadForm;

describe('validation', () => {
  it('should validate email correctly', () => {
    const valid = validateLead({ 
      first_name: 'John', 
      last_name: 'Doe', 
      email: 'test@example.com', 
      phone: '+12025550123', 
      country: 'US', 
      region: 'CA', 
      marketing_opt_in: false, 
      locale: 'en-US', 
      journey: 'home' 
    }, copy);
    expect(hasErrors(valid)).toBe(false);

    const invalid = validateLead({ 
      first_name: 'John', 
      last_name: 'Doe', 
      email: 'invalid', 
      phone: '+12025550123', 
      country: 'US', 
      region: 'CA', 
      marketing_opt_in: false, 
      locale: 'en-US', 
      journey: 'home' 
    }, copy);
    expect(invalid.email).toBeDefined();
    expect(hasErrors(invalid)).toBe(true);
  });

  it('should validate phone correctly', () => {
    const valid = validateLead({ 
      first_name: 'John', 
      last_name: 'Doe', 
      email: 'test@example.com', 
      phone: '+12025550123', 
      country: 'US', 
      region: 'CA', 
      marketing_opt_in: false, 
      locale: 'en-US', 
      journey: 'home' 
    }, copy);
    expect(hasErrors(valid)).toBe(false);

    const invalid = validateLead({ 
      first_name: 'John', 
      last_name: 'Doe', 
      email: 'test@example.com', 
      phone: '123', 
      country: 'US', 
      region: 'CA', 
      marketing_opt_in: false, 
      locale: 'en-US', 
      journey: 'home' 
    }, copy);
    expect(invalid.phone).toBeDefined();
    expect(hasErrors(invalid)).toBe(true);
  });

  it('should validate name correctly', () => {
    const valid = validateLead({ 
      first_name: 'John', 
      last_name: 'Doe', 
      email: 'test@example.com', 
      phone: '+12025550123', 
      country: 'US', 
      region: 'CA', 
      marketing_opt_in: false, 
      locale: 'en-US', 
      journey: 'home' 
    }, copy);
    expect(hasErrors(valid)).toBe(false);

    const invalid = validateLead({ 
      first_name: 'A', 
      last_name: 'Doe', 
      email: 'test@example.com', 
      phone: '+12025550123', 
      country: 'US', 
      region: 'CA', 
      marketing_opt_in: false, 
      locale: 'en-US', 
      journey: 'home' 
    }, copy);
    expect(invalid.first_name).toBeDefined();
    expect(hasErrors(invalid)).toBe(true);
  });

  it('should validate full form', () => {
    const valid = validateLeadForm({
      first_name: 'John',
      last_name: 'Doe',
      email: 'john@example.com',
      phone: '+12025550123',
      country: 'US',
      region: 'CA',
      marketing_opt_in: false,
      locale: 'en-US',
      journey: 'home'
    }, copy);

    expect(valid.valid).toBe(true);
    expect(Object.keys(valid.errors).length).toBe(0);

    const invalid = validateLeadForm({
      first_name: '',
      last_name: 'Doe',
      email: 'invalid',
      phone: '+12025550123',
      country: 'US',
      region: 'CA',
      marketing_opt_in: false,
      locale: 'en-US',
      journey: 'home'
    }, copy);

    expect(invalid.valid).toBe(false);
    expect(Object.keys(invalid.errors).length).toBeGreaterThan(0);
  });

  it('should sanitize form data', () => {
    const sanitized = sanitizeFormData({
      first_name: '  John  ',
      last_name: '  Doe  ',
      email: '  JOHN@EXAMPLE.COM  ',
      phone: '  +12025550123  ',
      country: 'US',
      region: 'CA',
      marketing_opt_in: true,
      locale: 'en-US',
      journey: 'home'
    });

    expect(sanitized.first_name).toBe('John');
    expect(sanitized.last_name).toBe('Doe');
    expect(sanitized.email).toBe('john@example.com');
    expect(sanitized.phone).toBe('+12025550123');
    expect(sanitized.marketing_opt_in).toBe(true);
  });
});

describe('countdown', () => {
  it('should calculate countdown correctly', () => {
    const future = new Date();
    future.setDate(future.getDate() + 1);
    
    const state = calculateCountdown(future);
    expect(state.active).toBe(true);
    expect(state.days).toBeGreaterThanOrEqual(0);
  });

  it('should detect expired countdown', () => {
    const past = new Date('2020-01-01');
    const state = calculateCountdown(past);
    expect(state.active).toBe(false);
  });
});











