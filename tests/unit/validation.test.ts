import { describe, it, expect } from 'vitest';
import { validateLead, hasErrors } from '@lib/validation';
import { enUS } from '@lib/i18n/en-us';

const copy = enUS.leadForm;

describe('validateLead', () => {
  it('accepts a valid payload', () => {
    const payload = {
      first_name: 'Kelly',
      last_name: 'Rivera',
      email: 'kelly@example.com',
      phone: '+12025550123',
      country: 'US',
      region: 'CA',
      marketing_opt_in: true,
      locale: 'en-US',
      journey: 'home'
    };
    const errors = validateLead(payload, copy);
    expect(hasErrors(errors)).toBe(false);
  });

  it('flags invalid email and phone', () => {
    const payload = {
      first_name: 'Kelly',
      last_name: 'Rivera',
      email: 'invalid-email',
      phone: '123',
      country: '',
      region: '',
      marketing_opt_in: false,
      locale: '',
      journey: ''
    };
    const errors = validateLead(payload, copy);
    expect(errors.email).toBeDefined();
    expect(errors.phone).toBeDefined();
    expect(errors.country).toBeDefined();
    expect(errors.region).toBeDefined();
    expect(errors.locale).toBeDefined();
    expect(errors.journey).toBeDefined();
    expect(hasErrors(errors)).toBe(true);
  });
});

