import { parsePhoneNumberFromString } from 'libphonenumber-js/min';
import type { LeadFormCopy } from './i18n/types';

export interface LeadPayload {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  country: string;
  region: string;
  marketing_opt_in: boolean;
  locale: string;
  journey: string;
}

export type LeadErrors = Partial<Record<keyof LeadPayload | 'turnstile', string>>;

const namePattern = /^[\p{L}\p{M}' -]{2,}$/u;
const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function validateLead(payload: LeadPayload, copy: LeadFormCopy): LeadErrors {
  const errors: LeadErrors = {};

  if (!payload.first_name || !namePattern.test(payload.first_name.trim())) {
    errors.first_name = payload.first_name
      ? copy.fields.firstName.errors.invalid ?? copy.fields.firstName.errors.required
      : copy.fields.firstName.errors.required;
  }

  if (!payload.last_name || !namePattern.test(payload.last_name.trim())) {
    errors.last_name = payload.last_name
      ? copy.fields.lastName.errors.invalid ?? copy.fields.lastName.errors.required
      : copy.fields.lastName.errors.required;
  }

  if (!payload.email || !emailPattern.test(payload.email.trim().toLowerCase())) {
    errors.email = payload.email
      ? copy.fields.email.errors.invalid ?? copy.fields.email.errors.required
      : copy.fields.email.errors.required;
  }

  if (!payload.country) {
    errors.country = copy.fields.country.errors.required ?? 'Country is required.';
  }

  if (!payload.region) {
    errors.region = copy.fields.region.errors.required ?? 'Region is required.';
  }

  if (!payload.phone) {
    errors.phone = copy.fields.phone.errors.required ?? 'Phone number is required.';
  } else {
    const parsed = parsePhoneNumberFromString(payload.phone);
    if (!parsed || !parsed.isValid()) {
      errors.phone = copy.fields.phone.errors.invalid ?? 'Invalid phone number.';
    }
  }

  if (!payload.locale) {
    errors.locale = 'Locale is required.';
  }

  if (!payload.journey) {
    errors.journey = 'Journey is required.';
  }

  return errors;
}

export function hasErrors(errors: LeadErrors): boolean {
  return Object.keys(errors).length > 0;
}












