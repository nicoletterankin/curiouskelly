export interface Region {
  code: string;
  name: string;
}

export interface Country {
  code: string;
  name: string;
  regions: Region[];
}

export const countries: Country[] = [
  {
    code: 'US',
    name: 'United States',
    regions: [
      { code: 'AL', name: 'Alabama' },
      { code: 'CA', name: 'California' },
      { code: 'NY', name: 'New York' },
      { code: 'TX', name: 'Texas' },
      { code: 'WA', name: 'Washington' }
    ]
  },
  {
    code: 'CA',
    name: 'Canada',
    regions: [
      { code: 'AB', name: 'Alberta' },
      { code: 'BC', name: 'British Columbia' },
      { code: 'ON', name: 'Ontario' },
      { code: 'QC', name: 'Québec' }
    ]
  },
  {
    code: 'MX',
    name: 'México',
    regions: [
      { code: 'CMX', name: 'Ciudad de México' },
      { code: 'NLE', name: 'Nuevo León' },
      { code: 'JAL', name: 'Jalisco' }
    ]
  },
  {
    code: 'ES',
    name: 'España',
    regions: [
      { code: 'MD', name: 'Madrid' },
      { code: 'CT', name: 'Cataluña' },
      { code: 'AN', name: 'Andalucía' }
    ]
  },
  {
    code: 'BR',
    name: 'Brasil',
    regions: [
      { code: 'SP', name: 'São Paulo' },
      { code: 'RJ', name: 'Rio de Janeiro' },
      { code: 'MG', name: 'Minas Gerais' },
      { code: 'BA', name: 'Bahia' }
    ]
  },
  {
    code: 'PT',
    name: 'Portugal',
    regions: [
      { code: 'LX', name: 'Lisboa' },
      { code: 'PT11', name: 'Norte' },
      { code: 'PT15', name: 'Algarve' }
    ]
  },
  {
    code: 'GB',
    name: 'United Kingdom',
    regions: [
      { code: 'ENG', name: 'England' },
      { code: 'SCT', name: 'Scotland' },
      { code: 'WLS', name: 'Wales' },
      { code: 'NIR', name: 'Northern Ireland' }
    ]
  },
  {
    code: 'FR',
    name: 'France',
    regions: [
      { code: 'IDF', name: 'Île-de-France' },
      { code: 'NAQ', name: 'Nouvelle-Aquitaine' },
      { code: 'ARA', name: 'Auvergne-Rhône-Alpes' }
    ]
  },
  {
    code: 'DE',
    name: 'Deutschland',
    regions: [
      { code: 'BY', name: 'Bayern' },
      { code: 'BE', name: 'Berlin' },
      { code: 'NRW', name: 'Nordrhein-Westfalen' }
    ]
  },
  {
    code: 'AU',
    name: 'Australia',
    regions: [
      { code: 'NSW', name: 'New South Wales' },
      { code: 'VIC', name: 'Victoria' },
      { code: 'QLD', name: 'Queensland' }
    ]
  },
  {
    code: 'OTHER',
    name: 'Other / Not listed',
    regions: [{ code: 'OTHER', name: 'Other / Not listed' }]
  }
];

export function getCountryByCode(code: string): Country | undefined {
  return countries.find((country) => country.code === code);
}

export function getRegionsForCountry(code: string): Region[] {
  return getCountryByCode(code)?.regions ?? [];
}

