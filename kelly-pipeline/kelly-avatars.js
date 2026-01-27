/**
 * KELLY AVATAR IDS - Complete Reference
 * 
 * HeyGen uses TWO IDs for video generation:
 * 1. Avatar Group ID (by age) - The base avatar
 * 2. Look ID (by archetype) - The specific appearance
 */

// Avatar Group IDs by Age
const AVATAR_GROUPS = {
  kid: '93bb788b97d847409ad7dcf69702ece5',     // age <= 12
  adult: 'a762125d3107477aba43d1bd79f90d6e',   // age 13-54
  senior: 'd8c4ffac39a546a682b603c56e15906a'   // age 55+
};

// Look IDs by Age and Archetype
const LOOK_IDS = {
  kid: {
    storyteller: '1024bc304a1146998bc4c360173b2c48',
    explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
    scientist: '82813816115c4fbe93b3f3f211bd9931',
    architect: 'cc1dd0e9e2fd432099985c9b036ed836',
    strategist: '6249632f58ce479891de00b4da5fb88d',
    diplomat: '48bddc41ae94473caa645ce9ab93136d',
    mystic: '5cff601bfb344015a65ff46c6b8cd70a',
    rebel: 'd4e960f7a3424d869877f3a951adfae7',
    macgyver: '7b6ab196f2c7430b945411df51a84c58',
    empath: 'deeb27f2648848b48c5c1ce59059bd54',
    provider: 'deaa213342944dc2bf671abe1442e316',
    survivor: 'bd579e4ca77444aca2bfea8ee9070830'
  },
  adult: {
    storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',  // DEFAULT
    explorer: '62516885ca4b4eae8f63b87b8c060e25',
    scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
    architect: '35d0115505824e3182eb9d2ee8cfe73d',
    strategist: '08d53d1b065041bda2e5b6bc32962a8a',
    diplomat: 'c3cdbe48fe274420a7f45a4da7e366aa',
    mystic: 'dfaf9fbd644a475595b178f0be65a39a',
    rebel: '390be3fb2b064883bb2304fc3968fd87',
    macgyver: 'c5aab6ab13d940f8ae4700d546bd6b6b',
    empath: '6bb1a05678c64213a1ed3a4dc790b81e',
    provider: '9a143feeb2994989b034cebeb78753be',
    survivor: '831c8d6048104ba0b03a74a36543cfb9'
  },
  senior: {
    storyteller: '98178c87897e4421884b535b7864ba86',
    explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
    scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
    architect: '42e9197ab9d84961915b00d5cc780190',
    strategist: 'e4ab0d4d1f1b4dc9b81a1076b018557f',
    diplomat: 'a82183881e284e3782db75b755c3f080',
    mystic: 'c6d104b2ca354b0a9593cb840988bf6e',
    rebel: 'dc835263eaa247f5b0e06106b848df18',
    macgyver: 'cb5b025506284d64b696e296ca2feead',
    empath: '493dac2cf2ba4509b3cc048ff819765e',
    provider: '12582467e9ff48889d7b2435642e2d65',
    survivor: '9a143feeb2994989b034cebeb78753be'
  }
};

// Kelly's Voice ID (same across all avatars)
const KELLY_VOICE_ID = '1bd001e7e50f421d891986aad5158bc8';

// Map our age categories to HeyGen age groups
const AGE_MAP = {
  child: 'kid',
  adult: 'adult', 
  elder: 'senior'
};

/**
 * Get Kelly avatar IDs for video generation
 * @param {string} age - 'child', 'adult', or 'elder'
 * @param {string} archetype - Default 'storyteller'
 * @returns {{ avatar_id: string, look_id: string, voice_id: string }}
 */
function getKellyAvatar(age, archetype = 'storyteller') {
  const ageGroup = AGE_MAP[age] || 'adult';
  
  return {
    avatar_id: AVATAR_GROUPS[ageGroup],
    look_id: LOOK_IDS[ageGroup][archetype] || LOOK_IDS[ageGroup].storyteller,
    voice_id: KELLY_VOICE_ID,
    age_group: ageGroup
  };
}

module.exports = {
  AVATAR_GROUPS,
  LOOK_IDS,
  KELLY_VOICE_ID,
  AGE_MAP,
  getKellyAvatar
};
