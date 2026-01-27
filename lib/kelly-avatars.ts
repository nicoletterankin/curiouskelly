/**
 * Kelly Avatar IDs for HeyGen API
 * 
 * IMPORTANT: HeyGen requires BOTH avatar_id AND look_id
 * - avatar_id: The age variant (kid, adult, senior)
 * - look_id: The archetype/personality look
 * 
 * @see docs/KELLY_AVATAR_IDS.md for full reference
 */

// Avatar Group IDs (by age)
export const KELLY_AVATAR_GROUPS = {
  kid: '93bb788b97d847409ad7dcf69702ece5',     // ages <= 12
  adult: 'a762125d3107477aba43d1bd79f90d6e',   // ages 13-54
  senior: 'd8c4ffac39a546a682b603c56e15906a',  // ages 55+
} as const;

// Adult Kelly Look IDs (12 archetypes)
export const ADULT_LOOKS = {
  storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
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
  survivor: '831c8d6048104ba0b03a74a36543cfb9',
} as const;

// Kid Kelly Look IDs (12 archetypes)
export const KID_LOOKS = {
  scientist: '82813816115c4fbe93b3f3f211bd9931',
  explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
  rebel: 'd4e960f7a3424d869877f3a951adfae7',
  architect: 'cc1dd0e9e2fd432099985c9b036ed836',
  diplomat: '48bddc41ae94473caa645ce9ab93136d',
  empath: 'deeb27f2648848b48c5c1ce59059bd54',
  macgyver: '7b6ab196f2c7430b945411df51a84c58',
  mystic: '5cff601bfb344015a65ff46c6b8cd70a',
  provider: 'deaa213342944dc2bf671abe1442e316',
  storyteller: '1024bc304a1146998bc4c360173b2c48',
  strategist: '6249632f58ce479891de00b4da5fb88d',
  survivor: 'bd579e4ca77444aca2bfea8ee9070830',
} as const;

// Senior Kelly Look IDs (12 archetypes)
export const SENIOR_LOOKS = {
  scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
  explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
  architect: '42e9197ab9d84961915b00d5cc780190',
  empath: '493dac2cf2ba4509b3cc048ff819765e',
  diplomat: 'a82183881e284e3782db75b755c3f080',
  macgyver: 'cb5b025506284d64b696e296ca2feead',
  provider: '12582467e9ff48889d7b2435642e2d65',
  storyteller: '98178c87897e4421884b535b7864ba86',
  strategist: 'e4ab0d4d1f1b4dc9b81a1076b018557f',
  rebel: 'dc835263eaa247f5b0e06106b848df18',
  mystic: 'c6d104b2ca354b0a9593cb840988bf6e',
  survivor: '9a143feeb2994989b034cebeb78753be',
} as const;

export type KellyArchetype = keyof typeof ADULT_LOOKS;
export type KellyAgeGroup = 'kid' | 'adult' | 'senior';

/**
 * Get HeyGen avatar_id and look_id for a given age and archetype
 */
export function getKellyAvatar(
  age: number, 
  archetype: KellyArchetype = 'storyteller'
): { avatar_id: string; look_id: string; age_group: KellyAgeGroup } {
  let avatarId: string;
  let lookMap: Record<string, string>;
  let ageGroup: KellyAgeGroup;
  
  if (age <= 12) {
    avatarId = KELLY_AVATAR_GROUPS.kid;
    lookMap = KID_LOOKS;
    ageGroup = 'kid';
  } else if (age >= 55) {
    avatarId = KELLY_AVATAR_GROUPS.senior;
    lookMap = SENIOR_LOOKS;
    ageGroup = 'senior';
  } else {
    avatarId = KELLY_AVATAR_GROUPS.adult;
    lookMap = ADULT_LOOKS;
    ageGroup = 'adult';
  }
  
  return {
    avatar_id: avatarId,
    look_id: lookMap[archetype] || lookMap['storyteller'],
    age_group: ageGroup
  };
}

/**
 * HeyGen API payload builder for Kelly video generation
 */
export function buildHeyGenPayload(
  audioUrl: string,
  age: number,
  archetype: KellyArchetype = 'storyteller',
  options: {
    width?: number;
    height?: number;
    background?: string;
    test?: boolean;
  } = {}
) {
  const { avatar_id, look_id } = getKellyAvatar(age, archetype);
  
  return {
    video_inputs: [{
      character: {
        type: 'avatar',
        avatar_id,
        look_id,
        avatar_style: 'normal'
      },
      voice: {
        type: 'audio',
        audio_url: audioUrl
      },
      background: {
        type: 'color',
        value: options.background || '#FFFFFF'
      }
    }],
    dimension: { 
      width: options.width || 1080, 
      height: options.height || 1920 
    },
    aspect_ratio: '9:16',
    test: options.test ?? false
  };
}
