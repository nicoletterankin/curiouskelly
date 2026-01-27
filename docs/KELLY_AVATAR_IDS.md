# Kelly Avatar IDs - COMPLETE REFERENCE

**Last Updated:** Jan 18, 2026
**Source:** HeyGen Dashboard - Verified by User

## IMPORTANT: API Format

HeyGen requires **TWO** fields for custom avatars:
- `avatar_id` - The avatar GROUP (age variant)
- `look_id` - The specific look/archetype

```typescript
// CORRECT FORMAT
{
  character: {
    type: 'avatar',
    avatar_id: 'a762125d3107477aba43d1bd79f90d6e', // Adult Kelly group
    look_id: '3d6a9d6f91b444469dae87ebb3d9eba6'    // Storyteller look
  }
}
```

---

## Avatar Group IDs (3 Ages)

| Age | avatar_id |
|-----|-----------|
| **adult** | `a762125d3107477aba43d1bd79f90d6e` |
| **kid** | `93bb788b97d847409ad7dcf69702ece5` |
| **senior** | `d8c4ffac39a546a682b603c56e15906a` |

---

## Adult Kelly Looks (12 Archetypes)

| Archetype | look_id |
|-----------|---------|
| storyteller | `3d6a9d6f91b444469dae87ebb3d9eba6` |
| explorer | `62516885ca4b4eae8f63b87b8c060e25` |
| scientist | `277aba5b86a14ff2a4eca2eab2402ab3` |
| architect | `35d0115505824e3182eb9d2ee8cfe73d` |
| strategist | `08d53d1b065041bda2e5b6bc32962a8a` |
| diplomat | `c3cdbe48fe274420a7f45a4da7e366aa` |
| mystic | `dfaf9fbd644a475595b178f0be65a39a` |
| rebel | `390be3fb2b064883bb2304fc3968fd87` |
| macgyver | `c5aab6ab13d940f8ae4700d546bd6b6b` |
| empath | `6bb1a05678c64213a1ed3a4dc790b81e` |
| provider | `9a143feeb2994989b034cebeb78753be` |
| survivor | `831c8d6048104ba0b03a74a36543cfb9` |

---

## Kid Kelly Looks (12 Archetypes)

| Archetype | look_id |
|-----------|---------|
| scientist | `82813816115c4fbe93b3f3f211bd9931` |
| explorer | `fa4a6780e25a49699ee4f75cb1f03103` |
| rebel | `d4e960f7a3424d869877f3a951adfae7` |
| architect | `cc1dd0e9e2fd432099985c9b036ed836` |
| diplomat | `48bddc41ae94473caa645ce9ab93136d` |
| empath | `deeb27f2648848b48c5c1ce59059bd54` |
| macgyver | `7b6ab196f2c7430b945411df51a84c58` |
| mystic | `5cff601bfb344015a65ff46c6b8cd70a` |
| provider | `deaa213342944dc2bf671abe1442e316` |
| storyteller | `1024bc304a1146998bc4c360173b2c48` |
| strategist | `6249632f58ce479891de00b4da5fb88d` |
| survivor | `bd579e4ca77444aca2bfea8ee9070830` |

---

## Senior Kelly Looks (12 Archetypes)

| Archetype | look_id |
|-----------|---------|
| scientist | `97e1c9dc1ed04e8fa357c69bde34e58e` |
| explorer | `c38e30f2a3cf4e81b0365abf41579f22` |
| architect | `42e9197ab9d84961915b00d5cc780190` |
| empath | `493dac2cf2ba4509b3cc048ff819765e` |
| diplomat | `a82183881e284e3782db75b755c3f080` |
| macgyver | `cb5b025506284d64b696e296ca2feead` |
| provider | `12582467e9ff48889d7b2435642e2d65` |
| storyteller | `98178c87897e4421884b535b7864ba86` |
| strategist | `e4ab0d4d1f1b4dc9b81a1076b018557f` |
| rebel | `dc835263eaa247f5b0e06106b848df18` |
| mystic | `c6d104b2ca354b0a9593cb840988bf6e` |
| survivor | `9a143feeb2994989b034cebeb78753be` |

---

## Age Group Mapping

```typescript
function getKellyAvatar(age: number, archetype: string = 'storyteller') {
  // Map numeric age to avatar group
  let avatarId: string;
  let lookMap: Record<string, string>;
  
  if (age <= 12) {
    avatarId = '93bb788b97d847409ad7dcf69702ece5'; // kid
    lookMap = KID_LOOKS;
  } else if (age >= 55) {
    avatarId = 'd8c4ffac39a546a682b603c56e15906a'; // senior
    lookMap = SENIOR_LOOKS;
  } else {
    avatarId = 'a762125d3107477aba43d1bd79f90d6e'; // adult
    lookMap = ADULT_LOOKS;
  }
  
  return {
    avatar_id: avatarId,
    look_id: lookMap[archetype] || lookMap['storyteller']
  };
}
```

---

## Notes

- **Audio Issues:** ElevenLabs audio needs regeneration (quality issues with previous batch)
- **Total Avatars:** 36 (3 ages × 12 archetypes)
- **Default:** Use `storyteller` archetype as default for most lessons
