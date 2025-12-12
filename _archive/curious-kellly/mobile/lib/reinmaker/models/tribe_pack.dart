import 'lens.dart';

enum TribeId {
  light('Light'),
  stone('Stone'),
  metal('Metal'),
  code('Code'),
  air('Air'),
  water('Water'),
  fire('Fire');

  const TribeId(this.displayName);

  final String displayName;

  String get key => name;

  static TribeId fromDisplayName(String value) {
    return TribeId.values.firstWhere(
      (tribe) => tribe.displayName.toLowerCase() == value.toLowerCase(),
      orElse: () => TribeId.light,
    );
  }
}

class TribePack {
  TribePack({
    required this.id,
    required this.tribe,
    required this.lensId,
    required this.color,
    required this.icon,
    required this.featuredQuoteKey,
    required this.ageDifficulty,
    required this.tiers,
    required this.finaleStoneId,
  });

  final String id;
  final TribeId tribe;
  final LensId lensId;
  final String color;
  final String icon;
  final String featuredQuoteKey;
  final Map<String, AgeDifficultyPreset> ageDifficulty;
  final List<TribeTier> tiers;
  final String finaleStoneId;

  factory TribePack.fromJson(Map<String, dynamic> json) {
    final ageDifficultyJson = json['ageDifficulty'] as Map<String, dynamic>? ?? {};
    return TribePack(
      id: json['id'] as String,
      tribe: TribeId.fromDisplayName(json['tribe'] as String),
      lensId: LensId.fromValue(json['lensId'] as String),
      color: json['color'] as String,
      icon: json['icon'] as String,
      featuredQuoteKey: json['featuredQuoteKey'] as String,
      ageDifficulty: ageDifficultyJson.map(
        (key, value) => MapEntry(key, AgeDifficultyPreset.fromJson(value as Map<String, dynamic>)),
      ),
      tiers: (json['tiers'] as List<dynamic>)
          .map((e) => TribeTier.fromJson(e as Map<String, dynamic>))
          .toList(),
      finaleStoneId: json['finaleStoneId'] as String,
    );
  }

  TribeTier? tierForLevel(int tier) {
    return tiers.firstWhere(
      (value) => value.tier == tier,
      orElse: () => TribeTier.empty(tier),
    );
  }
}

class AgeDifficultyPreset {
  AgeDifficultyPreset({required this.difficulty, required this.hintCadence});

  final String difficulty;
  final String hintCadence;

  factory AgeDifficultyPreset.fromJson(Map<String, dynamic> json) {
    return AgeDifficultyPreset(
      difficulty: json['difficulty'] as String? ?? 'spark',
      hintCadence: json['hintCadence'] as String? ?? 'moderate',
    );
  }
}

class TribeTier {
  TribeTier({
    required this.tier,
    required this.stoneId,
    required this.quests,
    required this.lessonRefs,
  });

  final int tier;
  final String stoneId;
  final List<String> quests;
  final List<String> lessonRefs;

  factory TribeTier.fromJson(Map<String, dynamic> json) {
    return TribeTier(
      tier: json['tier'] as int,
      stoneId: json['stoneId'] as String,
      quests: (json['quests'] as List<dynamic>).cast<String>(),
      lessonRefs: (json['lessonRefs'] as List<dynamic>).cast<String>(),
    );
  }

  factory TribeTier.empty(int tier) {
    return TribeTier(tier: tier, stoneId: '', quests: const [], lessonRefs: const []);
  }
}

class QuestSummary {
  QuestSummary({
    required this.id,
    required this.tribe,
    required this.tier,
    required this.kind,
    required this.ageBuckets,
    required this.rewards,
    required this.captions,
    this.lessonRef,
    this.estimatedDurationMin,
    this.localizationKey,
    this.audio,
  });

  final String id;
  final TribeId tribe;
  final int tier;
  final String kind;
  final List<String> ageBuckets;
  final QuestRewards rewards;
  final Map<String, String> captions;
  final String? lessonRef;
  final int? estimatedDurationMin;
  final String? localizationKey;
  final Map<String, dynamic>? audio;

  factory QuestSummary.fromJson(Map<String, dynamic> json) {
    return QuestSummary(
      id: json['id'] as String,
      tribe: TribeId.fromDisplayName(json['tribe'] as String),
      tier: json['tier'] as int,
      kind: json['kind'] as String,
      ageBuckets: (json['ageBuckets'] as List<dynamic>).cast<String>(),
      lessonRef: json['lessonRef'] as String?,
      estimatedDurationMin: json['estimatedDurationMin'] as int?,
      localizationKey: json['localizationKey'] as String?,
      rewards: QuestRewards.fromJson(json['rewards'] as Map<String, dynamic>),
      captions: (json['captions'] as Map<String, dynamic>? ?? {})
          .map((key, value) => MapEntry(key, value as String)),
      audio: json['audio'] as Map<String, dynamic>?,
    );
  }
}

class QuestRewards {
  QuestRewards({required this.xp, this.cosmetics = const [], this.stoneId});

  final int xp;
  final List<String> cosmetics;
  final String? stoneId;

  factory QuestRewards.fromJson(Map<String, dynamic> json) {
    return QuestRewards(
      xp: json['xp'] as int? ?? 0,
      cosmetics: (json['cosmetics'] as List<dynamic>? ?? const []).cast<String>(),
      stoneId: json['stoneId'] as String?,
    );
  }
}

class LocaleBundle {
  LocaleBundle({required this.path, required this.hash, required this.keys});

  final String path;
  final String hash;
  final List<String> keys;

  factory LocaleBundle.fromJson(Map<String, dynamic> json) {
    return LocaleBundle(
      path: json['path'] as String,
      hash: json['hash'] as String,
      keys: (json['keys'] as List<dynamic>).cast<String>(),
    );
  }
}

class ReinmakerManifest {
  ReinmakerManifest({
    required this.version,
    required this.generatedAt,
    required this.featuredRotation,
    required this.locales,
    required this.tribes,
    required this.quests,
    required this.assets,
    required this.contentHash,
  });

  final String version;
  final DateTime generatedAt;
  final List<String> featuredRotation;
  final Map<String, LocaleBundle> locales;
  final List<TribePack> tribes;
  final List<QuestSummary> quests;
  final Map<String, List<String>> assets;
  final String contentHash;

  factory ReinmakerManifest.fromJson(Map<String, dynamic> json) {
    final localesJson = json['locales'] as Map<String, dynamic>? ?? {};
    final assetsJson = json['assets'] as Map<String, dynamic>? ?? {};

    return ReinmakerManifest(
      version: json['version'] as String,
      generatedAt: DateTime.tryParse(json['generatedAt'] as String? ?? '') ?? DateTime.now(),
      featuredRotation: (json['featuredRotation'] as List<dynamic>? ?? const [])
          .map((e) => e.toString())
          .toList(),
      locales: localesJson.map(
        (key, value) => MapEntry(key, LocaleBundle.fromJson(value as Map<String, dynamic>)),
      ),
      tribes: (json['tribes'] as List<dynamic>)
          .map((e) => TribePack.fromJson(e as Map<String, dynamic>))
          .toList(),
      quests: (json['quests'] as List<dynamic>)
          .map((e) => QuestSummary.fromJson(e as Map<String, dynamic>))
          .toList(),
      assets: assetsJson.map(
        (key, value) => MapEntry(key, (value as List<dynamic>).map((e) => e.toString()).toList()),
      ),
      contentHash: json['contentHash'] as String? ?? '',
    );
  }

  TribePack? packForTribe(TribeId tribeId) {
    for (final pack in tribes) {
      if (pack.tribe == tribeId) {
        return pack;
      }
    }
    return null;
  }

  QuestSummary? questById(String questId) {
    for (final quest in quests) {
      if (quest.id == questId) {
        return quest;
      }
    }
    return null;
  }
}

