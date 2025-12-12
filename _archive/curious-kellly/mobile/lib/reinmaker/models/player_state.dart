import 'package:hive/hive.dart';

import 'lens.dart';

const _tribeKeys = ['light', 'stone', 'metal', 'code', 'air', 'water', 'fire'];

const _stoneTierByIndex = {
  1: 'spark',
  2: 'craft',
  3: 'mastery',
};

class PlayerState extends HiveObject {
  PlayerState({
    required this.version,
    required this.xp,
    required this.lenses,
    required this.stones,
    required this.quests,
    required this.unlockedCosmetics,
    required this.settings,
    this.ageBucket,
    this.activeCosmetic,
    this.featuredTribe,
    this.lastFeaturedRotation,
  });

  String version;
  String? ageBucket;
  int xp;
  Map<String, LensProgress> lenses;
  Map<String, List<String>> stones;
  Map<String, QuestProgress> quests;
  List<String> unlockedCosmetics;
  ActiveCosmetic? activeCosmetic;
  ReinmakerSettings settings;
  String? featuredTribe;
  DateTime? lastFeaturedRotation;

  static PlayerState initial() {
    return PlayerState(
      version: 'rmk.v1',
      xp: 0,
      lenses: {},
      stones: {for (final tribe in _tribeKeys) tribe: <String>[]},
      quests: {},
      unlockedCosmetics: <String>[],
      settings: ReinmakerSettings.defaults(),
      featuredTribe: 'Light',
    );
  }

  String stoneIdForTier(String tribeKey, int tier) {
    final suffix = _stoneTierByIndex[tier];
    if (suffix == null) {
      throw ArgumentError('Invalid tier: $tier');
    }
    return '$tribeKey.$suffix';
  }
}

class PlayerStateAdapter extends TypeAdapter<PlayerState> {
  @override
  final int typeId = 90;

  @override
  PlayerState read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{};
    for (var i = 0; i < numOfFields; i++) {
      fields[reader.readByte()] = reader.read();
    }

    return PlayerState(
      version: fields[0] as String,
      ageBucket: fields[1] as String?,
      xp: fields[2] as int,
      lenses: (fields[3] as Map?)?.map(
            (key, value) => MapEntry(key as String, value as LensProgress),
          ) ??
          <String, LensProgress>{},
      stones: (fields[4] as Map?)?.map(
            (key, value) => MapEntry(key as String, (value as List).cast<String>()),
          ) ??
          {for (final tribe in _tribeKeys) tribe: <String>[]},
      quests: (fields[5] as Map?)?.map(
            (key, value) => MapEntry(key as String, value as QuestProgress),
          ) ??
          <String, QuestProgress>{},
      unlockedCosmetics: (fields[6] as List?)?.cast<String>() ?? <String>[],
      activeCosmetic: fields[7] as ActiveCosmetic?,
      settings: fields[8] as ReinmakerSettings? ?? ReinmakerSettings.defaults(),
      featuredTribe: fields[9] as String?,
      lastFeaturedRotation: fields[10] as DateTime?,
    );
  }

  @override
  void write(BinaryWriter writer, PlayerState obj) {
    writer
      ..writeByte(11)
      ..writeByte(0)
      ..write(obj.version)
      ..writeByte(1)
      ..write(obj.ageBucket)
      ..writeByte(2)
      ..write(obj.xp)
      ..writeByte(3)
      ..write(obj.lenses)
      ..writeByte(4)
      ..write(obj.stones)
      ..writeByte(5)
      ..write(obj.quests)
      ..writeByte(6)
      ..write(obj.unlockedCosmetics)
      ..writeByte(7)
      ..write(obj.activeCosmetic)
      ..writeByte(8)
      ..write(obj.settings)
      ..writeByte(9)
      ..write(obj.featuredTribe)
      ..writeByte(10)
      ..write(obj.lastFeaturedRotation);
  }
}

class LensProgressAdapter extends TypeAdapter<LensProgress> {
  @override
  final int typeId = 91;

  @override
  LensProgress read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{};
    for (var i = 0; i < numOfFields; i++) {
      fields[reader.readByte()] = reader.read();
    }
    return LensProgress(
      lensId: fields[0] as String,
      level: fields[1] as int,
      unlockedAt: fields[2] as DateTime?,
    );
  }

  @override
  void write(BinaryWriter writer, LensProgress obj) {
    writer
      ..writeByte(3)
      ..writeByte(0)
      ..write(obj.lensId)
      ..writeByte(1)
      ..write(obj.level)
      ..writeByte(2)
      ..write(obj.unlockedAt);
  }
}

class QuestProgressAdapter extends TypeAdapter<QuestProgress> {
  @override
  final int typeId = 92;

  @override
  QuestProgress read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{};
    for (var i = 0; i < numOfFields; i++) {
      fields[reader.readByte()] = reader.read();
    }
    return QuestProgress(
      status: fields[0] as String,
      attempts: fields[1] as int,
      bestScore: fields[2] as double?,
      lastPlayedAt: fields[3] as DateTime?,
    );
  }

  @override
  void write(BinaryWriter writer, QuestProgress obj) {
    writer
      ..writeByte(4)
      ..writeByte(0)
      ..write(obj.status)
      ..writeByte(1)
      ..write(obj.attempts)
      ..writeByte(2)
      ..write(obj.bestScore)
      ..writeByte(3)
      ..write(obj.lastPlayedAt);
  }
}

class ReinmakerSettingsAdapter extends TypeAdapter<ReinmakerSettings> {
  @override
  final int typeId = 93;

  @override
  ReinmakerSettings read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{};
    for (var i = 0; i < numOfFields; i++) {
      fields[reader.readByte()] = reader.read();
    }
    return ReinmakerSettings(
      captionsEnabled: fields[0] as bool,
      iconOnlyMode: fields[1] as bool,
      highContrast: fields[2] as bool,
      textScale: fields[3] as double,
    );
  }

  @override
  void write(BinaryWriter writer, ReinmakerSettings obj) {
    writer
      ..writeByte(4)
      ..writeByte(0)
      ..write(obj.captionsEnabled)
      ..writeByte(1)
      ..write(obj.iconOnlyMode)
      ..writeByte(2)
      ..write(obj.highContrast)
      ..writeByte(3)
      ..write(obj.textScale);
  }
}

class ActiveCosmeticAdapter extends TypeAdapter<ActiveCosmetic> {
  @override
  final int typeId = 94;

  @override
  ActiveCosmetic read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{};
    for (var i = 0; i < numOfFields; i++) {
      fields[reader.readByte()] = reader.read();
    }
    return ActiveCosmetic(
      badgeId: fields[0] as String?,
      frameId: fields[1] as String?,
    );
  }

  @override
  void write(BinaryWriter writer, ActiveCosmetic obj) {
    writer
      ..writeByte(2)
      ..writeByte(0)
      ..write(obj.badgeId)
      ..writeByte(1)
      ..write(obj.frameId);
  }
}

class QuestProgress {
  QuestProgress({
    required this.status,
    required this.attempts,
    this.bestScore,
    this.lastPlayedAt,
  });

  String status;
  int attempts;
  double? bestScore;
  DateTime? lastPlayedAt;

  QuestProgress copyWith({
    String? status,
    int? attempts,
    double? bestScore,
    DateTime? lastPlayedAt,
  }) {
    return QuestProgress(
      status: status ?? this.status,
      attempts: attempts ?? this.attempts,
      bestScore: bestScore ?? this.bestScore,
      lastPlayedAt: lastPlayedAt ?? this.lastPlayedAt,
    );
  }
}

class ReinmakerSettings {
  ReinmakerSettings({
    required this.captionsEnabled,
    required this.iconOnlyMode,
    required this.highContrast,
    required this.textScale,
  });

  bool captionsEnabled;
  bool iconOnlyMode;
  bool highContrast;
  double textScale;

  factory ReinmakerSettings.defaults() {
    return ReinmakerSettings(
      captionsEnabled: true,
      iconOnlyMode: false,
      highContrast: false,
      textScale: 1.0,
    );
  }

  ReinmakerSettings copyWith({
    bool? captionsEnabled,
    bool? iconOnlyMode,
    bool? highContrast,
    double? textScale,
  }) {
    return ReinmakerSettings(
      captionsEnabled: captionsEnabled ?? this.captionsEnabled,
      iconOnlyMode: iconOnlyMode ?? this.iconOnlyMode,
      highContrast: highContrast ?? this.highContrast,
      textScale: textScale ?? this.textScale,
    );
  }
}

class ActiveCosmetic {
  ActiveCosmetic({this.badgeId, this.frameId});

  String? badgeId;
  String? frameId;
}




















