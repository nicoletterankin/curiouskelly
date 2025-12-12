import 'tribe_pack.dart';

enum QuestStepType { dialogue, puzzle, builder, empathy }

QuestStepType questStepTypeFromString(String value) {
  switch (value) {
    case 'dialogue':
      return QuestStepType.dialogue;
    case 'puzzle':
      return QuestStepType.puzzle;
    case 'builder':
      return QuestStepType.builder;
    case 'empathy':
    default:
      return QuestStepType.empathy;
  }
}

class QuestModel {
  QuestModel({
    required this.id,
    required this.tribe,
    required this.tier,
    required this.kind,
    required this.steps,
    required this.ageBuckets,
    required this.successCriteria,
    required this.rewards,
    this.lessonRef,
    this.localizationKey,
    this.estimatedDurationMin,
    this.captions = const {},
    this.audio,
  });

  final String id;
  final TribeId tribe;
  final int tier;
  final String kind;
  final List<QuestStep> steps;
  final List<String> ageBuckets;
  final Map<String, dynamic> successCriteria;
  final QuestRewards rewards;
  final String? lessonRef;
  final String? localizationKey;
  final int? estimatedDurationMin;
  final Map<String, String> captions;
  final Map<String, dynamic>? audio;

  factory QuestModel.fromJson(Map<String, dynamic> json) {
    return QuestModel(
      id: json['id'] as String,
      tribe: TribeId.fromDisplayName(json['tribe'] as String),
      tier: json['tier'] as int,
      kind: json['kind'] as String,
      steps: (json['steps'] as List<dynamic>)
          .map((step) => QuestStep.fromJson(step as Map<String, dynamic>))
          .toList(),
      ageBuckets: (json['ageBuckets'] as List<dynamic>).cast<String>(),
      successCriteria: json['successCriteria'] as Map<String, dynamic>,
      rewards: QuestRewards.fromJson(json['rewards'] as Map<String, dynamic>),
      lessonRef: json['lessonRef'] as String?,
      localizationKey: json['localizationKey'] as String?,
      estimatedDurationMin: json['estimatedDurationMin'] as int?,
      captions: (json['captions'] as Map<String, dynamic>? ?? {})
          .map((key, value) => MapEntry(key, value as String)),
      audio: json['audio'] as Map<String, dynamic>?,
    );
  }
}

class QuestStep {
  QuestStep({required this.type, required this.data});

  final QuestStepType type;
  final Map<String, dynamic> data;

  factory QuestStep.fromJson(Map<String, dynamic> json) {
    final type = questStepTypeFromString(json['type'] as String);
    final data = Map<String, dynamic>.from(json);
    data.remove('type');
    return QuestStep(type: type, data: data);
  }
}




















