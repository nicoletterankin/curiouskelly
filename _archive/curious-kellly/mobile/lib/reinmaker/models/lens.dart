enum LensId {
  uiComposition('lens.ui_composition'),
  systemDesign('lens.system_design'),
  mechProto('lens.mech_proto'),
  algReasoning('lens.alg_reasoning'),
  dialogueEmpathy('lens.dialogue_empathy'),
  metaReflection('lens.meta_reflection'),
  challengeMastery('lens.challenge_mastery');

  const LensId(this.value);

  final String value;

  static LensId fromValue(String value) {
    return LensId.values.firstWhere(
      (lens) => lens.value == value,
      orElse: () => LensId.uiComposition,
    );
  }
}

class LensProgress {
  LensProgress({
    required this.lensId,
    required this.level,
    this.unlockedAt,
  });

  String lensId;
  int level;
  DateTime? unlockedAt;

  LensProgress copyWith({int? level, DateTime? unlockedAt}) {
    return LensProgress(
      lensId: lensId,
      level: level ?? this.level,
      unlockedAt: unlockedAt ?? this.unlockedAt,
    );
  }
}




















