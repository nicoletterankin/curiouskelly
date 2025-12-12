import 'package:flutter/material.dart';

import '../models/lens.dart';

class LensBadge extends StatelessWidget {
  const LensBadge({super.key, required this.lensId, required this.level});

  final LensId lensId;
  final int level;

  String get _label {
    switch (lensId) {
      case LensId.uiComposition:
        return 'UI Composition';
      case LensId.systemDesign:
        return 'System Design';
      case LensId.mechProto:
        return 'Mech Proto';
      case LensId.algReasoning:
        return 'Algorithmic Reasoning';
      case LensId.dialogueEmpathy:
        return 'Dialogue Empathy';
      case LensId.metaReflection:
        return 'Meta Reflection';
      case LensId.challengeMastery:
        return 'Challenge Mastery';
    }
  }

  IconData get _icon {
    switch (lensId) {
      case LensId.uiComposition:
        return Icons.palette_outlined;
      case LensId.systemDesign:
        return Icons.architecture_outlined;
      case LensId.mechProto:
        return Icons.precision_manufacturing_outlined;
      case LensId.algReasoning:
        return Icons.code;
      case LensId.dialogueEmpathy:
        return Icons.chat_bubble_outline;
      case LensId.metaReflection:
        return Icons.insights_outlined;
      case LensId.challengeMastery:
        return Icons.local_fire_department_outlined;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = theme.colorScheme.secondary;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          colors: [color.withOpacity(0.75), color.withOpacity(0.35)],
        ),
        border: Border.all(color: color.withOpacity(0.8), width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(_icon, size: 18, color: Colors.white),
          const SizedBox(width: 8),
          Text(
            _label,
            style: theme.textTheme.labelLarge?.copyWith(color: Colors.white),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.3),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              'Lv $level',
              style: theme.textTheme.labelSmall?.copyWith(color: Colors.white70),
            ),
          ),
        ],
      ),
    );
  }
}




















