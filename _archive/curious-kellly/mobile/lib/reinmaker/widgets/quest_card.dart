import 'package:flutter/material.dart';

import '../models/player_state.dart';
import '../models/tribe_pack.dart';

class QuestCard extends StatelessWidget {
  const QuestCard({
    super.key,
    required this.summary,
    required this.progress,
    required this.tribe,
    required this.onTap,
  });

  final QuestSummary summary;
  final QuestProgress? progress;
  final TribePack tribe;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final status = progress?.status ?? 'available';
    final isCompleted = status == 'completed';
    final attempts = progress?.attempts ?? 0;

    return Card(
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      elevation: isCompleted ? 6 : 2,
      shadowColor: _colorFromHex(tribe.color).withOpacity(0.4),
      child: InkWell(
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                _colorFromHex(tribe.color).withOpacity(0.85),
                Colors.black.withOpacity(0.75),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.35),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      'Tier ${summary.tier}',
                      style: theme.textTheme.labelMedium?.copyWith(color: Colors.white70),
                    ),
                  ),
                  const Spacer(),
                  Icon(
                    isCompleted ? Icons.check_circle : Icons.play_circle_fill,
                    color: isCompleted ? Colors.limeAccent : Colors.white70,
                  )
                ],
              ),
              const SizedBox(height: 16),
              Text(
                summary.id,
                style: theme.textTheme.titleMedium?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _questSubtitle,
                style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
              ),
              const Spacer(),
              Row(
                children: [
                  _RewardPill(
                    icon: Icons.bolt,
                    label: '+${summary.rewards.xp} XP',
                  ),
                  if (summary.rewards.stoneId != null) ...[
                    const SizedBox(width: 12),
                    _RewardPill(
                      icon: Icons.auto_awesome,
                      label: summary.rewards.stoneId!,
                    ),
                  ],
                  const Spacer(),
                  Text(
                    attempts > 0 ? '$attempts attempts' : 'New',
                    style: theme.textTheme.labelSmall?.copyWith(color: Colors.white60),
                  )
                ],
              )
            ],
          ),
        ),
      ),
    );
  }

  String get _questSubtitle {
    switch (summary.kind) {
      case 'dialogue':
        return 'Dialogue journey';
      case 'puzzle':
        return 'Puzzle challenge';
      case 'builder':
        return 'Builder sprint';
      case 'empathy':
        return 'Empathy reflection';
      default:
        return 'Quest';
    }
  }

  Color _colorFromHex(String hex) {
    final buffer = StringBuffer();
    if (hex.length == 6 || hex.length == 7) buffer.write('ff');
    buffer.write(hex.replaceFirst('#', ''));
    return Color(int.parse(buffer.toString(), radix: 16));
  }
}

class _RewardPill extends StatelessWidget {
  const _RewardPill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.25),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Icon(icon, size: 16, color: Colors.white70),
          const SizedBox(width: 6),
          Text(
            label,
            style: Theme.of(context)
                .textTheme
                .labelMedium
                ?.copyWith(color: Colors.white70),
          ),
        ],
      ),
    );
  }
}




















