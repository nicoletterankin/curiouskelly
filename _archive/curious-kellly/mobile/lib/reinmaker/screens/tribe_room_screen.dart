import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../router/app_router.dart';
import '../models/tribe_pack.dart';
import '../state/reinmaker_store.dart';
import '../widgets/quest_card.dart';

class TribeRoomScreen extends StatelessWidget {
  const TribeRoomScreen({super.key, this.tribeId});

  final TribeId? tribeId;

  @override
  Widget build(BuildContext context) {
    final store = context.watch<ReinmakerStore>();
    final manifest = store.manifest;
    if (manifest == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final tribe = tribeId != null ? manifest.packForTribe(tribeId!) : manifest.tribes.first;
    if (tribe == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Tribe Room')),
        body: const Center(child: Text('Tribe not found in manifest.')),
      );
    }

    final quests = manifest.quests.where((quest) => quest.tribe == tribe.tribe).toList()
      ..sort((a, b) => a.tier.compareTo(b.tier));

    return Scaffold(
      appBar: AppBar(
        title: Text('${tribe.tribe.displayName} · Tier Stones'),
        actions: [
          IconButton(
            tooltip: 'Knowledge Forge',
            icon: const Icon(Icons.handyman_outlined),
            onPressed: () => Navigator.pushNamed(context, AppRoutes.reinmakerForge),
          ),
        ],
      ),
      body: ListView.separated(
        padding: const EdgeInsets.all(24),
        itemBuilder: (context, index) {
          final quest = quests[index];
          final progress = store.playerState.quests[quest.id];
          return QuestCard(
            summary: quest,
            progress: progress,
            tribe: tribe,
            onTap: () => Navigator.pushNamed(
              context,
              AppRoutes.reinmakerQuestRunner,
              arguments: QuestRunnerArgs(questId: quest.id),
            ),
          );
        },
        separatorBuilder: (_, __) => const SizedBox(height: 20),
        itemCount: quests.length,
      ),
    );
  }
}




















