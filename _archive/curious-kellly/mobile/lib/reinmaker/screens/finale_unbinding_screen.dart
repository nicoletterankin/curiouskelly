import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/tribe_pack.dart';
import '../state/reinmaker_store.dart';

class FinaleUnbindingScreen extends StatelessWidget {
  const FinaleUnbindingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final store = context.watch<ReinmakerStore>();
    final manifest = store.manifest;
    final masteryComplete = _hasAllMastery(store);

    return Scaffold(
      appBar: AppBar(title: const Text('The Great Unbinding')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              masteryComplete
                  ? 'The path is open.'
                  : 'Collect all mastery stones to unlock the finale.',
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Text(
              'When all seven tribes reach mastery, the forge reveals the final ritual that unbinds Rein from the shadows.',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 24),
            Expanded(
              child: manifest == null
                  ? const Center(child: CircularProgressIndicator())
                  : ListView(
                      children: manifest.tribes.map((tribe) {
                        final hasMastery = store
                                .playerState
                                .stones[tribe.tribe.key]
                                ?.contains('${tribe.tribe.key}.mastery') ??
                            false;
                        return ListTile(
                          leading: Icon(
                            hasMastery ? Icons.star : Icons.radio_button_unchecked,
                            color: hasMastery ? Colors.amber : Colors.white54,
                          ),
                          title: Text(tribe.tribe.displayName),
                          subtitle: Text(
                            hasMastery ? 'Mastery secured' : 'Mastery stone pending',
                          ),
                        );
                      }).toList(),
                    ),
            ),
            if (masteryComplete)
              FilledButton.icon(
                onPressed: () {
                  showDialog<void>(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('Finale Incoming'),
                      content: const Text(
                        'Prepare your tribe. The finale sequence will be delivered in the next build.',
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Ready'),
                        )
                      ],
                    ),
                  );
                },
                icon: const Icon(Icons.auto_awesome),
                label: const Text('Commence The Great Unbinding'),
              )
            else
              Text(
                'Earn the remaining mastery stones by completing Tier 3 quests for each tribe.',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
          ],
        ),
      ),
    );
  }

  bool _hasAllMastery(ReinmakerStore store) {
    for (final tribe in TribeId.values) {
      final stones = store.playerState.stones[tribe.key] ?? const [];
      if (!stones.contains('${tribe.key}.mastery')) {
        return false;
      }
    }
    return true;
  }
}

