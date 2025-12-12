import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/lens.dart';
import '../state/reinmaker_store.dart';
import '../widgets/lens_badge.dart';

class ForgeScreen extends StatelessWidget {
  const ForgeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final store = context.watch<ReinmakerStore>();
    final manifest = store.manifest;

    return Scaffold(
      appBar: AppBar(title: const Text('Knowledge Forge')),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Text('Session Settings', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),
          SwitchListTile.adaptive(
            title: const Text('Captions enabled'),
            subtitle: const Text('Show on-screen captions for dialogue cues'),
            value: store.captionsEnabled,
            onChanged: store.setCaptionsEnabled,
          ),
          SwitchListTile.adaptive(
            title: const Text('Icon-only mode'),
            subtitle: const Text('Disable real-time avatar view to conserve battery'),
            value: store.iconOnlyMode,
            onChanged: store.setIconOnlyMode,
          ),
          SwitchListTile.adaptive(
            title: const Text('High contrast'),
            subtitle: const Text('Increase contrast for improved readability'),
            value: store.highContrast,
            onChanged: store.setHighContrast,
          ),
          ListTile(
            title: const Text('Text scale'),
            subtitle: Slider(
              min: 0.8,
              max: 1.6,
              divisions: 8,
              value: store.textScale,
              onChanged: store.setTextScale,
              label: store.textScale.toStringAsFixed(1),
            ),
          ),
          const Divider(height: 32),
          Text('Lens Progress', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: LensId.values
                .map(
                  (lens) => LensBadge(
                    lensId: lens,
                    level: store.lensLevel(lens),
                  ),
                )
                .toList(),
          ),
          const Divider(height: 32),
          Text('Collected Stones', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),
          if (manifest != null)
            ...manifest.tribes.map(
              (tribe) {
                final stones = store.playerState.stones[tribe.tribe.key] ?? const [];
                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: Colors.black54,
                    child: Text(tribe.tribe.displayName.substring(0, 1)),
                  ),
                  title: Text(tribe.tribe.displayName),
                  subtitle: Text(
                    stones.isEmpty
                        ? 'No stones yet'
                        : stones.join(', '),
                  ),
                );
              },
            ),
        ],
      ),
    );
  }
}




















