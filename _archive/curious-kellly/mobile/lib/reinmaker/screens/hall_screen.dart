import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../router/app_router.dart';
import '../models/tribe_pack.dart';
import '../state/reinmaker_store.dart';
import '../widgets/lens_badge.dart';
import '../widgets/reinmaker_tile.dart';
import '../widgets/stone_ring.dart';

class HallScreen extends StatelessWidget {
  const HallScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final store = context.watch<ReinmakerStore>();
    final manifest = store.manifest;

    if (!store.isInitialized || manifest == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final tribes = manifest.tribes;
    final featured = store.featuredTribe;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Hall of the Seven Tribes'),
        actions: [
          IconButton(
            tooltip: 'Knowledge Forge',
            icon: const Icon(Icons.handyman_outlined),
            onPressed: () => Navigator.pushNamed(context, AppRoutes.reinmakerForge),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () => store.refreshFromNetwork(),
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            if (store.isOffline)
              _OfflineBanner(onRetry: () => store.refreshFromNetwork()),
            _HeaderRow(xp: store.xp, featuredTribe: featured, onRotate: () {
              store.rotateFeaturedTribe(DateTime.now());
            }),
            const SizedBox(height: 16),
            ReinmakerTile(
              onTap: () {
                final pack = manifest.packForTribe(
                  TribeId.fromDisplayName(featured),
                );
                if (pack != null && pack.tiers.isNotEmpty) {
                  Navigator.pushNamed(
                    context,
                    AppRoutes.reinmakerTribeRoom,
                    arguments: TribeRoomArgs(tribeId: pack.tribe),
                  );
                }
              },
              title: 'Visit the $featured Tribe',
              subtitle: 'Collect stones and unlock their lens gifts today.',
              highlight: true,
            ),
            const SizedBox(height: 24),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: tribes.length,
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                mainAxisSpacing: 20,
                crossAxisSpacing: 20,
                childAspectRatio: 1.05,
              ),
              itemBuilder: (context, index) {
                final tribe = tribes[index];
                final stones = store.playerState.stones[tribe.tribe.key] ?? const [];
                final lensLevel = store.lensLevel(tribe.lensId);
                final isFeatured = tribe.tribe.displayName == featured;

                return _TribeCard(
                  tribe: tribe,
                  stones: stones,
                  lensLevel: lensLevel,
                  isFeatured: isFeatured,
                  onTap: () => Navigator.pushNamed(
                    context,
                    AppRoutes.reinmakerTribeRoom,
                    arguments: TribeRoomArgs(tribeId: tribe.tribe),
                  ),
                );
              },
            ),
            const SizedBox(height: 32),
            FilledButton.icon(
              onPressed: () => Navigator.pushNamed(context, AppRoutes.reinmakerFinale),
              icon: const Icon(Icons.auto_fix_high),
              label: const Text('Finale · The Great Unbinding'),
            ),
          ],
        ),
      ),
    );
  }
}

class _HeaderRow extends StatelessWidget {
  const _HeaderRow({required this.xp, required this.featuredTribe, required this.onRotate});

  final int xp;
  final String featuredTribe;
  final VoidCallback onRotate;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Featured Tribe',
                style: Theme.of(context).textTheme.labelMedium,
              ),
              const SizedBox(height: 4),
              Text(
                featuredTribe,
                style: Theme.of(context)
                    .textTheme
                    .headlineSmall
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text('XP', style: Theme.of(context).textTheme.labelMedium),
            const SizedBox(height: 4),
            Text('$xp', style: Theme.of(context).textTheme.titleLarge),
          ],
        ),
        const SizedBox(width: 16),
        IconButton(
          tooltip: 'Rotate Featured Tribe',
          onPressed: onRotate,
          icon: const Icon(Icons.refresh),
        )
      ],
    );
  }
}

class _TribeCard extends StatelessWidget {
  const _TribeCard({
    required this.tribe,
    required this.stones,
    required this.lensLevel,
    required this.isFeatured,
    required this.onTap,
  });

  final TribePack tribe;
  final List<String> stones;
  final int lensLevel;
  final bool isFeatured;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cardColor = _colorFromHex(tribe.color);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: isFeatured ? cardColor : Colors.white12,
          width: isFeatured ? 3 : 1,
        ),
        boxShadow: [
          if (isFeatured)
            BoxShadow(
              color: cardColor.withOpacity(0.35),
              blurRadius: 18,
            ),
        ],
      ),
      child: Material(
        color: Colors.black.withOpacity(0.4),
        borderRadius: BorderRadius.circular(24),
        child: InkWell(
          borderRadius: BorderRadius.circular(24),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Text(
                  tribe.tribe.displayName,
                  style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                StoneRing(tribe: tribe, playerStones: stones, size: 88),
                const SizedBox(height: 12),
                LensBadge(lensId: tribe.lensId, level: lensLevel > 0 ? lensLevel : 0),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _colorFromHex(String hex) {
    final buffer = StringBuffer();
    if (hex.length == 6 || hex.length == 7) buffer.write('ff');
    buffer.write(hex.replaceFirst('#', ''));
    return Color(int.parse(buffer.toString(), radix: 16));
  }
}

class _OfflineBanner extends StatelessWidget {
  const _OfflineBanner({required this.onRetry});

  final Future<void> Function() onRetry;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.orange.withOpacity(0.2),
      margin: const EdgeInsets.only(bottom: 16),
      child: ListTile(
        leading: const Icon(Icons.wifi_off, color: Colors.orange),
        title: const Text('Offline mode'),
        subtitle: const Text('Using cached quests. Pull to refresh when back online.'),
        trailing: TextButton(
          onPressed: () => onRetry(),
          child: const Text('Retry'),
        ),
      ),
    );
  }
}

