import 'package:flutter/foundation.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../config/app_environment.dart';
import 'models/player_state.dart';
import 'state/reinmaker_store.dart';

void registerReinmakerAdapters() {
  if (!Hive.isAdapterRegistered(90)) {
    Hive.registerAdapter(PlayerStateAdapter());
  }
  if (!Hive.isAdapterRegistered(91)) {
    Hive.registerAdapter(LensProgressAdapter());
  }
  if (!Hive.isAdapterRegistered(92)) {
    Hive.registerAdapter(QuestProgressAdapter());
  }
  if (!Hive.isAdapterRegistered(93)) {
    Hive.registerAdapter(ReinmakerSettingsAdapter());
  }
  if (!Hive.isAdapterRegistered(94)) {
    Hive.registerAdapter(ActiveCosmeticAdapter());
  }
}

class ReinmakerBootstrap {
  ReinmakerBootstrap({
    required this.featureEnabled,
    required this.store,
    required this.playerBox,
    required this.cacheBox,
  });

  final bool featureEnabled;
  final ReinmakerStore store;
  final Box<PlayerState> playerBox;
  final Box<dynamic> cacheBox;
}

class ReinmakerBoot {
  static Future<ReinmakerBootstrap> initialize({
    required AppEnvironment environment,
  }) async {
    await Hive.initFlutter();
    registerReinmakerAdapters();

    final prefs = await SharedPreferences.getInstance();
    final featureEnabled = prefs.getBool(_featureFlagKey) ?? true;

    final playerBox = await Hive.openBox<PlayerState>('reinmaker_player');
    final cacheBox = await Hive.openBox('reinmaker_cache');

    final store = ReinmakerStore(
      environment: environment,
      preferences: prefs,
      playerBox: playerBox,
      cacheBox: cacheBox,
      featureEnabled: featureEnabled,
    );

    await store.bootstrap();

    return ReinmakerBootstrap(
      featureEnabled: featureEnabled,
      store: store,
      playerBox: playerBox,
      cacheBox: cacheBox,
    );
  }

}

const String _featureFlagKey = 'features.reinmaker_game';

