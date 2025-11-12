import 'dart:io';

import 'package:curious_kellly/config/app_environment.dart';
import 'package:curious_kellly/reinmaker/reinmaker_boot.dart';
import 'package:curious_kellly/reinmaker/state/reinmaker_store.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hive/hive.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:curious_kellly/reinmaker/models/player_state.dart';

void main() {
  late Directory tempDir;
  late Box<PlayerState> playerBox;
  late Box<dynamic> cacheBox;
  late SharedPreferences prefs;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('quest_award_test');
    Hive.init(tempDir.path);
    registerReinmakerAdapters();
    playerBox = await Hive.openBox<PlayerState>('reinmaker_player');
    cacheBox = await Hive.openBox('reinmaker_cache');
    SharedPreferences.setMockInitialValues({});
    prefs = await SharedPreferences.getInstance();
  });

  tearDown(() async {
    await playerBox.clear();
    await cacheBox.clear();
    await playerBox.close();
    await cacheBox.close();
    await Hive.deleteBoxFromDisk('reinmaker_player');
    await Hive.deleteBoxFromDisk('reinmaker_cache');
    await tempDir.delete(recursive: true);
  });

  test('quest completion grants xp and mastery gates finale', () async {
    final store = ReinmakerStore(
      environment: AppEnvironment(
        backendBaseUrl: 'http://localhost:3000',
        openAiApiKey: '',
        defaultLearnerAge: 12,
      ),
      preferences: prefs,
      playerBox: playerBox,
      cacheBox: cacheBox,
      featureEnabled: true,
    );

    await store.bootstrap();

    final initialXp = store.xp;
    await store.onQuestComplete('q.light.001', 0.9);

    expect(store.xp, greaterThan(initialXp));
    expect(
      store.playerState.stones['light']?.contains('light.spark'),
      isTrue,
    );

    await store.onQuestComplete('q.light.001', 0.95);

    // stone should still be unique
    expect(store.playerState.stones['light']?.length, equals(1));

    store.dispose();
  });
}

