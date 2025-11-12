import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hive/hive.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:curious_kellly/config/app_environment.dart';
import 'package:curious_kellly/reinmaker/models/lens.dart';
import 'package:curious_kellly/reinmaker/models/player_state.dart';
import 'package:curious_kellly/reinmaker/models/tribe_pack.dart';
import 'package:curious_kellly/reinmaker/reinmaker_boot.dart';
import 'package:curious_kellly/reinmaker/state/reinmaker_store.dart';

void main() {
  late Directory tempDir;
  late Box<PlayerState> playerBox;
  late Box<dynamic> cacheBox;
  late SharedPreferences prefs;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('reinmaker_store_test');
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

  test('awards stones idempotently and unlocks lens', () async {
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

    store.awardStone(TribeId.light, 1, 'light.spark');
    store.awardStone(TribeId.light, 1, 'light.spark');

    final stones = store.playerState.stones['light'] ?? [];
    expect(stones.length, 1);
    expect(stones.first, 'light.spark');

    final lensLevel = store.lensLevel(LensId.uiComposition);
    expect(lensLevel, equals(0));

    store.unlockLens(LensId.uiComposition, 1);
    expect(store.lensLevel(LensId.uiComposition), equals(1));

    store.dispose();
  });
}

