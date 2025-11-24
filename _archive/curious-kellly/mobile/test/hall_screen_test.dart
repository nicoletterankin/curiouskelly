import 'dart:io';

import 'package:curious_kellly/config/app_environment.dart';
import 'package:curious_kellly/reinmaker/reinmaker_boot.dart';
import 'package:curious_kellly/reinmaker/screens/hall_screen.dart';
import 'package:curious_kellly/reinmaker/state/reinmaker_store.dart';
import 'package:curious_kellly/reinmaker/widgets/stone_ring.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hive/hive.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:curious_kellly/reinmaker/models/player_state.dart';

void main() {
  late Directory tempDir;
  late Box<PlayerState> playerBox;
  late Box<dynamic> cacheBox;
  late SharedPreferences prefs;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('hall_screen_test');
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

  testWidgets('Hall screen shows seven tribe doors and featured tile', (tester) async {
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

    await tester.pumpWidget(
      ChangeNotifierProvider<ReinmakerStore>.value(
        value: store,
        child: const MaterialApp(home: HallScreen()),
      ),
    );

    await tester.pumpAndSettle();

    expect(find.text('Hall of the Seven Tribes'), findsOneWidget);
    expect(find.byType(StoneRing), findsNWidgets(7));
    expect(find.textContaining('Visit the'), findsOneWidget);

    store.dispose();
  });
}

