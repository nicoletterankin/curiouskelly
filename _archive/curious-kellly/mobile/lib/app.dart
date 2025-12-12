import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'config/app_environment.dart';
import 'controllers/voice_controller.dart';
import 'reinmaker/reinmaker_boot.dart';
import 'reinmaker/state/reinmaker_store.dart';
import 'router/app_router.dart';

class CuriousKelllyApp extends StatefulWidget {
  const CuriousKelllyApp({
    super.key,
    required this.environment,
    required this.reinmakerBootstrap,
  });

  final AppEnvironment environment;
  final ReinmakerBootstrap reinmakerBootstrap;

  @override
  State<CuriousKelllyApp> createState() => _CuriousKelllyAppState();
}

class _CuriousKelllyAppState extends State<CuriousKelllyApp> {
  late final AppRouter _router = AppRouter(widget.environment);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<AppEnvironment>.value(value: widget.environment),
        ChangeNotifierProvider<VoiceController>(
          create: (_) => VoiceController(
            apiKey: widget.environment.openAiApiKey,
            backendUrl: widget.environment.backendBaseUrl,
          ),
        ),
        Provider<ReinmakerBootstrap>.value(value: widget.reinmakerBootstrap),
        ChangeNotifierProvider<ReinmakerStore>.value(
          value: widget.reinmakerBootstrap.store,
        ),
      ],
      child: MaterialApp(
        title: 'Curious Kellly',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
          useMaterial3: true,
          fontFamily: 'Poppins',
        ),
        onGenerateRoute: _router.onGenerateRoute,
        initialRoute: widget.reinmakerBootstrap.featureEnabled
            ? AppRoutes.reinmakerHall
            : AppRoutes.conversation,
      ),
    );
  }
}



