import 'package:flutter/material.dart';

import 'app.dart';
import 'config/app_environment.dart';
import 'reinmaker/reinmaker_boot.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final environment = AppEnvironment.fromPlatform();
  final reinmakerBootstrap = await ReinmakerBoot.initialize(environment: environment);

  runApp(
    CuriousKelllyApp(
      environment: environment,
      reinmakerBootstrap: reinmakerBootstrap,
    ),
  );
}



