import 'package:flutter/material.dart';

import '../config/app_environment.dart';
import '../screens/conversation_screen.dart';
import '../reinmaker/models/tribe_pack.dart';
import '../reinmaker/screens/finale_unbinding_screen.dart';
import '../reinmaker/screens/forge_screen.dart';
import '../reinmaker/screens/hall_screen.dart';
import '../reinmaker/screens/quest_runner_screen.dart';
import '../reinmaker/screens/tribe_room_screen.dart';

/// Central routing table for the Curious Kellly mobile app.
class AppRouter {
  AppRouter(this._environment);

  final AppEnvironment _environment;

  Route<dynamic>? onGenerateRoute(RouteSettings settings) {
    switch (settings.name) {
      case AppRoutes.root:
      case AppRoutes.conversation:
        final args = settings.arguments;
        final learnerAge = args is ConversationRouteArgs
            ? args.learnerAge
            : _environment.defaultLearnerAge;

        return MaterialPageRoute(
          builder: (_) => ConversationScreen(learnerAge: learnerAge),
          settings: RouteSettings(
            name: settings.name ?? AppRoutes.conversation,
            arguments: args,
          ),
        );

      case AppRoutes.reinmakerHall:
        return MaterialPageRoute(
          builder: (context) => const HallScreen(),
          settings: settings,
        );

      case AppRoutes.reinmakerTribeRoom:
        final args = settings.arguments as TribeRoomArgs?;
        return MaterialPageRoute(
          builder: (_) => TribeRoomScreen(tribeId: args?.tribeId),
          settings: settings,
        );

      case AppRoutes.reinmakerQuestRunner:
        final args = settings.arguments as QuestRunnerArgs?;
        return MaterialPageRoute(
          builder: (_) => QuestRunnerScreen(questId: args?.questId),
          settings: settings,
        );

      case AppRoutes.reinmakerForge:
        return MaterialPageRoute(
          builder: (_) => const ForgeScreen(),
          settings: settings,
        );

      case AppRoutes.reinmakerFinale:
        return MaterialPageRoute(
          builder: (_) => const FinaleUnbindingScreen(),
          settings: settings,
        );

      case AppRoutes.settings:
        // Settings screen will be implemented in a subsequent iteration.
        return _unsupportedRoute(settings);

      default:
        return _unsupportedRoute(settings);
    }
  }

  Route<dynamic> _unsupportedRoute(RouteSettings settings) {
    return MaterialPageRoute(
      builder: (_) => Scaffold(
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.route, size: 48),
                const SizedBox(height: 16),
                Text(
                  'Route "${settings.name ?? 'unknown'}" is not implemented yet.',
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 18),
                ),
              ],
            ),
          ),
        ),
      ),
      settings: settings,
    );
  }
}

/// Route name constants used throughout the app.
abstract class AppRoutes {
  static const String root = '/';
  static const String conversation = '/conversation';
  static const String settings = '/settings';
  static const String reinmakerHall = '/reinmaker/hall';
  static const String reinmakerTribeRoom = '/reinmaker/tribe';
  static const String reinmakerQuestRunner = '/reinmaker/quest';
  static const String reinmakerForge = '/reinmaker/forge';
  static const String reinmakerFinale = '/reinmaker/finale';
}

class TribeRoomArgs {
  const TribeRoomArgs({required this.tribeId});

  final TribeId tribeId;
}

class QuestRunnerArgs {
  const QuestRunnerArgs({required this.questId});

  final String questId;
}

/// Optional arguments for navigating to the conversation route.
class ConversationRouteArgs {
  const ConversationRouteArgs({required this.learnerAge});

  final int learnerAge;
}

