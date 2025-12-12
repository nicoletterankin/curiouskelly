import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../flutter_unity_bridge.dart';
import '../../router/app_router.dart';
import '../models/quest.dart';
import '../state/reinmaker_store.dart';

class QuestRunnerScreen extends StatefulWidget {
  const QuestRunnerScreen({super.key, this.questId});

  final String? questId;

  @override
  State<QuestRunnerScreen> createState() => _QuestRunnerScreenState();
}

class _QuestRunnerScreenState extends State<QuestRunnerScreen> {
  late Future<QuestModel> _questFuture;
  FlutterUnityBridge? _bridge;
  int _currentStep = 0;
  double _earnedScore = 0;
  int _scoreSlots = 0;
  Timer? _timer;
  int _elapsedSeconds = 0;
  bool _completed = false;
  bool _failed = false;

  @override
  void initState() {
    super.initState();
    final store = context.read<ReinmakerStore>();
    final questId = widget.questId ?? store.manifest?.quests.first.id;
    if (questId == null) {
      _questFuture = Future.error('Quest id missing');
    } else {
      _questFuture = store.loadQuest(questId);
    }
    _startTimer();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      setState(() => _elapsedSeconds += 1);
    });
  }

  Future<void> _completeQuest(QuestModel quest, double finalScore) async {
    if (_completed) return;
    _completed = true;
    _timer?.cancel();

    final store = context.read<ReinmakerStore>();
    await store.onQuestComplete(quest.id, finalScore);

    if (!mounted) return;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black87,
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.auto_awesome, color: Colors.amber),
                  const SizedBox(width: 12),
                  Text(
                    'Quest Complete!',
                    style: Theme.of(context)
                        .textTheme
                        .headlineSmall
                        ?.copyWith(fontWeight: FontWeight.bold),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text('Score ${(finalScore * 100).toStringAsFixed(0)}% · Time ${_elapsedSeconds}s'),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: () {
                  Navigator.pop(context);
                  Navigator.pop(context);
                },
                icon: const Icon(Icons.check),
                label: const Text('Return to Tribe Room'),
              ),
            ],
          ),
        );
      },
    );
  }

  void _failQuest(QuestModel quest) {
    if (_failed || _completed) return;
    _failed = true;
    _timer?.cancel();
    if (!mounted) return;

    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Time ran out'),
        content: const Text('Take a breather and try the quest again soon.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Stay'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
            },
            child: const Text('Leave'),
          ),
        ],
      ),
    );
  }

  void _onStepComplete(QuestModel quest, double scoreDelta) {
    _earnedScore += scoreDelta;
    _scoreSlots += 1;
    if (_currentStep + 1 >= quest.steps.length) {
      final finalScore = _scoreSlots == 0 ? 1.0 : (_earnedScore / _scoreSlots);
      final minScore = (quest.successCriteria['scoreMin'] as num?)?.toDouble() ?? 0.6;
      if (finalScore >= minScore) {
        _completeQuest(quest, finalScore);
      } else {
        _failQuest(quest);
      }
    } else {
      setState(() => _currentStep += 1);
    }
  }

  @override
  Widget build(BuildContext context) {
    final store = context.watch<ReinmakerStore>();
    final iconOnly = store.iconOnlyMode;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Quest Runner'),
        actions: [
          IconButton(
            icon: const Icon(Icons.flag),
            tooltip: 'Hall',
            onPressed: () => Navigator.pushNamedAndRemoveUntil(
              context,
              AppRoutes.reinmakerHall,
              (route) => route.isFirst,
            ),
          ),
        ],
      ),
      body: FutureBuilder<QuestModel>(
        future: _questFuture,
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            if (snapshot.hasError) {
              return Center(child: Text('Failed to load quest: ${snapshot.error}'));
            }
            return const Center(child: CircularProgressIndicator());
          }

          final quest = snapshot.data!;
          final step = quest.steps[_currentStep];

          final maxSeconds = (quest.successCriteria['timeMaxSec'] as num?)?.toInt();
          if (maxSeconds != null && _elapsedSeconds > maxSeconds && !_failed && !_completed) {
            _failQuest(quest);
          }

          return Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (!iconOnly)
                  SizedBox(
                    height: 240,
                    child: KellyAvatarWidget(
                      learnerAge: _ageFromBucket(store.playerState.ageBucket),
                      onBridgeReady: (bridge) {
                        _bridge = bridge;
                        if (step.type == QuestStepType.dialogue) {
                          _speakStep(step);
                        }
                      },
                    ),
                  )
                else
                  Container(
                    height: 180,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      color: Colors.black26,
                    ),
                    child: const Center(
                      child: Icon(Icons.auto_fix_high, size: 64, color: Colors.white70),
                    ),
                  ),
                const SizedBox(height: 24),
                _QuestHeader(
                  quest: quest,
                  currentStep: _currentStep + 1,
                  totalSteps: quest.steps.length,
                  elapsedSeconds: _elapsedSeconds,
                ),
                const SizedBox(height: 20),
                Expanded(
                  child: _buildStepBody(context, quest, step, iconOnly),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildStepBody(BuildContext context, QuestModel quest, QuestStep step, bool iconOnly) {
    switch (step.type) {
      case QuestStepType.dialogue:
        return _DialogueStepView(
          data: step.data,
          onContinue: () => _onStepComplete(quest, 1.0),
          captionsEnabled: context.watch<ReinmakerStore>().captionsEnabled,
          onSpeak: () => _speakStep(step),
        );
      case QuestStepType.puzzle:
        return _PuzzleStepView(
          data: step.data,
          onSolved: () => _onStepComplete(quest, 1.0),
        );
      case QuestStepType.builder:
        return _BuilderStepView(
          data: step.data,
          onSubmit: () => _onStepComplete(quest, 1.0),
        );
      case QuestStepType.empathy:
        return _EmpathyStepView(
          data: step.data,
          onResult: (score) => _onStepComplete(quest, score),
        );
    }
  }

  void _speakStep(QuestStep step) {
    final bridge = _bridge;
    if (bridge == null) return;
    final text = step.data['text'] as String? ?? step.data['cue'] as String? ?? 'Greetings adventurer';
    bridge.speak(text, _ageFromBucket(context.read<ReinmakerStore>().playerState.ageBucket));
  }

  int _ageFromBucket(String? bucket) {
    switch (bucket) {
      case '2-5':
        return 4;
      case '6-12':
        return 9;
      case '13-17':
        return 15;
      case '18-35':
        return 26;
      case '36-60':
        return 45;
      case '61-102':
        return 70;
      default:
        return 12;
    }
  }
}

class _QuestHeader extends StatelessWidget {
  const _QuestHeader({
    required this.quest,
    required this.currentStep,
    required this.totalSteps,
    required this.elapsedSeconds,
  });

  final QuestModel quest;
  final int currentStep;
  final int totalSteps;
  final int elapsedSeconds;

  @override
  Widget build(BuildContext context) {
    final minutes = elapsedSeconds ~/ 60;
    final seconds = elapsedSeconds % 60;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          quest.id,
          style: Theme.of(context)
              .textTheme
              .titleLarge
              ?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 6),
        Row(
          children: [
            Text('Step $currentStep of $totalSteps'),
            const Spacer(),
            Text('${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}'),
          ],
        ),
      ],
    );
  }
}

class _DialogueStepView extends StatelessWidget {
  const _DialogueStepView({
    required this.data,
    required this.onContinue,
    required this.captionsEnabled,
    required this.onSpeak,
  });

  final Map<String, dynamic> data;
  final VoidCallback onContinue;
  final bool captionsEnabled;
  final VoidCallback onSpeak;

  @override
  Widget build(BuildContext context) {
    final text = data['text'] as String? ?? 'Share what you notice in this scene.';
    final npc = data['npc'] as String? ?? 'Guide';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$npc says:', style: Theme.of(context).textTheme.labelLarge),
        const SizedBox(height: 12),
        if (captionsEnabled)
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.black45,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(text, style: Theme.of(context).textTheme.bodyLarge),
          ),
        const Spacer(),
        Row(
          children: [
            TextButton.icon(
              onPressed: onSpeak,
              icon: const Icon(Icons.volume_up),
              label: const Text('Replay'),
            ),
            const Spacer(),
            ElevatedButton(
              onPressed: onContinue,
              child: const Text('Continue'),
            ),
          ],
        ),
      ],
    );
  }
}

class _PuzzleStepView extends StatelessWidget {
  const _PuzzleStepView({required this.data, required this.onSolved});

  final Map<String, dynamic> data;
  final VoidCallback onSolved;

  @override
  Widget build(BuildContext context) {
    final goal = data['goal'] as String? ?? 'Complete the puzzle';
    final spec = data['spec'] as String? ?? 'grid_select';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Puzzle · $spec', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 12),
        Text(goal, style: Theme.of(context).textTheme.bodyLarge),
        const Spacer(),
        FilledButton.icon(
          onPressed: onSolved,
          icon: const Icon(Icons.check_circle_outline),
          label: const Text('Mark as solved'),
        ),
      ],
    );
  }
}

class _BuilderStepView extends StatelessWidget {
  const _BuilderStepView({required this.data, required this.onSubmit});

  final Map<String, dynamic> data;
  final VoidCallback onSubmit;

  @override
  Widget build(BuildContext context) {
    final goal = data['goal'] as String? ?? 'Arrange the UI blocks to meet the brief.';
    final template = data['template'] as String? ?? 'wireframe_a';
    final palette = (data['palette'] as List<dynamic>? ?? const [])
        .map((entry) => entry.toString())
        .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Builder · $template', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 12),
        Text(goal, style: Theme.of(context).textTheme.bodyLarge),
        const SizedBox(height: 12),
        if (palette.isNotEmpty)
          Wrap(
            spacing: 8,
            children: palette
                .map((item) => Chip(
                      label: Text(item),
                      backgroundColor: Colors.blueGrey.withOpacity(0.3),
                    ))
                .toList(),
          ),
        const Spacer(),
        FilledButton.icon(
          onPressed: onSubmit,
          icon: const Icon(Icons.rocket_launch),
          label: const Text('Submit build'),
        ),
      ],
    );
  }
}

class _EmpathyStepView extends StatefulWidget {
  const _EmpathyStepView({required this.data, required this.onResult});

  final Map<String, dynamic> data;
  final void Function(double score) onResult;

  @override
  State<_EmpathyStepView> createState() => _EmpathyStepViewState();
}

class _EmpathyStepViewState extends State<_EmpathyStepView> {
  int _branchIndex = 0;
  double _score = 0;

  List<dynamic> get branches => widget.data['branches'] as List<dynamic>? ?? const [];

  @override
  Widget build(BuildContext context) {
    if (_branchIndex >= branches.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        widget.onResult(branches.isEmpty ? 1.0 : _score / branches.length);
      });
      return const Center(child: Text('Great reflection!'));
    }

    final branch = branches[_branchIndex] as Map<String, dynamic>;
    final responses = (branch['responses'] as List<dynamic>? ?? const [])
        .map((res) => res as Map<String, dynamic>)
        .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          branch['prompt'] as String? ?? 'How would you respond?',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 16),
        ...responses.map(
          (response) => Card(
            child: ListTile(
              title: Text(response['text'] as String? ?? ''),
              trailing: const Icon(Icons.navigate_next),
              onTap: () {
                final impact = response['impact'] as String? ?? 'neutral';
                if (impact == 'positive') {
                  _score += 1.0;
                } else if (impact == 'neutral') {
                  _score += 0.5;
                } else {
                  _score += 0.2;
                }
                final next = response['next'] as int?;
                setState(() {
                  _branchIndex = next ?? _branchIndex + 1;
                });
              },
            ),
          ),
        ),
      ],
    );
  }
}

