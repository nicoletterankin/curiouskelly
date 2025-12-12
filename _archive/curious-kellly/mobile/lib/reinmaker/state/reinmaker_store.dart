import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:flutter/foundation.dart';
import 'package:hive/hive.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../../config/app_environment.dart';
import '../data/default_manifest.dart';
import '../models/lens.dart';
import '../models/player_state.dart';
import '../models/quest.dart';
import '../models/tribe_pack.dart';

const _playerBoxKey = 'player_state';
const _manifestCacheKey = 'manifest_json';
const _questCachePrefix = 'quest:';
const _featureFlagKey = 'features.reinmaker_game';
const _rotationPrefKey = 'reinmaker.lastFeaturedAt';
const _defaultRotation = ['Light', 'Stone', 'Metal', 'Code', 'Air', 'Water', 'Fire'];

String _stoneSuffixForTier(int tier) {
  switch (tier) {
    case 1:
      return 'spark';
    case 2:
      return 'craft';
    case 3:
      return 'mastery';
    default:
      return 'spark';
  }
}

class ReinmakerStore extends ChangeNotifier {
  ReinmakerStore({
    required this.environment,
    required this.preferences,
    required this.playerBox,
    required this.cacheBox,
    required bool featureEnabled,
  })  : _featureEnabled = featureEnabled,
        _httpClient = http.Client();

  final AppEnvironment environment;
  final SharedPreferences preferences;
  final Box<PlayerState> playerBox;
  final Box<dynamic> cacheBox;
  final http.Client _httpClient;

  bool _featureEnabled;
  bool _initialized = false;
  bool _isOffline = false;

  late PlayerState _playerState;
  ReinmakerManifest? _manifest;
  final Map<String, QuestModel> _questCache = <String, QuestModel>{};

  Timer? _saveTimer;
  final Duration _saveDebounce = const Duration(milliseconds: 350);

  bool get isEnabled => _featureEnabled;
  bool get isInitialized => _initialized;
  bool get isOffline => _isOffline;
  PlayerState get playerState => _playerState;
  ReinmakerManifest? get manifest => _manifest;

  String get featuredTribe => _playerState.featuredTribe ?? _defaultRotation.first;

  int get xp => _playerState.xp;

  Future<void> bootstrap() async {
    _playerState = playerBox.get(_playerBoxKey) ?? PlayerState.initial();
    if (!playerBox.containsKey(_playerBoxKey)) {
      await playerBox.put(_playerBoxKey, _playerState);
    }

    final lastRotationIso = preferences.getString(_rotationPrefKey);
    if (lastRotationIso != null) {
      _playerState.lastFeaturedRotation = DateTime.tryParse(lastRotationIso);
    }

    _loadDefaultContent();
    await _loadCachedManifest();

    if (_featureEnabled) {
      await _syncManifest();
    }

    _initialized = true;
    notifyListeners();
  }

  Future<void> refreshFromNetwork() async {
    if (!_featureEnabled) return;
    await _syncManifest(force: true);
    notifyListeners();
  }

  void setFeatureEnabled(bool enabled) {
    _featureEnabled = enabled;
    preferences.setBool(_featureFlagKey, enabled);
    notifyListeners();
  }

  Future<ReinmakerManifest?> _syncManifest({bool force = false}) async {
    final uri = Uri.parse('${environment.backendBaseUrl}/api/reinmaker/manifest');

    try {
      final response = await _httpClient
          .get(uri)
          .timeout(const Duration(seconds: 8));

      if (response.statusCode != 200) {
        _isOffline = true;
        return _manifest;
      }

      final payload = jsonDecode(response.body) as Map<String, dynamic>;
      final manifestJson = payload['manifest'] as Map<String, dynamic>?;
      if (manifestJson == null) {
        _isOffline = true;
        return _manifest;
      }

      final manifest = ReinmakerManifest.fromJson(manifestJson);
      _manifest = manifest;
      _isOffline = false;

      await cacheBox.put(_manifestCacheKey, jsonEncode(manifestJson));

      for (final quest in manifest.quests) {
        final cached = _questCache[quest.id];
        if (cached != null) continue;
        await _fetchQuestFromNetwork(quest.id);
      }

      return manifest;
    } catch (error) {
      _isOffline = true;
      return _manifest;
    }
  }

  void rotateFeaturedTribe(DateTime now) {
    final rotation = _manifest?.featuredRotation ?? _defaultRotation;
    final current = featuredTribe;
    final last = _playerState.lastFeaturedRotation;

    if (last != null && now.difference(last).inMinutes < 60) {
      return;
    }

    final currentIndex = rotation.indexWhere(
      (value) => value.toLowerCase() == current.toLowerCase(),
    );

    final nextIndex = (currentIndex >= 0 ? currentIndex + 1 : 0) % rotation.length;
    final next = rotation[nextIndex];

    _playerState.featuredTribe = next;
    _playerState.lastFeaturedRotation = now;
    preferences.setString(_rotationPrefKey, now.toIso8601String());
    _logEvent('rmk_featured_hour', {'tribe': next});
    _scheduleSave();
    notifyListeners();
  }

  Future<void> onQuestComplete(String questId, double score) async {
    final manifestSummary = _manifest?.questById(questId);
    final quest = await loadQuest(questId);

    if (manifestSummary == null) {
      throw StateError('Quest $questId is not part of the active manifest');
    }

    final existing = _playerState.quests[questId];
    final progress = (existing ??
            QuestProgress(status: 'available', attempts: 0))
        .copyWith(
      status: 'completed',
      attempts: (existing?.attempts ?? 0) + 1,
      bestScore: existing?.bestScore != null
          ? max(existing!.bestScore!, score)
          : score,
      lastPlayedAt: DateTime.now(),
    );

    _playerState.quests[questId] = progress;

    _playerState.xp += manifestSummary.rewards.xp;
    _logEvent('rmk_complete_quest', {
      'quest_id': questId,
      'tribe': manifestSummary.tribe.displayName,
      'tier': manifestSummary.tier,
      'score': score,
    });

    for (final cosmetic in manifestSummary.rewards.cosmetics) {
      if (!_playerState.unlockedCosmetics.contains(cosmetic)) {
        _playerState.unlockedCosmetics.add(cosmetic);
      }
    }

    if (manifestSummary.rewards.stoneId != null) {
      awardStone(manifestSummary.tribe, manifestSummary.tier,
          manifestSummary.rewards.stoneId!);
    }

    final pack = _manifest?.packForTribe(manifestSummary.tribe);
    if (pack != null) {
      unlockLens(pack.lensId, manifestSummary.tier);
    }

    _scheduleSave();
    notifyListeners();
    await cacheQuest(quest);
  }

  void awardStone(TribeId tribe, int tier, [String? explicitStoneId]) {
    final stoneId = explicitStoneId ?? '${tribe.key}.${_stoneSuffixForTier(tier)}';
    final stonesForTribe =
        _playerState.stones[tribe.key] ?? <String>[];
    if (!stonesForTribe.contains(stoneId)) {
      stonesForTribe.add(stoneId);
      stonesForTribe.sort();
      _playerState.stones[tribe.key] = stonesForTribe;
      _logEvent('rmk_award_stone', {
        'tribe': tribe.displayName,
        'stone_id': stoneId,
      });
      _scheduleSave();
      notifyListeners();
    }
  }

  void unlockLens(LensId lensId, int levelDelta) {
    final key = lensId.value;
    final existing = _playerState.lenses[key];
    final now = DateTime.now();

    if (existing == null) {
      _playerState.lenses[key] =
          LensProgress(lensId: key, level: levelDelta.clamp(1, 9), unlockedAt: now);
    } else {
      final newLevel = (existing.level + levelDelta).clamp(1, 9);
      existing.level = newLevel;
      existing.unlockedAt ??= now;
      _playerState.lenses[key] = existing;
    }

    _logEvent('rmk_unlock_lens', {'lens_id': key, 'level_delta': levelDelta});
    _scheduleSave();
    notifyListeners();
  }

  Future<QuestModel> loadQuest(String questId, {bool forceRefresh = false}) async {
    if (!forceRefresh && _questCache.containsKey(questId)) {
      return _questCache[questId]!;
    }

    if (!forceRefresh) {
      final cachedJson = cacheBox.get('$_questCachePrefix$questId');
      if (cachedJson is String) {
        final decoded = jsonDecode(cachedJson) as Map<String, dynamic>;
        final quest = QuestModel.fromJson(decoded);
        _questCache[questId] = quest;
        return quest;
      }
    }

    if (_featureEnabled) {
      final quest = await _fetchQuestFromNetwork(questId);
      if (quest != null) {
        return quest;
      }
    }

    final fallback = defaultQuestLibrary[questId];
    if (fallback != null) {
      final quest = QuestModel.fromJson(fallback);
      _questCache[questId] = quest;
      return quest;
    }

    throw StateError('Quest $questId not found in cache or defaults.');
  }

  Future<void> cacheQuest(QuestModel quest) async {
    cacheBox.put('$_questCachePrefix${quest.id}', jsonEncode({
      'id': quest.id,
      'tribe': quest.tribe.displayName,
      'tier': quest.tier,
      'kind': quest.kind,
      'steps': quest.steps
          .map((step) => {'type': step.type.name, ...step.data})
          .toList(),
      'ageBuckets': quest.ageBuckets,
      'successCriteria': quest.successCriteria,
      'rewards': {
        'xp': quest.rewards.xp,
        'cosmetics': quest.rewards.cosmetics,
        'stoneId': quest.rewards.stoneId,
      },
      'lessonRef': quest.lessonRef,
      'localizationKey': quest.localizationKey,
      'estimatedDurationMin': quest.estimatedDurationMin,
      'captions': quest.captions,
      'audio': quest.audio,
    }));
  }

  bool hasStone(String stoneId) {
    for (final entry in _playerState.stones.values) {
      if (entry.contains(stoneId)) {
        return true;
      }
    }
    return false;
  }

  int lensLevel(LensId lensId) {
    return _playerState.lenses[lensId.value]?.level ?? 0;
  }

  bool get captionsEnabled => _playerState.settings.captionsEnabled;
  bool get iconOnlyMode => _playerState.settings.iconOnlyMode;
  bool get highContrast => _playerState.settings.highContrast;
  double get textScale => _playerState.settings.textScale;

  void setCaptionsEnabled(bool value) {
    _playerState.settings = _playerState.settings.copyWith(captionsEnabled: value);
    _scheduleSave();
    notifyListeners();
  }

  void setIconOnlyMode(bool value) {
    _playerState.settings = _playerState.settings.copyWith(iconOnlyMode: value);
    _logEvent('rmk_toggle_low_power', {'enabled': value});
    _scheduleSave();
    notifyListeners();
  }

  void setHighContrast(bool value) {
    _playerState.settings = _playerState.settings.copyWith(highContrast: value);
    _scheduleSave();
    notifyListeners();
  }

  void setTextScale(double value) {
    _playerState.settings = _playerState.settings.copyWith(textScale: value);
    _scheduleSave();
    notifyListeners();
  }

  Future<void> _loadCachedManifest() async {
    final cachedManifest = cacheBox.get(_manifestCacheKey);
    if (cachedManifest is String) {
      try {
        final decoded = jsonDecode(cachedManifest) as Map<String, dynamic>;
        _manifest = ReinmakerManifest.fromJson(decoded);
      } catch (_) {
        // ignore corrupt cache
      }
    }
  }

  void _loadDefaultContent() {
    _manifest = ReinmakerManifest.fromJson(defaultReinmakerManifest);
    defaultQuestLibrary.forEach((key, value) {
      _questCache[key] = QuestModel.fromJson(value);
    });
  }

  Future<QuestModel?> _fetchQuestFromNetwork(String questId) async {
    final uri = Uri.parse('${environment.backendBaseUrl}/api/reinmaker/quests/$questId');
    try {
      final response = await _httpClient
          .get(uri)
          .timeout(const Duration(seconds: 8));
      if (response.statusCode != 200) {
        return null;
      }

      final payload = jsonDecode(response.body) as Map<String, dynamic>;
      final questJson = payload['quest'] as Map<String, dynamic>?;
      if (questJson == null) {
        return null;
      }

      final quest = QuestModel.fromJson(questJson);
      _questCache[questId] = quest;
      await cacheQuest(quest);
      return quest;
    } catch (_) {
      return null;
    }
  }

  void _scheduleSave() {
    _saveTimer?.cancel();
    _saveTimer = Timer(_saveDebounce, _persistPlayerState);
  }

  Future<void> _persistPlayerState() async {
    await playerBox.put(_playerBoxKey, _playerState);
  }

  void _logEvent(String name, Map<String, Object?> params) {
    try {
      FirebaseAnalytics.instance.logEvent(name: name, parameters: params);
    } catch (_) {
      // no-op if Firebase not initialized
    }
  }

  @override
  void dispose() {
    _saveTimer?.cancel();
    _httpClient.close();
    super.dispose();
  }
}

