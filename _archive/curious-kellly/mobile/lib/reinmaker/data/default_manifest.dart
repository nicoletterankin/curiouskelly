import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import '../models/quest.dart';
import '../models/tribe_pack.dart';

const defaultReinmakerManifest = {
  'version': 'rmk.manifest.v1',
  'generatedAt': '2025-01-01T00:00:00.000Z',
  'featuredRotation': ['Light', 'Stone', 'Metal', 'Code', 'Air', 'Water', 'Fire'],
  'locales': {
    'en': {'path': 'locales/en.json', 'hash': '', 'keys': []},
    'es': {'path': 'locales/es.json', 'hash': '', 'keys': []},
    'fr': {'path': 'locales/fr.json', 'hash': '', 'keys': []},
  },
  'tribes': [
    {
      'id': 'tribe.light.v1',
      'tribe': 'Light',
      'lensId': 'lens.ui_composition',
      'color': '#FFE066',
      'icon': 'assets/reinmaker/icons/tribe_light.png',
      'featuredQuoteKey': 'reinmaker.tribes.light.quote',
      'ageDifficulty': {
        '2-5': {'difficulty': 'spark', 'hintCadence': 'frequent'},
        '6-12': {'difficulty': 'spark', 'hintCadence': 'frequent'},
        '13-17': {'difficulty': 'craft', 'hintCadence': 'moderate'},
        '18-35': {'difficulty': 'craft', 'hintCadence': 'moderate'},
        '36-60': {'difficulty': 'craft', 'hintCadence': 'moderate'},
        '61-102': {'difficulty': 'spark', 'hintCadence': 'frequent'},
      },
      'tiers': [
        {
          'tier': 1,
          'stoneId': 'light.spark',
          'quests': ['q.light.001', 'q.light.002'],
          'lessonRefs': ['leaves-change-color'],
        },
        {
          'tier': 2,
          'stoneId': 'light.craft',
          'quests': ['q.light.101'],
          'lessonRefs': ['water-cycle'],
        },
        {
          'tier': 3,
          'stoneId': 'light.mastery',
          'quests': ['q.light.201'],
          'lessonRefs': [],
        },
      ],
      'finaleStoneId': 'light.mastery',
    },
  ],
  'quests': [
    {
      'id': 'q.light.001',
      'tribe': 'Light',
      'tier': 1,
      'kind': 'puzzle',
      'ageBuckets': ['6-12', '13-17', '18-35', '36-60', '61-102'],
      'lessonRef': 'leaves-change-color',
      'localizationKey': 'reinmaker.quests.light_001',
      'estimatedDurationMin': 4,
      'rewards': {
        'xp': 25,
        'cosmetics': ['badge.light.spark'],
        'stoneId': 'light.spark',
      },
      'captions': {
        'en': 'captions/q.light.001.en.vtt',
        'es': 'captions/q.light.001.es.vtt',
        'fr': 'captions/q.light.001.fr.vtt',
      },
    },
    {
      'id': 'q.light.002',
      'tribe': 'Light',
      'tier': 1,
      'kind': 'builder',
      'ageBuckets': ['6-12', '13-17', '18-35', '36-60'],
      'lessonRef': 'leaves-change-color',
      'localizationKey': 'reinmaker.quests.light_002',
      'estimatedDurationMin': 5,
      'rewards': {
        'xp': 20,
        'cosmetics': ['frame.hall.light'],
        'stoneId': 'light.spark',
      },
      'captions': {
        'en': 'captions/q.light.002.en.vtt',
        'es': 'captions/q.light.002.es.vtt',
        'fr': 'captions/q.light.002.fr.vtt',
      },
    },
  ],
  'assets': {
    'icons': [],
    'captions': [],
    'audio': [],
  },
  'contentHash': '',
};

final defaultQuestLibrary = <String, Map<String, dynamic>>{
  'q.light.001': {
    'id': 'q.light.001',
    'tribe': 'Light',
    'tier': 1,
    'kind': 'puzzle',
    'ageBuckets': ['6-12', '13-17', '18-35', '36-60', '61-102'],
    'lessonRef': 'leaves-change-color',
    'localizationKey': 'reinmaker.quests.light_001',
    'estimatedDurationMin': 4,
    'steps': [
      {
        'type': 'dialogue',
        'npc': 'Kelly',
        'cue': 'light.spark.intro',
        'emotion': 'encouraging',
        'text': 'Welcome to the Hall of Light! Let\'s explore how design helps us see patterns.',
      },
      {
        'type': 'puzzle',
        'spec': 'grid_select',
        'goal': 'Highlight the highest-contrast composition for the autumn poster',
        'data': {
          'grid': 6,
          'targets': [1, 5, 17, 19],
          'hint': 'Contrast means placing light next to dark so details pop.',
        },
      },
      {
        'type': 'empathy',
        'npc': 'Nova',
        'branches': [
          {
            'prompt': 'Nova wants the poster to feel calm. What do you suggest?',
            'responses': [
              {'text': 'Use gentle blues with lots of space', 'impact': 'positive', 'next': 1},
              {'text': 'Add flashing lights everywhere', 'impact': 'growth'},
            ],
          },
          {
            'prompt': 'Great! How will you guide the viewer\'s eye?',
            'responses': [
              {'text': 'Place the brightest leaf near the message', 'impact': 'positive'},
              {'text': 'Hide the message behind leaves', 'impact': 'growth'},
            ],
          },
        ],
      },
    ],
    'successCriteria': {'scoreMin': 0.7, 'timeMaxSec': 240},
    'rewards': {
      'xp': 25,
      'cosmetics': ['badge.light.spark'],
      'stoneId': 'light.spark',
    },
    'captions': {
      'en': 'captions/q.light.001.en.vtt',
      'es': 'captions/q.light.001.es.vtt',
      'fr': 'captions/q.light.001.fr.vtt',
    },
  },
  'q.light.002': {
    'id': 'q.light.002',
    'tribe': 'Light',
    'tier': 1,
    'kind': 'builder',
    'ageBuckets': ['6-12', '13-17', '18-35', '36-60'],
    'lessonRef': 'leaves-change-color',
    'localizationKey': 'reinmaker.quests.light_002',
    'estimatedDurationMin': 5,
    'steps': [
      {
        'type': 'dialogue',
        'npc': 'Kelly',
        'cue': 'light.spark.palette',
        'emotion': 'curious',
        'text': 'Let\'s build a flowing layout that guides learners through autumn colors.',
      },
      {
        'type': 'builder',
        'template': 'poster_layout_v1',
        'goal': 'Arrange panels so warm colors lead to cool reflections',
        'palette': ['panel.photo', 'panel.fact', 'arrow.flow', 'icon.sun'],
      },
      {
        'type': 'dialogue',
        'npc': 'Kelly',
        'cue': 'light.spark.wrap',
        'emotion': 'encouraging',
        'text': 'Notice how the light flows across your design—that\'s the lens of composition!',
      },
    ],
    'successCriteria': {'scoreMin': 0.6, 'timeMaxSec': 300},
    'rewards': {
      'xp': 20,
      'cosmetics': ['frame.hall.light'],
      'stoneId': 'light.spark',
    },
    'captions': {
      'en': 'captions/q.light.002.en.vtt',
      'es': 'captions/q.light.002.es.vtt',
      'fr': 'captions/q.light.002.fr.vtt',
    },
  },
};

