import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:lesson_models/lesson_models.dart';

/// Loads lesson data from JSON
class LessonLoader {
  /// Load sample lesson from assets
  static Future<Lesson> loadSampleLesson() async {
    try {
      final jsonString = await rootBundle.loadString(
        'assets/lessons/sample_lesson.json',
      );
      final json = jsonDecode(jsonString) as Map<String, dynamic>;
      return Lesson.fromJson(json);
    } catch (e) {
      throw Exception('Failed to load sample lesson: $e');
    }
  }

  /// Load lesson from file path
  static Future<Lesson> loadLessonFromFile(String path) async {
    try {
      final file = File(path);
      if (!await file.exists()) {
        throw Exception('Lesson file not found: $path');
      }
      final jsonString = await file.readAsString();
      final json = jsonDecode(jsonString) as Map<String, dynamic>;
      return Lesson.fromJson(json);
    } catch (e) {
      throw Exception('Failed to load lesson from file: $e');
    }
  }

  /// Load A2F frames from JSON
  static Future<Map<String, dynamic>> loadA2fFrames(String path) async {
    try {
      final file = File(path);
      if (!await file.exists()) {
        throw Exception('A2F file not found: $path');
      }
      final jsonString = await file.readAsString();
      return jsonDecode(jsonString) as Map<String, dynamic>;
    } catch (e) {
      throw Exception('Failed to load A2F frames: $e');
    }
  }
}


























