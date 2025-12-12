/// Lesson model representing a teachable unit with content and metadata
class Lesson {
  /// Unique identifier for the lesson
  final String id;

  /// Display title for the lesson
  final String title;

  /// Full script/text content of the lesson
  final String script;

  /// Optional audio file path (WAV)
  final String? audioPath;

  /// Optional A2F frames path (JSON)
  final String? a2fPath;

  /// Duration in seconds (if known)
  final int? duration;

  const Lesson({
    required this.id,
    required this.title,
    required this.script,
    this.audioPath,
    this.a2fPath,
    this.duration,
  });

  factory Lesson.fromJson(Map<String, dynamic> json) => Lesson(
        id: json['id'] as String,
        title: json['title'] as String,
        script: json['script'] as String,
        audioPath: json['audioPath'] as String?,
        a2fPath: json['a2fPath'] as String?,
        duration: json['duration'] as int?,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'script': script,
        if (audioPath != null) 'audioPath': audioPath,
        if (a2fPath != null) 'a2fPath': a2fPath,
        if (duration != null) 'duration': duration,
      };
}


























