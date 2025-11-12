# Lesson Models

Shared Dart models and schemas for lesson data in Kelly OS.

## Usage

```dart
import 'package:lesson_models/lesson_models.dart';

final lesson = Lesson.fromJson(jsonData);
print('Lesson: ${lesson.title}');
```

## Schema

See `schema/lesson.schema.json` for JSON Schema definition.

Future: Add AJV validation for JSON validation at runtime.


















