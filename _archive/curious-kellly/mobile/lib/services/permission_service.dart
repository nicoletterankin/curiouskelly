import 'package:permission_handler/permission_handler.dart';
import 'package:logger/logger.dart';

/// Permission Service
/// Handles microphone and other permissions
class PermissionService {
  final Logger _logger = Logger();
  
  /// Request microphone permission
  Future<bool> requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    
    _logger.i('[Permission] Microphone: $status');
    
    if (status.isGranted) {
      return true;
    } else if (status.isDenied) {
      _logger.w('[Permission] Microphone denied');
      return false;
    } else if (status.isPermanentlyDenied) {
      _logger.e('[Permission] Microphone permanently denied, opening settings');
      await openAppSettings();
      return false;
    }
    
    return false;
  }
  
  /// Check if microphone permission is granted
  Future<bool> hasMicrophonePermission() async {
    final status = await Permission.microphone.status;
    return status.isGranted;
  }
  
  /// Request storage permission (for caching audio)
  Future<bool> requestStoragePermission() async {
    final status = await Permission.storage.request();
    _logger.i('[Permission] Storage: $status');
    return status.isGranted;
  }
  
  /// Request notification permission (for reminders)
  Future<bool> requestNotificationPermission() async {
    final status = await Permission.notification.request();
    _logger.i('[Permission] Notification: $status');
    return status.isGranted;
  }
  
  /// Request all required permissions
  Future<Map<String, bool>> requestAllPermissions() async {
    return {
      'microphone': await requestMicrophonePermission(),
      'storage': await requestStoragePermission(),
      'notification': await requestNotificationPermission(),
    };
  }
  
  /// Check all permissions
  Future<Map<String, bool>> checkAllPermissions() async {
    return {
      'microphone': await Permission.microphone.status.isGranted,
      'storage': await Permission.storage.status.isGranted,
      'notification': await Permission.notification.status.isGranted,
    };
  }
}






















