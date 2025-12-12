import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_unity_widget/flutter_unity_widget.dart';

/// Wraps UnityWidget and provides helpers for communication
class UnityView extends StatefulWidget {
  final Function(UnityViewController) onCreated;

  const UnityView({
    super.key,
    required this.onCreated,
  });

  @override
  State<UnityView> createState() => _UnityViewState();
}

class _UnityViewState extends State<UnityView> {
  UnityViewController? _controller;

  @override
  void initState() {
    super.initState();
  }

  void onUnityCreated(UnityViewController controller) {
    setState(() {
      _controller = controller;
    });
    widget.onCreated(controller);
  }

  @override
  Widget build(BuildContext context) {
    return UnityWidget(
      onUnityCreated: onUnityCreated,
      fullscreen: true,
      onUnityMessage: (message) {
        print('📥 Kelly OS: Unity message: $message');
      },
    );
  }
}

/// Helper to post messages to Unity
extension UnityControllerExtension on UnityViewController {
  void postMessage(String object, String method, String payload) {
    postMessage(
      object: object,
      method: method,
      args: payload,
    );
  }
}


























