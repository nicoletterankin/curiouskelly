import 'package:flutter/material.dart';

import '../models/tribe_pack.dart';

class StoneRing extends StatelessWidget {
  const StoneRing({
    super.key,
    required this.tribe,
    required this.playerStones,
    this.size = 96,
  });

  final TribePack tribe;
  final List<String> playerStones;
  final double size;

  static const _tiers = ['spark', 'craft', 'mastery'];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          _buildHalo(),
          _buildStoneRow(),
        ],
      ),
    );
  }

  Widget _buildHalo() {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: SweepGradient(
          colors: [
            _colorFromHex(tribe.color).withOpacity(0.6),
            _colorFromHex(tribe.color).withOpacity(0.2),
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: _colorFromHex(tribe.color).withOpacity(0.35),
            blurRadius: 16,
            spreadRadius: 4,
          )
        ],
      ),
    );
  }

  Widget _buildStoneRow() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      mainAxisSize: MainAxisSize.min,
      children: _tiers
          .map((tier) => _StoneGem(
                filled: playerStones.any((stone) => stone.endsWith('.$tier')),
                tier: tier,
              ))
          .toList(),
    );
  }

  Color _colorFromHex(String hex) {
    final buffer = StringBuffer();
    if (hex.length == 6 || hex.length == 7) buffer.write('ff');
    buffer.write(hex.replaceFirst('#', ''));
    return Color(int.parse(buffer.toString(), radix: 16));
  }
}

class _StoneGem extends StatelessWidget {
  const _StoneGem({required this.filled, required this.tier});

  final bool filled;
  final String tier;

  @override
  Widget build(BuildContext context) {
    final baseColor = filled ? Colors.amberAccent : Colors.white24;
    final label = tier.substring(0, 1).toUpperCase();
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: 28,
      height: 28,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: baseColor.withOpacity(filled ? 0.9 : 0.2),
        border: Border.all(color: baseColor, width: filled ? 2 : 1),
      ),
      alignment: Alignment.center,
      child: Text(
        label,
        style: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.bold,
          color: filled ? Colors.black87 : Colors.white70,
        ),
      ),
    );
  }
}




















