import 'package:flutter/material.dart';

import '../config.dart';
import '../services/api.dart';
import 'capture_screen.dart';
import 'result_screen.dart';

/// 당구장 없이 가이드 화면을 확인하는 데모용 샘플 배치.
TopViewResult demoTopViewResult() => TopViewResult(
      ok: true,
      diamondCount: 20,
      diamondErrMm: 0.5,
      refined: true,
      clothColor: 'blue',
      balls: [
        BallInfo(color: 'white', xMm: 500, yMm: 300, score: 1),
        BallInfo(color: 'yellow', xMm: 2200, yMm: 1000, score: 1),
        BallInfo(color: 'red', xMm: 1500, yMm: 635, score: 1),
        BallInfo(color: 'red', xMm: 1500, yMm: 1100, score: 1),
      ],
    );

/// 4구 당구공 3종(흰/노랑/빨강) 로고.
class _BilliardBallsLogo extends StatelessWidget {
  const _BilliardBallsLogo();

  Widget _ball(Color color) => Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: RadialGradient(
            center: const Alignment(-0.4, -0.4),
            colors: [
              Color.lerp(color, Colors.white, 0.55)!,
              color,
              Color.lerp(color, Colors.black, 0.35)!,
            ],
            stops: const [0.0, 0.55, 1.0],
          ),
          boxShadow: const [
            BoxShadow(color: Colors.black45, blurRadius: 6, offset: Offset(0, 3)),
          ],
        ),
      );

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _ball(Colors.white),
        const SizedBox(width: 14),
        _ball(const Color(0xFFD32F2F)),
        const SizedBox(width: 14),
        _ball(const Color(0xFFF9C825)),
      ],
    );
  }
}

/// [인트로] — 타이틀, 중대/대대 선택(대대 잠금), 4구/3구 선택(3구 잠금), 입장.
class IntroScreen extends StatefulWidget {
  const IntroScreen({super.key});

  @override
  State<IntroScreen> createState() => _IntroScreenState();
}

class _IntroScreenState extends State<IntroScreen> {
  // MVP: 중대 + 4구 고정, 나머지는 잠금
  final String _table = '중대';
  final String _game = '4구';

  Future<void> _editServerUrl() async {
    final current = await AppConfig.serverUrl();
    if (!mounted) return;
    final controller = TextEditingController(text: current);
    final url = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('서버 주소'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(hintText: 'http://192.168.x.x:8000'),
          keyboardType: TextInputType.url,
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('취소')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('저장'),
          ),
        ],
      ),
    );
    if (url != null && url.isNotEmpty) {
      await AppConfig.setServerUrl(url);
    }
  }

  Widget _choiceRow(String label, List<(String, bool)> options, String selected) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 14, color: Colors.white70)),
        const SizedBox(height: 8),
        Row(
          children: [
            for (final (name, locked) in options) ...[
              Expanded(
                child: ChoiceChip(
                  label: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(name),
                      if (locked) ...[
                        const SizedBox(width: 4),
                        const Icon(Icons.lock, size: 14),
                      ],
                    ],
                  ),
                  selected: name == selected,
                  onSelected: locked ? null : (_) {},
                ),
              ),
              const SizedBox(width: 8),
            ],
          ],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D2137),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white54),
            tooltip: '서버 주소 설정',
            onPressed: _editServerUrl,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const _BilliardBallsLogo(),
            const SizedBox(height: 20),
            const Text(
              'BILLIARD COACH',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.white,
                letterSpacing: 2,
              ),
            ),
            const SizedBox(height: 48),
            _choiceRow('당구대', [('중대', false), ('대대', true)], _table),
            const SizedBox(height: 24),
            _choiceRow('게임', [('4구', false), ('3구', true)], _game),
            const SizedBox(height: 48),
            FilledButton(
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Colors.amber,
                foregroundColor: Colors.black,
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CaptureScreen()),
                );
              },
              child: const Text('입장', style: TextStyle(fontSize: 18)),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.white70,
                side: const BorderSide(color: Colors.white24),
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              icon: const Icon(Icons.visibility, size: 18),
              label: const Text('데모 보기 (샘플 배치)'),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => ResultScreen(result: demoTopViewResult()),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
