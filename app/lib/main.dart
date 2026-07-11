import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/intro_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // 인트로/촬영 화면은 세로 고정 (기획서 6.4 — 가로 지원은 추후 메인화면 한정)
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  runApp(const BilliardCoachApp());
}

class BilliardCoachApp extends StatelessWidget {
  const BilliardCoachApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BILLIARD COACH',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorSchemeSeed: Colors.amber,
        useMaterial3: true,
      ),
      home: const IntroScreen(),
    );
  }
}
