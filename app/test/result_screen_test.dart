import 'dart:convert';

import 'package:billiard_coach/screens/result_screen.dart';
import 'package:billiard_coach/services/api.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

TopViewResult _fakeResult() => TopViewResult(
      ok: true,
      diamondCount: 20,
      diamondErrMm: 0.9,
      refined: true,
      clothColor: 'blue',
      balls: [
        BallInfo(color: 'white', xMm: 500, yMm: 300, score: 0.9),
        BallInfo(color: 'yellow', xMm: 2200, yMm: 1000, score: 0.9),
        BallInfo(color: 'red', xMm: 1500, yMm: 635, score: 0.9),
        BallInfo(color: 'red', xMm: 1500, yMm: 1100, score: 0.9),
      ],
    );

ApiClient _mockApi() {
  final mock = MockClient((req) async {
    expect(req.url.path, '/guides');
    final body = jsonDecode(req.body) as Map<String, dynamic>;
    expect((body['balls'] as List).length, 4);
    return http.Response(
      jsonEncode({
        'ok': true,
        'guides': [
          {
            'shot_id': 'direct_0',
            'name': '직접치기 (빨강① 먼저)',
            'feasible': true,
            'category': 'direct',
            'cushions': 0,
            'reason': '',
            'cue_path': [
              [500.0, 300.0],
              [1450.0, 600.0],
              [1500.0, 1030.0],
            ],
            'object_path': [
              [1500.0, 635.0],
              [1800.0, 500.0],
            ],
            'ghost': [1450.0, 600.0],
            'thickness': 0.7,
            'thickness_label': '3/4 두께',
            'aim_offset': 0.5,
            'tip': '중단',
            'tip_delta_deg': 0.0,
            'kiss': '없음',
            'kiss_level': 1,
            'difficulty': 2,
            'difficulty_label': '쉬움',
          },
          {
            'shot_id': 'direct_1',
            'name': '직접치기 B (빨강② 먼저)',
            'feasible': false,
            'category': 'direct',
            'cushions': 0,
            'reason': '경로 막힘',
            'cue_path': [],
            'object_path': [],
            'ghost': null,
            'thickness': 0,
            'thickness_label': '',
            'aim_offset': 0,
            'tip': '중단',
            'tip_delta_deg': 0.0,
            'kiss': '없음',
            'kiss_level': 1,
            'difficulty': 3,
            'difficulty_label': '보통',
          },
          {
            'shot_id': 'three_0_red1',
            'name': '뒤돌려치기 (빨강① 먼저)',
            'feasible': true,
            'category': 'three',
            'cushions': 3,
            'reason': '',
            'cue_path': [
              [500.0, 300.0],
              [1450.0, 600.0],
              [2507.0, 900.0],
              [1800.0, 1237.0],
              [32.0, 800.0],
              [1400.0, 1050.0],
            ],
            'object_path': [
              [1500.0, 635.0],
              [1800.0, 500.0],
            ],
            'ghost': [1450.0, 600.0],
            'thickness': 0.55,
            'thickness_label': '1/2 두께',
            'aim_offset': -0.7,
            'tip': '상단 (밀어치기)',
            'tip_delta_deg': 20.0,
            'kiss': '보통',
            'kiss_level': 3,
            'difficulty': 4,
            'difficulty_label': '어려움',
          },
        ],
      }),
      200,
      headers: {'content-type': 'application/json'},
    );
  });
  return ApiClient('http://test', client: mock);
}

void main() {
  testWidgets('가이드 로드 → 카드 캐러셀 표시, 실행 가능한 샷 자동 선택', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(result: _fakeResult(), api: _mockApi()),
    ));
    await tester.pumpAndSettle();

    // 카드 캐러셀 존재 + 첫 카드(실행 가능) 내용
    expect(find.byType(PageView), findsOneWidget);
    expect(find.text('직접치기 (빨강① 먼저)'), findsOneWidget);
    expect(find.textContaining('3/4 두께'), findsOneWidget);
    // 난이도/키스 등급 게이지
    expect(find.text('난이도'), findsWidgets);
    expect(find.text('키스'), findsWidgets);
    // 당점 스나이퍼 + 두께 다이어그램 표시
    expect(find.text('당점'), findsWidgets);
    expect(find.text('두께'), findsWidgets);
    expect(
      find.byWidgetPredicate((w) => w is CustomPaint && w.painter is TipPainter),
      findsWidgets,
    );
    expect(
      find.byWidgetPredicate(
          (w) => w is CustomPaint && w.painter is ThicknessPainter),
      findsWidgets,
    );
    // 메인 당구대(미니 아님)에 선택된 가이드가 전달됨
    final painters = tester
        .widgetList<CustomPaint>(find.byWidgetPredicate(
            (w) => w is CustomPaint && w.painter is TablePainter))
        .map((w) => w.painter as TablePainter)
        .toList();
    final main = painters.firstWhere((p) => !p.mini);
    expect(main.guide?.id, 'direct_0');
    expect(main.balls.length, 4);
  });

  testWidgets('직접치기/1쿠션/2쿠션/3쿠션 탭: 전환 시 해당 길만 표시', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(result: _fakeResult(), api: _mockApi()),
    ));
    await tester.pumpAndSettle();

    expect(find.text('1쿠션'), findsOneWidget);
    expect(find.text('2쿠션'), findsOneWidget);
    // 기본 탭(직접치기): 3쿠션 길은 안 보임
    expect(find.text('직접치기 (빨강① 먼저)'), findsOneWidget);
    expect(find.text('뒤돌려치기 (빨강① 먼저)'), findsNothing);

    await tester.tap(find.text('3쿠션'));
    await tester.pumpAndSettle();
    expect(find.text('뒤돌려치기 (빨강① 먼저)'), findsOneWidget);
    expect(find.text('직접치기 (빨강① 먼저)'), findsNothing);

    // 빈 탭(2쿠션)은 안내 문구
    await tester.tap(find.text('2쿠션'));
    await tester.pumpAndSettle();
    expect(find.text('이 탭에서 가능한 길이 없어요'), findsOneWidget);
  });

  testWidgets('플레이 버튼 → 공 움직임 애니메이션', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(result: _fakeResult(), api: _mockApi()),
    ));
    await tester.pumpAndSettle();

    await tester.tap(find.byIcon(Icons.play_circle_fill));
    await tester.pump(const Duration(milliseconds: 500));

    final painters = tester
        .widgetList<CustomPaint>(find.byWidgetPredicate(
            (w) => w is CustomPaint && w.painter is TablePainter))
        .map((w) => w.painter as TablePainter)
        .toList();
    final main = painters.firstWhere((p) => !p.mini);
    expect(main.animCueMm, isNotNull, reason: '수구가 경로를 따라 이동 중이어야 함');
    // 애니메이션 끝까지 진행
    await tester.pumpAndSettle();
  });

  testWidgets('불가 샷 카드로 스와이프하면 사유 표시 + 경로 제거', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(result: _fakeResult(), api: _mockApi()),
    ));
    await tester.pumpAndSettle();

    await tester.drag(find.byType(PageView), const Offset(-400, 0));
    await tester.pumpAndSettle();

    expect(find.textContaining('— 불가'), findsOneWidget);
    expect(find.text('경로 막힘'), findsOneWidget);
    final painters = tester
        .widgetList<CustomPaint>(find.byWidgetPredicate(
            (w) => w is CustomPaint && w.painter is TablePainter))
        .map((w) => w.painter as TablePainter)
        .toList();
    final main = painters.firstWhere((p) => !p.mini);
    expect(main.guide, isNull);
  });

  testWidgets('수구 선택 토글(흰공/노란공) 표시', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(result: _fakeResult(), api: _mockApi()),
    ));
    await tester.pumpAndSettle();
    expect(find.text('흰공'), findsOneWidget);
    expect(find.text('노란공'), findsOneWidget);
  });

  testWidgets('공 미검출이면 안내 문구', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(
        result: TopViewResult(ok: true),
        api: _mockApi(),
      ),
    ));
    await tester.pumpAndSettle();
    expect(find.textContaining('가이드를 계산할 수 없어요'), findsOneWidget);
  });

  testWidgets('세로 방향 캔버스: 세로가 가로보다 길다', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: ResultScreen(result: _fakeResult(), api: _mockApi()),
    ));
    await tester.pumpAndSettle();
    final box = tester.getSize(find.byWidgetPredicate((w) =>
        w is CustomPaint &&
        w.painter is TablePainter &&
        !(w.painter as TablePainter).mini));
    expect(box.height, greaterThan(box.width));
  });
}
