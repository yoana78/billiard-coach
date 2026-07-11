import 'package:billiard_coach/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  testWidgets('인트로 화면: 타이틀·선택지·입장 버튼 표시', (tester) async {
    await tester.pumpWidget(const BilliardCoachApp());

    expect(find.text('BILLIARD COACH'), findsOneWidget);
    expect(find.text('중대'), findsOneWidget);
    expect(find.text('대대'), findsOneWidget);
    expect(find.text('4구'), findsOneWidget);
    expect(find.text('3구'), findsOneWidget);
    expect(find.text('입장'), findsOneWidget);
    expect(find.text('데모 보기 (샘플 배치)'), findsOneWidget);
  });

  testWidgets('대대/3구 선택 가능', (tester) async {
    await tester.pumpWidget(const BilliardCoachApp());

    // 기본 선택: 중대 + 4구
    var chips = tester.widgetList<ChoiceChip>(find.byType(ChoiceChip)).toList();
    expect(chips.length, 4);
    expect(chips.where((c) => c.selected).length, 2);

    // 대대·3구 탭하면 선택이 바뀐다
    await tester.tap(find.text('대대'));
    await tester.tap(find.text('3구'));
    await tester.pump();
    chips = tester.widgetList<ChoiceChip>(find.byType(ChoiceChip)).toList();
    final selected = <String>[];
    for (final c in chips) {
      if (c.selected) {
        final center = (c.label as Center).child as Text;
        selected.add(center.data ?? '');
      }
    }
    expect(selected, containsAll(['대대', '3구']));
  });

  testWidgets('서버 주소 설정 다이얼로그 열기', (tester) async {
    await tester.pumpWidget(const BilliardCoachApp());

    await tester.tap(find.byIcon(Icons.settings));
    await tester.pumpAndSettle();
    expect(find.text('서버 주소'), findsOneWidget);
    expect(find.byType(TextField), findsOneWidget);
  });
}
