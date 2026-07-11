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
    // 잠금 아이콘 2개 (대대, 3구)
    expect(find.byIcon(Icons.lock), findsNWidgets(2));
  });

  testWidgets('잠금 선택지는 비활성화되어 있다', (tester) async {
    await tester.pumpWidget(const BilliardCoachApp());

    final chips = tester.widgetList<ChoiceChip>(find.byType(ChoiceChip)).toList();
    expect(chips.length, 4);
    // 중대/4구는 선택됨, 대대/3구는 onSelected == null (잠금)
    final lockedCount = chips.where((c) => c.onSelected == null).length;
    expect(lockedCount, 2);
    final selectedCount = chips.where((c) => c.selected).length;
    expect(selectedCount, 2);
  });

  testWidgets('서버 주소 설정 다이얼로그 열기', (tester) async {
    await tester.pumpWidget(const BilliardCoachApp());

    await tester.tap(find.byIcon(Icons.settings));
    await tester.pumpAndSettle();
    expect(find.text('서버 주소'), findsOneWidget);
    expect(find.byType(TextField), findsOneWidget);
  });
}
