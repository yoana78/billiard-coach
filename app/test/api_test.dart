import 'dart:convert';
import 'dart:typed_data';

import 'package:billiard_coach/services/api.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test('analyzeFrame: 응답 JSON 파싱', () async {
    final mock = MockClient((req) async {
      expect(req.url.path, '/analyze');
      return http.Response(
        jsonEncode({
          'ok': true,
          'ready': true,
          'table_found': true,
          'diamond_long': 10,
          'diamond_short': 4,
          'message': '좋아요! 이대로 촬영하세요',
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    });
    final api = ApiClient('http://test', client: mock);
    final r = await api.analyzeFrame(Uint8List.fromList([1, 2, 3]));
    expect(r.ready, true);
    expect(r.tableFound, true);
    expect(r.diamondLong, 10);
    expect(r.diamondShort, 4);
    expect(r.message, contains('촬영'));
  });

  test('topview: 성공 응답에서 PNG 디코딩', () async {
    final png = base64Encode([137, 80, 78, 71]);
    final mock = MockClient((req) async {
      expect(req.url.path, '/topview');
      return http.Response(
        jsonEncode({
          'ok': true,
          'diamond_count': 20,
          'diamond_err_mm': 0.9,
          'refined': true,
          'balls': [
            {'color': 'white', 'x_mm': 500.0, 'y_mm': 300.0, 'score': 0.9},
            {'color': 'yellow', 'x_mm': 2000.0, 'y_mm': 800.0, 'score': 0.9},
            {'color': 'red', 'x_mm': 1200.0, 'y_mm': 600.0, 'score': 0.9},
            {'color': 'red', 'x_mm': 1800.0, 'y_mm': 200.0, 'score': 0.8},
          ],
          'image_base64': png,
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    });
    final api = ApiClient('http://test', client: mock);
    final r = await api.topview(Uint8List.fromList([1]));
    expect(r.ok, true);
    expect(r.diamondCount, 20);
    expect(r.diamondErrMm, closeTo(0.9, 0.001));
    expect(r.refined, true);
    expect(r.imagePng, isNotNull);
    expect(r.imagePng!.length, 4);
    expect(r.balls.length, 4);
    expect(r.balls.where((b) => b.color == 'red').length, 2);
    expect(r.balls.first.xMm, 500.0);
  });

  test('topview: 실패 응답', () async {
    final mock = MockClient((req) async => http.Response(
          jsonEncode({'ok': false, 'reason': '당구대(천)를 찾지 못함'}),
          200,
          headers: {'content-type': 'application/json'},
        ));
    final api = ApiClient('http://test', client: mock);
    final r = await api.topview(Uint8List.fromList([1]));
    expect(r.ok, false);
    expect(r.reason, contains('당구대'));
  });
}
