import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

/// /analyze 응답 — 프리뷰 프레임 1장의 실시간 분석 결과.
class AnalyzeResult {
  final bool ready;
  final bool tableFound;
  final int diamondLong;
  final int diamondShort;
  final String message;

  AnalyzeResult({
    required this.ready,
    required this.tableFound,
    required this.diamondLong,
    required this.diamondShort,
    required this.message,
  });

  factory AnalyzeResult.fromJson(Map<String, dynamic> j) => AnalyzeResult(
        ready: j['ready'] == true,
        tableFound: j['table_found'] == true,
        diamondLong: (j['diamond_long'] ?? 0) as int,
        diamondShort: (j['diamond_short'] ?? 0) as int,
        message: (j['message'] ?? '') as String,
      );
}

/// 검출된 공 1개 — 좌표는 경기면 mm 기준 (탑뷰 좌표계).
class BallInfo {
  final String color; // 'white' | 'yellow' | 'red'
  final double xMm;
  final double yMm;
  final double score;

  BallInfo({required this.color, required this.xMm, required this.yMm, required this.score});

  factory BallInfo.fromJson(Map<String, dynamic> j) => BallInfo(
        color: (j['color'] ?? '') as String,
        xMm: ((j['x_mm'] ?? 0) as num).toDouble(),
        yMm: ((j['y_mm'] ?? 0) as num).toDouble(),
        score: ((j['score'] ?? 0) as num).toDouble(),
      );
}

/// /topview 응답 — 탑뷰 변환 결과.
class TopViewResult {
  final bool ok;
  final String reason;
  final int diamondCount;
  final double diamondErrMm;
  final bool refined;
  final bool partial;         // 일부 촬영에서 역산했는가
  final String clothColor;    // 'blue' | 'green'
  final List<BallInfo> balls;
  final Uint8List? imagePng;  // 서버의 워프 결과 (디버그용)

  TopViewResult({
    required this.ok,
    this.reason = '',
    this.diamondCount = 0,
    this.diamondErrMm = -1,
    this.refined = false,
    this.partial = false,
    this.clothColor = 'blue',
    this.balls = const [],
    this.imagePng,
  });

  factory TopViewResult.fromJson(Map<String, dynamic> j) {
    if (j['ok'] != true) {
      return TopViewResult(ok: false, reason: (j['reason'] ?? '알 수 없는 오류') as String);
    }
    return TopViewResult(
      ok: true,
      diamondCount: (j['diamond_count'] ?? 0) as int,
      diamondErrMm: ((j['diamond_err_mm'] ?? -1) as num).toDouble(),
      refined: j['refined'] == true,
      partial: j['partial'] == true,
      clothColor: (j['cloth_color'] ?? 'blue') as String,
      balls: ((j['balls'] ?? []) as List)
          .map((b) => BallInfo.fromJson(b as Map<String, dynamic>))
          .toList(),
      imagePng: j['image_base64'] == null
          ? null
          : base64Decode(j['image_base64'] as String),
    );
  }
}

/// 샷 가이드 1개 — 좌표는 모두 경기면 mm.
class GuideShot {
  final String id;
  final String name;
  final bool feasible;
  final String category;               // 'direct' | 'one' | 'two' | 'three'
  final int cushions;                  // 경유 쿠션 수
  final String reason;
  final List<List<double>> cuePath;    // 수구 경로 꼭짓점들
  final List<List<double>> objectPath; // 1적구 예상 진행
  final List<double>? ghost;           // 겨냥점 (고스트볼 중심)
  final double thickness;              // 두께 0~1
  final String thicknessLabel;
  final double aimOffset;              // 겨냥 가로 오프셋 (공 지름 단위, +오른쪽)
  final String tip;
  final double tipDeltaDeg;            // 당점 상하 보정각 (+위/-아래)
  final int tipX;                      // 당점 가로 단계 (-3 좌 ~ +3 우)
  final int tipY;                      // 당점 세로 단계 (-3 하 ~ +3 상)
  final int power;                     // 강도 1(매우약)~5(매우강)
  final String powerLabel;
  final String kiss;                   // 없음/거의없음/보통/높음/매우높음
  final int kissLevel;                 // 1(없음)~5(매우높음)
  final int difficulty;                // 1(매우쉬움)~5(매우어려움)
  final String difficultyLabel;

  GuideShot({
    required this.id,
    required this.name,
    required this.feasible,
    this.category = 'direct',
    this.cushions = 0,
    required this.reason,
    required this.cuePath,
    required this.objectPath,
    required this.ghost,
    this.thickness = 0,
    required this.thicknessLabel,
    this.aimOffset = 0,
    required this.tip,
    this.tipDeltaDeg = 0,
    this.tipX = 0,
    this.tipY = 0,
    this.power = 3,
    this.powerLabel = '보통',
    required this.kiss,
    this.kissLevel = 1,
    this.difficulty = 3,
    this.difficultyLabel = '보통',
  });

  static List<List<double>> _points(dynamic v) => ((v ?? []) as List)
      .map((p) => (p as List).map((e) => (e as num).toDouble()).toList())
      .toList();

  factory GuideShot.fromJson(Map<String, dynamic> j) => GuideShot(
        id: (j['shot_id'] ?? '') as String,
        name: (j['name'] ?? '') as String,
        feasible: j['feasible'] == true,
        category: (j['category'] ?? 'direct') as String,
        cushions: (j['cushions'] ?? 0) as int,
        reason: (j['reason'] ?? '') as String,
        cuePath: _points(j['cue_path']),
        objectPath: _points(j['object_path']),
        ghost: j['ghost'] == null
            ? null
            : (j['ghost'] as List).map((e) => (e as num).toDouble()).toList(),
        thickness: ((j['thickness'] ?? 0) as num).toDouble(),
        thicknessLabel: (j['thickness_label'] ?? '') as String,
        aimOffset: ((j['aim_offset'] ?? 0) as num).toDouble(),
        tip: (j['tip'] ?? '') as String,
        tipDeltaDeg: ((j['tip_delta_deg'] ?? 0) as num).toDouble(),
        tipX: (j['tip_x'] ?? 0) as int,
        tipY: (j['tip_y'] ?? 0) as int,
        power: (j['power'] ?? 3) as int,
        powerLabel: (j['power_label'] ?? '보통') as String,
        kiss: (j['kiss'] ?? '없음') as String,
        kissLevel: (j['kiss_level'] ?? 1) as int,
        difficulty: (j['difficulty'] ?? 3) as int,
        difficultyLabel: (j['difficulty_label'] ?? '보통') as String,
      );
}

class ApiClient {
  final String baseUrl;
  final http.Client _client;

  ApiClient(this.baseUrl, {http.Client? client}) : _client = client ?? http.Client();

  Future<List<GuideShot>> guides(List<BallInfo> balls, String cue,
      {String game = 'four', String table = 'medium'}) async {
    final res = await _client
        .post(
          Uri.parse('$baseUrl/guides'),
          headers: {'content-type': 'application/json'},
          body: jsonEncode({
            'balls': [
              for (final b in balls)
                {'color': b.color, 'x_mm': b.xMm, 'y_mm': b.yMm}
            ],
            'cue': cue,
            'game': game,
            'table': table,
          }),
        )
        // 클라우드 무료 플랜은 유휴 후 첫 요청에 콜드스타트(수십 초)가 있음
        .timeout(const Duration(seconds: 75));
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    if (data['ok'] != true) {
      throw Exception((data['reason'] ?? '가이드 계산 실패') as String);
    }
    return ((data['guides'] ?? []) as List)
        .map((g) => GuideShot.fromJson(g as Map<String, dynamic>))
        .toList();
  }

  Future<AnalyzeResult> analyzeFrame(Uint8List jpeg) async {
    final req = http.MultipartRequest('POST', Uri.parse('$baseUrl/analyze'))
      ..files.add(http.MultipartFile.fromBytes('file', jpeg, filename: 'frame.jpg'));
    final res = await _client.send(req).timeout(const Duration(seconds: 5));
    final body = await res.stream.bytesToString();
    return AnalyzeResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  Future<TopViewResult> topview(Uint8List imageBytes,
      {String table = 'medium'}) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/topview?table=$table'))
      ..files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: 'photo.jpg'));
    // 콜드스타트 + 이미지 분석 시간 여유
    final res = await _client.send(req).timeout(const Duration(seconds: 120));
    final body = await res.stream.bytesToString();
    return TopViewResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  /// 서버 깨우기 (클라우드 콜드스타트 선제 해소). 실패해도 무시.
  Future<void> warmUp() async {
    try {
      await _client
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 90));
    } catch (_) {}
  }
}
