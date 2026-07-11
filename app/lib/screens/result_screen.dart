import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../config.dart';
import '../services/api.dart';

/// 당구대 규격 (mm) — server/detection/spec.py 와 동기화 유지.
const double kTableWMm = 2540.0; // 장쿠션 방향 (화면 세로로 표시)
const double kTableHMm = 1270.0; // 단쿠션 방향 (화면 가로로 표시)
const double kBallRadiusMm = 32.75;
const double kDiamondOffsetMm = 95.0; // 쿠션 날 → 다이아몬드 중심
const double kDiamondSpacingMm = 317.5;
const double kCushionOverhangMm = 40.0;
const double kRailMm = 150.0; // 표시용 레일 폭 (다이아몬드 포함)

// 세로(portrait) 캔버스 mm 크기
const double kPortraitWMm = kTableHMm + 2 * kRailMm;
const double kPortraitHMm = kTableWMm + 2 * kRailMm;

/// 월드 좌표(mm, 가로 기준) → 세로 캔버스 좌표(mm).
/// 90도 회전(반사 아님)이라 가이드 경로의 좌우가 뒤집히지 않는다.
Offset worldToPortraitMm(double xMm, double yMm) {
  return Offset(kRailMm + yMm, kRailMm + (kTableWMm - xMm));
}

/// 탑뷰 + 샷 가이드 화면 (기획서 [메인]의 초기 형태).
/// 촬영 사진이 아니라 규격 모델 당구대를 세로로 렌더링하고,
/// 검출된 공 위치와 샷 가이드를 그 위에 그린다.
class ResultScreen extends StatefulWidget {
  final TopViewResult result;
  final ApiClient? api; // 테스트 주입용. null 이면 설정된 서버 주소 사용.

  const ResultScreen({super.key, required this.result, this.api});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen>
    with TickerProviderStateMixin {
  String _cue = 'white';
  List<GuideShot> _all = [];     // 서버가 준 전체 가이드
  List<GuideShot> _guides = [];  // 현재 탭(일반/3쿠션)의 가이드
  GuideShot? _selected;
  String _guideError = '';
  bool _loading = false;
  PageController? _cardController;
  static const _categories = ['direct', 'one', 'two', 'three'];
  late final TabController _tabController =
      TabController(length: _categories.length, vsync: this)
        ..addListener(_applyTab);
  late final AnimationController _anim = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2800),
  )..addListener(() => setState(() {}));

  bool get _hasBothCues =>
      widget.result.balls.any((b) => b.color == 'white') &&
      widget.result.balls.any((b) => b.color == 'yellow');

  @override
  void initState() {
    super.initState();
    // 수구 자동 선택: 흰공이 없으면 노란공으로 (3구만 검출된 경우 대응)
    final colors = widget.result.balls.map((b) => b.color).toSet();
    if (!colors.contains('white') && colors.contains('yellow')) {
      _cue = 'yellow';
    }
    _loadGuides();
  }

  Future<void> _loadGuides() async {
    if (widget.result.balls.isEmpty) {
      setState(() => _guideError = '공이 검출되지 않아 가이드를 계산할 수 없어요');
      return;
    }
    setState(() {
      _loading = true;
      _guideError = '';
      _selected = null;
    });
    try {
      final api = widget.api ?? ApiClient(await AppConfig.serverUrl());
      final guides = await api.guides(widget.result.balls, _cue);
      if (!mounted) return;
      _all = guides;
      _loading = false;
      _applyTab();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _guideError = '$e'.replaceFirst('Exception: ', '');
      });
    }
  }

  /// 현재 탭(직접치기/1쿠션/2쿠션/3쿠션)에 맞는 가이드 목록으로 갱신.
  void _applyTab() {
    if (_tabController.indexIsChanging) return; // 전환 완료 시점에만 적용
    final cat = _categories[_tabController.index];
    final filtered = _all.where((g) => g.category == cat).toList();
    // 탭을 바꾸면 항상 맨 앞(가장 쉬운) 카드부터
    const idx = 0;
    _cardController?.dispose();
    _cardController = filtered.isEmpty
        ? null
        : PageController(viewportFraction: 0.82, initialPage: idx);
    _anim.reset();
    final sel = filtered.isEmpty
        ? null
        : (filtered[idx].feasible ? filtered[idx] : null);
    _setAngleFor(sel);
    setState(() {
      _guides = filtered;
      _selected = sel;
    });
  }

  /// 플레이 애니메이션 진행도에 따른 수구/1적구 위치 (mm).
  (Offset?, Offset?) _animPositionsMm() {
    if (_anim.isDismissed || _selected == null) return (null, null);
    final path = _selected!.cuePath;
    if (path.length < 2) return (null, null);
    final pts = [for (final p in path) Offset(p[0], p[1])];
    final segLens = <double>[];
    var total = 0.0;
    for (var i = 0; i < pts.length - 1; i++) {
      final l = (pts[i + 1] - pts[i]).distance;
      segLens.add(l);
      total += l;
    }
    if (total <= 0) return (null, null);
    var remain = _anim.value * total;
    Offset cuePos = pts.last;
    for (var i = 0; i < segLens.length; i++) {
      if (remain <= segLens[i]) {
        cuePos = pts[i] + (pts[i + 1] - pts[i]) * (remain / segLens[i]);
        break;
      }
      remain -= segLens[i];
    }
    // 충돌 시점 = 겨냥점(고스트)까지의 누적 거리.
    // (쿠션 걸어치기는 겨냥점이 경로 중간에 있으므로 첫 구간이 아님)
    var impactLen = segLens[0];
    final gh = _selected!.ghost;
    if (gh != null) {
      var acc = 0.0;
      var bestD = double.infinity;
      final gp = Offset(gh[0], gh[1]);
      for (var i = 0; i < pts.length; i++) {
        if (i > 0) acc += segLens[i - 1];
        final d = (pts[i] - gp).distance;
        if (d < bestD) {
          bestD = d;
          impactLen = acc;
        }
      }
    }
    // 1적구: 수구가 겨냥점에 도달한 뒤부터 진행
    Offset? objPos;
    final op = _selected!.objectPath;
    if (op.length >= 2) {
      final traveled = _anim.value * total - impactLen;
      if (traveled > 0) {
        final a = Offset(op[0][0], op[0][1]);
        final b = Offset(op[1][0], op[1][1]);
        final objLen = (b - a).distance;
        final f = (traveled / (objLen * 1.6)).clamp(0.0, 1.0);
        objPos = a + (b - a) * f;
      }
    }
    return (cuePos, objPos);
  }

  void _play() {
    if (_selected == null) return;
    _anim.forward(from: 0);
  }

  // 시점 자동 회전 (기획서 6.2): 치는 방향이 화면 위를 향하도록.
  double _angleFrom = 0;
  double _angleTarget = 0;

  void _setAngleFor(GuideShot? g) {
    var t = 0.0;
    if (g != null && g.feasible && g.cuePath.length >= 2) {
      final d = Offset(g.cuePath[1][0] - g.cuePath[0][0],
          g.cuePath[1][1] - g.cuePath[0][1]);
      // 세로 캔버스 좌표계에서의 진행 방향 (px=y, py=Wtot-x → (dy, -dx))
      final pd = Offset(d.dy, -d.dx);
      t = -math.pi / 2 - math.atan2(pd.dy, pd.dx);
    }
    // 최단 방향으로 회전하도록 정규화
    while (t - _angleTarget > math.pi) {
      t -= 2 * math.pi;
    }
    while (t - _angleTarget < -math.pi) {
      t += 2 * math.pi;
    }
    _angleFrom = _angleTarget;
    _angleTarget = t;
  }

  @override
  void dispose() {
    _cardController?.dispose();
    _tabController.dispose();
    _anim.dispose();
    super.dispose();
  }

  String _ballSummary() {
    final balls = widget.result.balls;
    if (balls.isEmpty) return '공 검출 안 됨';
    int count(String c) => balls.where((b) => b.color == c).length;
    return '공 ${balls.length}/4 — 흰 ${count('white')} · 노랑 ${count('yellow')} · 빨강 ${count('red')}'
        '${widget.result.partial ? '  ·  일부 촬영 역산' : ''}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D2137),
      appBar: AppBar(
        title: const Text('탑뷰 · 샷 가이드'),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        actions: [
          if (_hasBothCues)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: SegmentedButton<String>(
                showSelectedIcon: false,
                style: const ButtonStyle(visualDensity: VisualDensity.compact),
                segments: const [
                  ButtonSegment(value: 'white', label: Text('흰공')),
                  ButtonSegment(value: 'yellow', label: Text('노란공')),
                ],
                selected: {_cue},
                onSelectionChanged: (s) {
                  setState(() => _cue = s.first);
                  _loadGuides();
                },
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // 모델 당구대 (세로) + 공 + 가이드 오버레이 + 플레이 버튼
          Expanded(
            child: Stack(
              children: [
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: InteractiveViewer(
                    maxScale: 5,
                    child: Center(
                      child: AspectRatio(
                        aspectRatio: kPortraitWMm / kPortraitHMm,
                        child: TweenAnimationBuilder<double>(
                          key: ValueKey('rot-$_cue-${_selected?.id}'),
                          tween: Tween(begin: _angleFrom, end: _angleTarget),
                          duration: const Duration(milliseconds: 450),
                          curve: Curves.easeInOut,
                          builder: (_, angle, _) {
                            final (animCue, animObj) = _animPositionsMm();
                            return CustomPaint(
                              painter: TablePainter(
                                balls: widget.result.balls,
                                clothColor: widget.result.clothColor,
                                guide: _selected,
                                cueColor: _cue,
                                animCueMm: animCue,
                                animObjMm: animObj,
                                viewAngle: angle,
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                  ),
                ),
                // 플레이 버튼 (기획서 3.3: 상단 우측)
                if (_selected != null)
                  Positioned(
                    top: 4,
                    right: 8,
                    child: IconButton(
                      iconSize: 44,
                      tooltip: '공 움직임 재생',
                      icon: Icon(
                        _anim.isAnimating
                            ? Icons.stop_circle_outlined
                            : Icons.play_circle_fill,
                        color: Colors.amber,
                      ),
                      onPressed:
                          _anim.isAnimating ? () => _anim.reset() : _play,
                    ),
                  ),
              ],
            ),
          ),
          // 검출 요약 / 오류
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              children: [
                Text(_ballSummary(),
                    style: const TextStyle(color: Colors.white54, fontSize: 12)),
                if (_guideError.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(_guideError,
                        style: const TextStyle(
                            color: Colors.orangeAccent, fontSize: 13)),
                  ),
              ],
            ),
          ),
          // 직접치기 / 1쿠션 / 2쿠션 / 3쿠션 탭
          TabBar(
            controller: _tabController,
            labelColor: Colors.amber,
            unselectedLabelColor: Colors.white54,
            indicatorColor: Colors.amber,
            labelStyle: const TextStyle(fontSize: 13),
            tabs: const [
              Tab(text: '직접치기'),
              Tab(text: '1쿠션'),
              Tab(text: '2쿠션'),
              Tab(text: '3쿠션'),
            ],
          ),
          const SizedBox(height: 6),
          // 샷 카드 캐러셀 (기획서 6.1) — 스와이프하면 탑뷰에 경로가 즉시 반영
          if (_loading)
            const Padding(
              padding: EdgeInsets.all(8),
              child: CircularProgressIndicator(color: Colors.amber),
            )
          else if (_guides.isEmpty && _guideError.isEmpty)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Text('이 탭에서 가능한 길이 없어요',
                  style: TextStyle(color: Colors.white38)),
            )
          else if (_guides.isNotEmpty)
            SizedBox(
              height: 150,
              child: PageView.builder(
                // 탭별로 위젯을 새로 만들어 이전 탭의 페이지 위치가 남지 않게 함
                key: ValueKey('cards-${_categories[_tabController.index]}'),
                controller: _cardController,
                itemCount: _guides.length,
                onPageChanged: (i) {
                  final g = _guides[i];
                  _anim.reset();
                  final sel = g.feasible ? g : null;
                  _setAngleFor(sel);
                  setState(() => _selected = sel);
                },
                itemBuilder: (ctx, i) => _GuideCard(
                  guide: _guides[i],
                  selected: _selected?.id == _guides[i].id,
                  cueColor: _cue,
                  balls: widget.result.balls,
                  clothColor: widget.result.clothColor,
                ),
              ),
            ),
          // 다시 촬영하기
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(24, 8, 24, 12),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    backgroundColor: Colors.amber,
                    foregroundColor: Colors.black,
                  ),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('다시 촬영하기'),
                  onPressed: () => Navigator.pop(context),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// 샷 카드: 미니 경로 미리보기 + 샷 이름/정보 + 당점 스나이퍼 표시.
class _GuideCard extends StatelessWidget {
  final GuideShot guide;
  final bool selected;
  final String cueColor;
  final List<BallInfo> balls;
  final String clothColor;

  const _GuideCard({
    required this.guide,
    required this.selected,
    required this.cueColor,
    required this.balls,
    required this.clothColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF16324F),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: selected ? Colors.amber : Colors.white12,
          width: selected ? 2 : 1,
        ),
      ),
      child: guide.feasible ? _feasibleBody() : _infeasibleBody(),
    );
  }

  /// 5칸 등급 게이지 — 쉬울수록 초록, 어려울수록 빨강.
  static Widget gradeGauge(String label, int level) {
    const colors = [
      Color(0xFF34C759), // 1 초록
      Color(0xFF8BC34A),
      Color(0xFFFFC107),
      Color(0xFFFF9800),
      Color(0xFFFF3B30), // 5 빨강
    ];
    final idx = level.clamp(1, 5);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 34,
          child: Text(label,
              style: const TextStyle(color: Colors.white54, fontSize: 9)),
        ),
        for (var i = 0; i < 5; i++)
          Container(
            width: 10,
            height: 6,
            margin: const EdgeInsets.only(right: 2),
            decoration: BoxDecoration(
              color: i < idx ? colors[idx - 1] : Colors.white12,
              borderRadius: BorderRadius.circular(1.5),
            ),
          ),
      ],
    );
  }

  Widget _feasibleBody() {
    final cueBallColor =
        cueColor == 'white' ? Colors.white : const Color(0xFFF9C825);
    return Row(
      children: [
        // 미니 경로 미리보기 (세로 당구대 축소판)
        AspectRatio(
          aspectRatio: kPortraitWMm / kPortraitHMm,
          child: CustomPaint(
            painter: TablePainter(
              balls: balls,
              clothColor: clothColor,
              guide: guide,
              cueColor: cueColor,
              mini: true,
            ),
          ),
        ),
        const SizedBox(width: 8),
        // 샷 정보 + 등급 그래프
        Expanded(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(guide.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12.5,
                      fontWeight: FontWeight.bold)),
              const SizedBox(height: 5),
              gradeGauge('난이도', guide.difficulty),
              const SizedBox(height: 3),
              gradeGauge('키스', guide.kissLevel),
              const SizedBox(height: 4),
              Text('${guide.thicknessLabel} · ${guide.tip}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      color: Colors.amberAccent, fontSize: 10.5)),
            ],
          ),
        ),
        const SizedBox(width: 6),
        // 당점 + 두께 다이어그램 (세로 배치, 고정 크기 → 겹침 없음)
        Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 40,
              height: 40,
              child: CustomPaint(
                painter: TipPainter(
                  tipDeltaDeg: guide.tipDeltaDeg,
                  ballColor: cueBallColor,
                ),
              ),
            ),
            const Text('당점',
                style: TextStyle(color: Colors.white54, fontSize: 8.5)),
            const SizedBox(height: 4),
            SizedBox(
              width: 40,
              height: 40,
              child: CustomPaint(
                painter: ThicknessPainter(
                  thickness: guide.thickness,
                  aimOffset: guide.aimOffset,
                  cueBallColor: cueBallColor,
                ),
              ),
            ),
            const Text('두께',
                style: TextStyle(color: Colors.white54, fontSize: 8.5)),
          ],
        ),
      ],
    );
  }

  Widget _infeasibleBody() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text('${guide.name} — 불가',
              style: const TextStyle(color: Colors.white54, fontSize: 14)),
          const SizedBox(height: 6),
          Text(guide.reason,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white38, fontSize: 12)),
        ],
      ),
    );
  }
}

/// 당점 스나이퍼: 수구 단면 위에 쳐야 할 지점을 조준선으로 표시.
class TipPainter extends CustomPainter {
  final double tipDeltaDeg; // +위(밀어치기) / -아래(끌어치기) / 0 중단
  final Color ballColor;

  TipPainter({required this.tipDeltaDeg, required this.ballColor});

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.width / 2 * 0.92;

    // 수구 단면
    canvas.drawCircle(c, r, Paint()..color = ballColor);
    canvas.drawCircle(
        c, r,
        Paint()
          ..color = Colors.black45
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5);

    // 조준 십자선 (연한 회색)
    final cross = Paint()
      ..color = Colors.black26
      ..strokeWidth = 1;
    canvas.drawLine(c - Offset(r, 0), c + Offset(r, 0), cross);
    canvas.drawLine(c - Offset(0, r), c + Offset(0, r), cross);

    // 당점 (보정각 ±40° → 공 반지름의 ±0.7 위치로 스케일)
    final dy = -(tipDeltaDeg / 40.0) * r * 0.7;
    final target = c + Offset(0, dy);
    final red = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawCircle(target, r * 0.28, red);
    canvas.drawCircle(target, r * 0.09, Paint()..color = Colors.redAccent);
    // 스나이퍼 눈금 (타깃 원 밖 십자)
    for (final d in [Offset(r * 0.42, 0), Offset(-r * 0.42, 0),
                     Offset(0, r * 0.42), Offset(0, -r * 0.42)]) {
      canvas.drawLine(target + d * 0.65, target + d, red);
    }
  }

  @override
  bool shouldRepaint(TipPainter old) =>
      old.tipDeltaDeg != tipDeltaDeg || old.ballColor != ballColor;
}

/// 두께(타점) 다이어그램: 1적구를 정면에서 봤을 때 수구를 얼마나
/// 겹쳐 맞출지를 표시. aimOffset(+오른쪽, 공 지름 단위)만큼 수구
/// 실루엣을 비껴 그려서 겹침 정도(두께)가 한눈에 보이게 한다.
class ThicknessPainter extends CustomPainter {
  final double thickness;   // 0~1 (1=정면)
  final double aimOffset;   // 공 지름 단위 가로 오프셋 (+오른쪽)
  final Color cueBallColor;

  ThicknessPainter({
    required this.thickness,
    required this.aimOffset,
    required this.cueBallColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.width / 2 * 0.55; // 공 반지름 (두 공이 다 들어가게 축소)

    // 1적구 (빨강, 뒤)
    canvas.drawCircle(c, r, Paint()..color = const Color(0xFFD32F2F));
    canvas.drawCircle(
        c, r,
        Paint()
          ..color = Colors.black38
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1);

    // 수구 실루엣 (반투명, 겹침 = 두께)
    final cueCenter = c + Offset(aimOffset * 2 * r, 0);
    canvas.drawCircle(
        cueCenter, r, Paint()..color = cueBallColor.withValues(alpha: 0.55));
    canvas.drawCircle(
        cueCenter, r,
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5);
    // 겨냥 중심선
    canvas.drawLine(
      Offset(cueCenter.dx, c.dy - r * 1.35),
      Offset(cueCenter.dx, c.dy + r * 1.35),
      Paint()
        ..color = Colors.amberAccent
        ..strokeWidth = 1.2,
    );
  }

  @override
  bool shouldRepaint(ThicknessPainter old) =>
      old.thickness != thickness ||
      old.aimOffset != aimOffset ||
      old.cueBallColor != cueBallColor;
}

/// 규격 모델 당구대를 세로로 그리고 공/가이드를 표시하는 페인터.
class TablePainter extends CustomPainter {
  final List<BallInfo> balls;
  final String clothColor; // 'blue' | 'green'
  final GuideShot? guide;
  final String cueColor;
  final bool mini; // 카드 미리보기용 (다이아몬드 생략, 선 두껍게)
  final Offset? animCueMm; // 플레이 애니메이션 중 수구 위치 (mm)
  final Offset? animObjMm; // 플레이 애니메이션 중 1적구 위치 (mm)
  final double viewAngle;  // 시점 회전 (라디안, 치는 방향이 위로 가도록)

  TablePainter({
    required this.balls,
    required this.clothColor,
    this.guide,
    required this.cueColor,
    this.mini = false,
    this.animCueMm,
    this.animObjMm,
    this.viewAngle = 0,
  });

  double _scale(Size size) => size.width / kPortraitWMm;

  Offset _map(double xMm, double yMm, Size size) {
    final p = worldToPortraitMm(xMm, yMm);
    final s = _scale(size);
    return Offset(p.dx * s, p.dy * s);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final s = _scale(size);

    // 시점 회전: 중앙 기준 회전 + 화면 안에 들어오도록 축소
    if (viewAngle != 0) {
      final cx = size.width / 2, cy = size.height / 2;
      final c = math.cos(viewAngle).abs();
      final si = math.sin(viewAngle).abs();
      final rw = size.width * c + size.height * si;
      final rh = size.width * si + size.height * c;
      final fit = math.min(size.width / rw, size.height / rh);
      canvas.translate(cx, cy);
      canvas.rotate(viewAngle);
      canvas.scale(fit);
      canvas.translate(-cx, -cy);
    }

    // 목재 레일 (전체 배경)
    final railPaint = Paint()..color = const Color(0xFF4E342E);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
          Offset.zero & size, Radius.circular(40 * s)),
      railPaint,
    );

    // 천 (경기면 + 쿠션 오버행)
    final cloth = clothColor == 'green'
        ? const Color(0xFF1B7A3D)
        : const Color(0xFF1565C0);
    final clothDark = clothColor == 'green'
        ? const Color(0xFF14602F)
        : const Color(0xFF104E92);
    final ovTl = _map(kTableWMm + kCushionOverhangMm, -kCushionOverhangMm, size);
    final ovBr = _map(-kCushionOverhangMm, kTableHMm + kCushionOverhangMm, size);
    canvas.drawRect(Rect.fromPoints(ovTl, ovBr), Paint()..color = clothDark);

    // 경기면
    final pfTl = _map(kTableWMm, 0, size);
    final pfBr = _map(0, kTableHMm, size);
    final pfRect = Rect.fromPoints(pfTl, pfBr);
    canvas.drawRect(pfRect, Paint()..color = cloth);

    // 다이아몬드 (장쿠션 9개×2, 단쿠션 5개×2 — 시작점 포함) — 미니 모드 생략
    if (!mini) {
      final diamondPaint = Paint()..color = Colors.white;
      for (int i = 0; i <= 8; i++) {
        final x = i * kDiamondSpacingMm;
        canvas.drawCircle(
            _map(x, -kDiamondOffsetMm, size), 7 * s, diamondPaint);
        canvas.drawCircle(
            _map(x, kTableHMm + kDiamondOffsetMm, size), 7 * s, diamondPaint);
      }
      for (int j = 0; j <= 4; j++) {
        final y = j * kDiamondSpacingMm;
        canvas.drawCircle(
            _map(-kDiamondOffsetMm, y, size), 7 * s, diamondPaint);
        canvas.drawCircle(
            _map(kTableWMm + kDiamondOffsetMm, y, size), 7 * s, diamondPaint);
      }
    }

    // 가이드 (공 아래에 깔리지 않게 공보다 먼저 경로만)
    if (guide != null) {
      _paintGuide(canvas, size, guide!);
    }

    // 공 (애니메이션 중이면 수구/1적구는 진행 위치에 그림)
    int cueIdx = balls.indexWhere((b) => b.color == cueColor);
    int objIdx = -1;
    if (guide != null && guide!.objectPath.isNotEmpty) {
      double bestD = double.infinity;
      for (var i = 0; i < balls.length; i++) {
        final d = (Offset(balls[i].xMm, balls[i].yMm) -
                Offset(guide!.objectPath[0][0], guide!.objectPath[0][1]))
            .distance;
        if (d < bestD) {
          bestD = d;
          objIdx = i;
        }
      }
    }
    for (var i = 0; i < balls.length; i++) {
      final b = balls[i];
      var xMm = b.xMm, yMm = b.yMm;
      if (i == cueIdx && animCueMm != null) {
        xMm = animCueMm!.dx;
        yMm = animCueMm!.dy;
      } else if (i == objIdx && animObjMm != null) {
        xMm = animObjMm!.dx;
        yMm = animObjMm!.dy;
      }
      final c = _map(xMm, yMm, size);
      final r = kBallRadiusMm * s;
      final color = switch (b.color) {
        'white' => Colors.white,
        'yellow' => const Color(0xFFF9C825),
        _ => const Color(0xFFD32F2F),
      };
      canvas.drawCircle(c + Offset(2 * s, 3 * s), r,
          Paint()..color = Colors.black.withValues(alpha: 0.3));
      canvas.drawCircle(c, r, Paint()..color = color);
      canvas.drawCircle(
          c - Offset(r * 0.3, r * 0.35), r * 0.25,
          Paint()..color = Colors.white.withValues(alpha: 0.5));
    }
  }

  void _paintGuide(Canvas canvas, Size size, GuideShot g) {
    final s = _scale(size);
    final lineW = mini ? 40.0 : 12.0; // 미니 카드에서도 경로가 보이도록
    final cuePaint = Paint()
      ..color = cueColor == 'white' ? Colors.white : const Color(0xFFF9C825)
      ..strokeWidth = lineW * s
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    if (g.cuePath.length >= 2) {
      final pts = g.cuePath.map((p) => _map(p[0], p[1], size)).toList();
      final path = Path()..moveTo(pts.first.dx, pts.first.dy);
      for (final p in pts.skip(1)) {
        path.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(path, cuePaint);
    }

    if (g.ghost != null) {
      canvas.drawCircle(
        _map(g.ghost![0], g.ghost![1], size),
        kBallRadiusMm * s,
        Paint()
          ..color = Colors.white70
          ..strokeWidth = 6 * s
          ..style = PaintingStyle.stroke,
      );
    }

    if (g.objectPath.length >= 2) {
      final a = _map(g.objectPath[0][0], g.objectPath[0][1], size);
      final b = _map(g.objectPath[1][0], g.objectPath[1][1], size);
      final objPaint = Paint()
        ..color = Colors.redAccent
        ..strokeWidth = (mini ? 35 : 10) * s
        ..strokeCap = StrokeCap.round;
      canvas.drawLine(a, b, objPaint);
      final dir = b - a;
      final len = dir.distance;
      if (len > 0) {
        final u = dir / len;
        final n = Offset(-u.dy, u.dx);
        final tip = b;
        final arrow = 45.0 * s;
        canvas.drawLine(tip, tip - u * arrow + n * arrow * 0.6, objPaint);
        canvas.drawLine(tip, tip - u * arrow - n * arrow * 0.6, objPaint);
      }
    }
  }

  @override
  bool shouldRepaint(TablePainter old) =>
      old.guide?.id != guide?.id ||
      old.cueColor != cueColor ||
      old.clothColor != clothColor ||
      old.mini != mini ||
      old.animCueMm != animCueMm ||
      old.animObjMm != animObjMm ||
      old.viewAngle != viewAngle ||
      old.balls != balls;
}
