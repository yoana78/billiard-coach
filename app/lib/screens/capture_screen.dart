import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../config.dart';
import '../services/api.dart';
import '../utils/yuv.dart';
import 'result_screen.dart';

/// isolate 에서 실행할 변환 함수 (compute 용 최상위 진입점).
Uint8List _convertFrame(YuvFrame f) => yuvFrameToJpeg(f);

/// [촬영] — 실시간 검출 + 피드백 (기획서 3.2 B안).
/// 프리뷰 프레임을 주기적으로 서버 /analyze 에 보내 상태를 오버레이로 표시,
/// 조건 충족이 3회 연속 유지되면 자동 촬영 → /topview → 결과 화면.
class CaptureScreen extends StatefulWidget {
  final String table; // 'medium' | 'large'
  final String game;  // 'four' | 'three'

  const CaptureScreen({super.key, this.table = 'medium', this.game = 'four'});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _camera;
  ApiClient? _api;
  String _initError = '';

  bool _analyzing = false; // /analyze 요청 진행 중
  bool _capturing = false; // 촬영/업로드 진행 중
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);
  static const _minInterval = Duration(milliseconds: 700);

  AnalyzeResult? _last;
  bool _serverReachable = true;
  bool _autoCapture = false; // 기본은 수동 촬영 (원하는 각도에서 직접)
  static const _stableCountForAuto = 3;
  int _readyStreak = 0;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final url = await AppConfig.serverUrl();
      _api = ApiClient(url);
      final cams = await availableCameras();
      final back = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );
      final cam = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await cam.initialize();
      await cam.startImageStream(_onFrame);
      if (!mounted) {
        await cam.dispose();
        return;
      }
      setState(() => _camera = cam);
    } catch (e) {
      if (mounted) setState(() => _initError = '카메라 초기화 실패: $e');
    }
  }

  Future<void> _onFrame(CameraImage image) async {
    if (_analyzing || _capturing || _api == null) return;
    if (DateTime.now().difference(_lastSent) < _minInterval) return;
    _analyzing = true;
    _lastSent = DateTime.now();
    try {
      final frame = YuvFrame(
        width: image.width,
        height: image.height,
        y: image.planes[0].bytes,
        u: image.planes[1].bytes,
        v: image.planes[2].bytes,
        yRowStride: image.planes[0].bytesPerRow,
        uvRowStride: image.planes[1].bytesPerRow,
        uvPixelStride: image.planes[1].bytesPerPixel ?? 1,
      );
      final jpeg = await compute(_convertFrame, frame);
      final result = await _api!.analyzeFrame(jpeg);
      if (!mounted || _capturing) return;
      setState(() {
        _serverReachable = true;
        _last = result;
        _readyStreak = result.ready ? _readyStreak + 1 : 0;
      });
      // 자동촬영은 옵션일 때만 (기본 off — 사용자가 각도를 직접 잡도록)
      if (_autoCapture && _readyStreak >= _stableCountForAuto) {
        unawaited(_capture());
      }
    } catch (_) {
      if (mounted) setState(() => _serverReachable = false);
    } finally {
      _analyzing = false;
    }
  }

  Future<void> _stopStream() async {
    final cam = _camera;
    if (cam != null && cam.value.isStreamingImages) {
      await cam.stopImageStream();
    }
  }

  Future<void> _resumeStream() async {
    final cam = _camera;
    if (cam != null && !cam.value.isStreamingImages) {
      try {
        await cam.startImageStream(_onFrame);
      } catch (_) {}
    }
  }

  /// 이미지 바이트를 서버로 보내 탑뷰 변환 → 결과 화면. 실패 시 프리뷰 재개.
  Future<void> _processBytes(Uint8List bytes) async {
    if (_api == null) return;
    try {
      final result = await _api!.topview(bytes, table: widget.table);
      if (!mounted) return;
      if (result.ok && result.imagePng != null) {
        await Navigator.pushReplacement(
          context,
          MaterialPageRoute(
              builder: (_) => ResultScreen(
                  result: result,
                  table: widget.table,
                  game: widget.game)),
        );
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('변환 실패: ${result.reason} — 다시 시도해주세요')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('업로드 오류: $e')),
      );
    }
    await _resumeStream();
    if (mounted) setState(() => _capturing = false);
  }

  /// 촬영 직전 카메라 노출을 어둡게 낮춘다 (빛반사 저감 → 다이아몬드 검출↑).
  /// 당구장 조명은 레일에 강한 반사를 만들어 흰 다이아몬드 점을 하얗게
  /// 날려버리는데, 노출을 내리면 레일이 어두워지고 점 대비가 살아난다.
  Future<double> _applyLowExposure(CameraController cam) async {
    try {
      final minEv = await cam.getMinExposureOffset();
      if (minEv >= 0) return 0;
      // 최소 노출의 60% 지점 (완전 최소는 색 인식이 불리 → 절충)
      final target = (minEv * 0.6).clamp(minEv, 0.0);
      final applied = await cam.setExposureOffset(target);
      // 노출 반영에 약간의 시간 필요
      await Future.delayed(const Duration(milliseconds: 250));
      return applied;
    } catch (_) {
      return 0;
    }
  }

  Future<void> _capture() async {
    final cam = _camera;
    if (cam == null || _capturing || _api == null) return;
    setState(() {
      _capturing = true;
      _readyStreak = 0;
    });
    try {
      await _stopStream();
      await _applyLowExposure(cam);
      final shot = await cam.takePicture();
      // 노출 원복 (프리뷰 재개 시 정상 밝기)
      try {
        await cam.setExposureOffset(0);
      } catch (_) {}
      final bytes = await shot.readAsBytes();
      await _processBytes(bytes);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('촬영 오류: $e')),
      );
      try {
        await cam.setExposureOffset(0);
      } catch (_) {}
      await _resumeStream();
      setState(() => _capturing = false);
    }
  }

  /// 갤러리에서 저장된 사진을 불러와 변환 (테스트/실사진 검증용).
  Future<void> _pickFromGallery() async {
    if (_capturing || _api == null) return;
    setState(() {
      _capturing = true;
      _readyStreak = 0;
    });
    await _stopStream();
    final picked = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (picked == null) {
      // 선택 취소 → 프리뷰 재개
      await _resumeStream();
      if (mounted) setState(() => _capturing = false);
      return;
    }
    final bytes = await picked.readAsBytes();
    await _processBytes(bytes);
  }

  @override
  void dispose() {
    _camera?.dispose();
    super.dispose();
  }

  Color get _guideColor {
    if (_last == null || !_serverReachable) return Colors.red;
    if (_last!.ready) return Colors.greenAccent;
    if (_last!.tableFound) return Colors.orangeAccent;
    return Colors.red;
  }

  String get _statusMessage {
    if (!_serverReachable) return '서버에 연결할 수 없어요 (설정에서 주소 확인)';
    if (_last == null) return '당구대를 화면에 맞춰주세요';
    return _last!.message;
  }

  @override
  Widget build(BuildContext context) {
    if (_initError.isNotEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('촬영')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(_initError, textAlign: TextAlign.center),
          ),
        ),
      );
    }
    final cam = _camera;
    if (cam == null || !cam.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final ready = _last?.ready == true;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(child: CameraPreview(cam)),
          // 가이드 오버레이 (테두리 색으로 상태 표시)
          IgnorePointer(
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: _guideColor, width: 4),
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
          // 상단 안내 문구
          SafeArea(
            child: Column(
              children: [
                Container(
                  margin: const EdgeInsets.all(24),
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Text(
                    _statusMessage,
                    style: TextStyle(color: _guideColor, fontSize: 16),
                  ),
                ),
                if (_last != null && _last!.tableFound)
                  Text(
                    '다이아몬드  장쿠션 ${_last!.diamondLong}/18 · 단쿠션 ${_last!.diamondShort}/10',
                    style: const TextStyle(color: Colors.white70, fontSize: 13),
                  ),
              ],
            ),
          ),
          // 하단 촬영/갤러리 버튼
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: _capturing
                  ? const Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: Colors.amber),
                        SizedBox(height: 12),
                        Text('탑뷰 변환 중...', style: TextStyle(color: Colors.white)),
                      ],
                    )
                  : Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // 갤러리에서 불러오기 (저장된 사진으로 테스트)
                        IconButton.filledTonal(
                          onPressed: _pickFromGallery,
                          tooltip: '저장된 사진 불러오기',
                          iconSize: 28,
                          icon: const Icon(Icons.photo_library),
                        ),
                        const SizedBox(width: 32),
                        // 셔터: 항상 누를 수 있음 (원하는 각도에서 직접 촬영).
                        // 초록이면 인식 조건 충족(권장), 아니어도 촬영 가능.
                        FloatingActionButton.large(
                          backgroundColor:
                              ready ? Colors.greenAccent : Colors.white,
                          onPressed: _capture,
                          child: const Icon(Icons.camera_alt,
                              color: Colors.black),
                        ),
                        const SizedBox(width: 32),
                        // 자동촬영 토글
                        IconButton.filledTonal(
                          onPressed: () =>
                              setState(() => _autoCapture = !_autoCapture),
                          tooltip: _autoCapture ? '자동촬영 켜짐' : '자동촬영 꺼짐',
                          iconSize: 28,
                          isSelected: _autoCapture,
                          icon: Icon(_autoCapture
                              ? Icons.motion_photos_on
                              : Icons.motion_photos_off),
                        ),
                      ],
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
