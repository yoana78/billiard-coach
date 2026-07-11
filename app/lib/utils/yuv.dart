import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// CameraImage(YUV420)를 isolate 로 넘기기 위한 순수 데이터 구조.
class YuvFrame {
  final int width;
  final int height;
  final Uint8List y;
  final Uint8List u;
  final Uint8List v;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;

  YuvFrame({
    required this.width,
    required this.height,
    required this.y,
    required this.u,
    required this.v,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
  });
}

/// YUV420 프레임을 다운스케일하며 RGB 변환 후 JPEG 인코딩.
/// 프리뷰 분석용이므로 [downscale]=2 로 픽셀 수를 1/4로 줄여 속도 확보.
Uint8List yuvFrameToJpeg(YuvFrame f, {int downscale = 2, int quality = 80}) {
  final ow = f.width ~/ downscale;
  final oh = f.height ~/ downscale;
  final out = img.Image(width: ow, height: oh);

  for (int oy = 0; oy < oh; oy++) {
    final sy = oy * downscale;
    for (int ox = 0; ox < ow; ox++) {
      final sx = ox * downscale;
      final yv = f.y[sy * f.yRowStride + sx];
      final uvIndex = (sy ~/ 2) * f.uvRowStride + (sx ~/ 2) * f.uvPixelStride;
      final uv = f.u[uvIndex] - 128;
      final vv = f.v[uvIndex] - 128;

      int r = (yv + 1.402 * vv).round();
      int g = (yv - 0.344136 * uv - 0.714136 * vv).round();
      int b = (yv + 1.772 * uv).round();
      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);
      out.setPixelRgb(ox, oy, r, g, b);
    }
  }
  return Uint8List.fromList(img.encodeJpg(out, quality: quality));
}
