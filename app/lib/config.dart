import 'package:shared_preferences/shared_preferences.dart';

/// 서버 주소 설정. 기본값은 Cloudflare 터널 공개 주소 (외부에서 접속 가능).
/// 터널이 재시작되면 주소가 바뀌므로, 그때는 인트로 톱니바퀴에서 변경.
/// 같은 Wi-Fi 개발 시에는 PC LAN IP (예: http://192.168.219.102:8000).
class AppConfig {
  static const String defaultServerUrl =
      'https://frog-compete-enough-golf.trycloudflare.com';
  static const String _key = 'server_url';

  static Future<String> serverUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_key) ?? defaultServerUrl;
  }

  static Future<void> setServerUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, url);
  }
}
