import 'package:shared_preferences/shared_preferences.dart';

/// 서버 주소 설정. 기본값은 Render 클라우드 고정 주소 (PC 안 켜도 동작).
/// 로컬 개발 시에는 인트로 톱니바퀴에서 PC LAN IP로 변경
/// (예: http://192.168.219.102:8000).
class AppConfig {
  static const String defaultServerUrl = 'https://billiard-coach.onrender.com';
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
