const { getDefaultConfig } = require('expo/metro-config');

module.exports = (() => {
  const config = getDefaultConfig(__dirname);
  config.resolver.assetExts.push('tflite', 'txt', 'jpeg', 'png', 'jpg', 'json', 'onnx');
  return config;
})();
