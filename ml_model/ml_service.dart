import 'package:tflite/tflite.dart';

class MLService {
  Future<void> loadModel() async {
    await Tflite.loadModel(
      model: "assets/model.tflite",
      labels: "assets/labels.txt",
    );
  }

  Future<List?> classifyImage(String imagePath) async {
    return await Tflite.runModelOnImage(
      path: imagePath,
      imageMean: 127.5,
      imageStd: 127.5,
      numResults: 5,
      threshold: 0.5,
      asynch: true,
    );
  }

  void disposeModel() {
    Tflite.close();
  }
}
