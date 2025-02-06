import React, { useState, useEffect, useRef } from "react";
import { StyleSheet, Text, TouchableOpacity, View, Dimensions } from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import * as ImageManipulator from "expo-image-manipulator";
import { Asset } from "expo-asset";
import Svg, { Rect, Text as SvgText } from 'react-native-svg';

const CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush'
];

const COLORS = ['#FF3B30', '#34C759', '#007AFF', '#5856D6', '#FF9500'];

interface Detection {
  class: number;
  confidence: number;
  bbox: number[];
}

export default function App() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [modelLoading, setModelLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const cameraRef = useRef<any>(null);

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      console.log("Loading YOLO model...");
      const modelAsset = Asset.fromModule(require("./assets/yolov8n_with_pre_post_processing.onnx"));
      await modelAsset.downloadAsync();
      const session = await InferenceSession.create(modelAsset.localUri || modelAsset.uri);
      setSession(session);
      setModelLoading(false);
      console.log("Model loaded successfully");
    } catch (err) {
      console.error("Failed to load model:", err);
      setError("Failed to load model");
      setModelLoading(false);
    }
  };

  const runDetection = async () => {
    if (!session || !cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setError(null);

      // Capture photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1,
        base64: true,
        skipProcessing: true
      });

      // Resize image to a reasonable size first
      const resized = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 640, height: 640 } }],
        { format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );

      if (!resized.base64) throw new Error("Failed to process image");

      // Convert base64 to byte array
      const imageData = atob(resized.base64);
      const imageBytes = new Uint8Array(imageData.length);
      for (let i = 0; i < imageData.length; i++) {
        imageBytes[i] = imageData.charCodeAt(i);
      }

      // Create input tensor with correct shape for image
      const inputTensor = new Tensor(
        "uint8",
        imageBytes,
        [1, imageBytes.length]  // Add batch dimension
      );

      // Run inference with the preprocessed model
      const results = await session.run({
        "image": inputTensor
      });

      if (!results.output0?.data) {
        throw new Error("No detection results");
      }

      // Process detection boxes
      const output = results.output0.data as Float32Array;
      const detections: Detection[] = [];

      // Each detection has format [x1, y1, x2, y2, score, class_id]
      const valuesPerBox = 6;  // 4 coords + score + class_id
      const numDetections = output.length / valuesPerBox;

      for (let i = 0; i < numDetections; i++) {
        const offset = i * valuesPerBox;
        const confidence = output[offset + 4];
        const classId = Math.floor(output[offset + 5]);
        
        // Skip if invalid class
        if (classId < 0 || classId >= CLASSES.length) continue;

        detections.push({
          bbox: [
            output[offset],     // x1 (already normalized)
            output[offset + 1], // y1
            output[offset + 2], // x2
            output[offset + 3]  // y2
          ],
          confidence: confidence,
          class: classId
        });
      }

      console.log(`Found ${detections.length} objects`);
      setDetections(detections);

    } catch (err) {
      console.error("Detection failed:", err);
      setError("Detection failed: " + (err as Error).message);
    } finally {
      setIsProcessing(false);
    }
  };

  const renderBoxes = () => {
    const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

    return (
      <Svg style={StyleSheet.absoluteFill}>
        {detections.map((det, idx) => {
          const [x1, y1, x2, y2] = det.bbox;
          const color = COLORS[idx % COLORS.length];
          
          // Convert normalized coordinates to screen space
          const boxX = x1 * screenWidth;
          const boxY = y1 * screenHeight;
          const boxWidth = (x2 - x1) * screenWidth;
          const boxHeight = (y2 - y1) * screenHeight;

          return (
            <React.Fragment key={idx}>
              <Rect
                x={boxX}
                y={boxY}
                width={boxWidth}
                height={boxHeight}
                strokeWidth={2}
                stroke={color}
                fill="none"
              />
              <SvgText
                x={boxX}
                y={boxY - 5}
                fill={color}
                fontSize="14"
                fontWeight="bold"
              >
                {`${CLASSES[det.class]} ${(det.confidence * 100).toFixed(0)}%`}
              </SvgText>
            </React.Fragment>
          );
        })}
      </Svg>
    );
  };

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <TouchableOpacity onPress={requestPermission}>
          <Text style={styles.text}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {modelLoading ? (
        <Text style={styles.text}>Loading model...</Text>
      ) : (
        <>
          <CameraView
            style={styles.camera}
            facing={facing}
            ref={cameraRef}
          />
          {renderBoxes()}
          <View style={styles.controls}>
            <TouchableOpacity
              style={styles.button}
              onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
            >
              <Text style={styles.buttonText}>Flip</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.button, isProcessing && styles.buttonDisabled]}
              onPress={runDetection}
              disabled={isProcessing}
            >
              <Text style={styles.buttonText}>
                {isProcessing ? 'Processing...' : 'Detect'}
              </Text>
            </TouchableOpacity>
          </View>
          {error && (
            <Text style={styles.error}>{error}</Text>
          )}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  controls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  button: {
    backgroundColor: '#4CAF50',
    padding: 15,
    borderRadius: 30,
    width: 100,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#888888',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  text: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    margin: 20,
  },
  error: {
    position: 'absolute',
    top: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(255,0,0,0.7)',
    padding: 10,
    borderRadius: 5,
    color: 'white',
    textAlign: 'center',
  }
});