package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.tasks.vision.core.RunningMode;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    private PreviewView previewView;
    private boolean hasPredictedOnce = false;
    private Button startButton, stopButton;
    private TextView predictionText;
    private boolean isPredicting = false;
    private ExecutorService cameraExecutor;
    private List<Bitmap> lipFrames = new ArrayList<>();
    private Interpreter tflite;
    private LipExtractor lipExtractor;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                    loadModel();
                    setupLipExtractor();
                } else {
                    Toast.makeText(MainActivity.this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        predictionText = findViewById(R.id.predictionText);

        cameraExecutor = Executors.newSingleThreadExecutor();

        startButton.setOnClickListener(v -> {
            isPredicting = true;
            hasPredictedOnce = false; // Reset prediction flag
            predictionText.setText("Prediction started...");
            lipFrames.clear();
        });

        stopButton.setOnClickListener(v -> {
            isPredicting = false;
            predictionText.setText("Prediction stopped.");
            lipFrames.clear();
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
            loadModel();
            setupLipExtractor();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreviewAndAnalysis(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreviewAndAnalysis(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, imageProxy -> {
            if (isPredicting && !hasPredictedOnce) {  // Only predict once
                Bitmap frameBitmap = ImageUtils.imageToBitmap(imageProxy);
                Bitmap lipBitmap = lipExtractor.extractLip(frameBitmap);
                if (lipBitmap != null) {
                    Bitmap processed = preprocessFrame(lipBitmap);
                    lipFrames.add(processed);
                    if (lipFrames.size() == 30) {
                        runInference();
                        hasPredictedOnce = true; // Mark prediction done
                        lipFrames.clear();
                    }
                } else {
                    Log.d("LipReading", "No lip detected in frame.");
                }
            }
            imageProxy.close();
        });


        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void setupLipExtractor() {
        lipExtractor = new LipExtractor(this, RunningMode.IMAGE); // or LIVE_STREAM if supported
    }

    private void loadModel() {
        try {
            AssetFileDescriptor fileDescriptor = getAssets().openFd("lip_reading_model_vowels.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            Interpreter.Options options = new Interpreter.Options();
            tflite = new Interpreter(modelBuffer, options);
            Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
        }
    }

    private Bitmap preprocessFrame(Bitmap bmp) {
        // Create an output bitmap of the same size
        Bitmap grayBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);

        // Convert to grayscale manually
        for (int y = 0; y < bmp.getHeight(); y++) {
            for (int x = 0; x < bmp.getWidth(); x++) {
                int pixel = bmp.getPixel(x, y);

                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;

                // Calculate grayscale using standard formula
                int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);

                int newPixel = 0xFF << 24 | (gray << 16) | (gray << 8) | gray;

                grayBmp.setPixel(x, y, newPixel);
            }
        }

        // Scale to 64x64 (model input size)
        return Bitmap.createScaledBitmap(grayBmp, 64, 64, true);
    }


    private void runInference() {
        if (lipFrames.size() != 30) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this, "Not enough frames for inference", Toast.LENGTH_SHORT).show());
            return;
        }

        float[][][][][] input = new float[1][30][64][64][1];
        for (int f = 0; f < 30; f++) {
            Bitmap bmp = lipFrames.get(f);
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 64; j++) {
                    int pixel = bmp.getPixel(j, i);
                    int gray = pixel & 0xFF;
                    input[0][f][i][j][0] = gray / 255.0f;
                }
            }
        }

        String[] labels = {"HELLO", "BYE", "THANKS", "A", "E", "I", "O", "U"};
        float[][] output = new float[1][labels.length];
        try {
            tflite.run(input, output);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        Log.d("LipReading", "Model output: " + Arrays.toString(output[0]));

        int maxIndex = 0;
        float maxProb = output[0][0];
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }

        StringBuilder sb = new StringBuilder("Probabilities: ");
        for (int i = 0; i < labels.length; i++) {
            sb.append(labels[i]).append(": ")
                    .append(String.format("%.2f", output[0][i] * 100)).append("%, ");
        }
        Log.d("LipReading", sb.toString());

        String prediction = labels[maxIndex] + " (" + String.format("%.2f", maxProb * 100) + "%)";
        Log.d("LipReading", "Prediction: " + prediction);

        runOnUiThread(() -> {
            predictionText.setText("Prediction: " + prediction);
            Toast.makeText(MainActivity.this, "Prediction: " + prediction, Toast.LENGTH_SHORT).show();
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        if (lipExtractor != null) {
            lipExtractor.close();
        }
    }
}

