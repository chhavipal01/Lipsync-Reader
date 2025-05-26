package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Bitmap.Config;
import com.google.mediapipe.framework.image.BitmapImageBuilder;


import java.util.List;

public class LipExtractor {
    private static final String TAG = "LipExtractor";
    private final FaceLandmarker faceLandmarker;

    public LipExtractor(Context context, RunningMode liveStream) {
        FaceLandmarkerOptions options = FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .build();

        try {
            faceLandmarker = FaceLandmarker.createFromOptions(context, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Bitmap extractLip(Bitmap bitmap) {
        MPImage mpImage = new BitmapImageBuilder(bitmap).build();
        FaceLandmarkerResult result = faceLandmarker.detect(mpImage);

        if (result != null && !result.faceLandmarks().isEmpty()) {
            List<NormalizedLandmark> landmarks = result.faceLandmarks().get(0);
            return extractLipFromLandmarks(landmarks, bitmap);
        }
        return null;
    }

    // Keep the rest of the extractLipFromLandmarks method from previous answer
    // ...
    private Bitmap extractLipFromLandmarks(List<NormalizedLandmark> landmarks, Bitmap originalBitmap) {
        // Lip landmark indices in FaceMesh
        int[] lipIndices = {
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 308, 324, 318, 402, 317, 14, 87, 178, 88
        };

        float minX = 1.0f, minY = 1.0f, maxX = 0.0f, maxY = 0.0f;
        for (int index : lipIndices) {
            NormalizedLandmark point = landmarks.get(index);
            float x = point.x();
            float y = point.y();
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        // Convert normalized coordinates to pixel values
        int width = originalBitmap.getWidth();
        int height = originalBitmap.getHeight();
        int left = (int) (minX * width);
        int top = (int) (minY * height);
        int right = (int) (maxX * width);
        int bottom = (int) (maxY * height);

        // Make sure the rectangle is within the image bounds
        left = Math.max(0, left);
        top = Math.max(0, top);
        right = Math.min(width, right);
        bottom = Math.min(height, bottom);

        return Bitmap.createBitmap(originalBitmap, left, top, right - left, bottom - top);
    }
    public void close() {
        if (faceLandmarker != null) {
            faceLandmarker.close();
        }
    }
}