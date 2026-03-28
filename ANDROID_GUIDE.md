# Android Integration Guide — Zerotwo Wake Word

## 1. Add ONNX Runtime Dependency

In your `build.gradle (app)`:
```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
}
```

## 2. Place Model in Assets

```
app/
└── src/main/assets/
    └── zerotwo_v1.onnx   ← copy from models/zerotwo_v1.onnx
```

## 3. AudioRecord Configuration

```kotlin
val SAMPLE_RATE    = 16000
val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
val AUDIO_FORMAT   = AudioFormat.ENCODING_PCM_16BIT
val BUFFER_FRAMES  = SAMPLE_RATE  // 1 second buffer

val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
    .coerceAtLeast(BUFFER_FRAMES * 2)

val audioRecord = AudioRecord(
    MediaRecorder.AudioSource.VOICE_RECOGNITION,  // bypass AGC/noise suppression
    SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize
)
```

## 4. Model Input Preprocessing (Kotlin)

```kotlin
import ai.onnxruntime.*

class WakeWordDetector(context: Context) {
    private val N_MELS    = 80
    private val N_FRAMES  = 100   // 1 second at 16kHz, hop=160
    private val HOP_LEN   = 160
    private val WIN_LEN   = 400
    private val N_FFT     = 512
    private val THRESHOLD = 0.80f
    private val CONFIRM_N = 3

    private val session: OrtSession
    private val env = OrtEnvironment.getEnvironment()
    private var confirmCount = 0

    init {
        val modelBytes = context.assets.open("zerotwo_v1.onnx").readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    /**
     * Call this with a 1-second (16000 samples) float32 PCM buffer.
     * Returns true if wake word is detected with 3-frame confirmation.
     */
    fun process(pcmFloat: FloatArray): Boolean {
        val mel = extractMelSpectrogram(pcmFloat)  // shape: [1, 1, 80, 100]
        val tensor = OnnxTensor.createTensor(env, mel, longArrayOf(1, 1, N_MELS.toLong(), N_FRAMES.toLong()))
        val output = session.run(mapOf("melspectrogram" to tensor))
        val logits = (output[0].value as Array<*>)[0] as FloatArray

        val score = softmax(logits)[1]

        return if (score >= THRESHOLD) {
            confirmCount++
            confirmCount >= CONFIRM_N
        } else {
            confirmCount = 0
            false
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()!!
        val exp = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }

    private fun extractMelSpectrogram(pcm: FloatArray): Array<Array<Array<FloatArray>>> {
        // NOTE: This is a simplified placeholder.
        // For production, use TarsosDSP or a Kotlin FFT library.
        // The mel spectrogram MUST match training: sr=16000, n_mels=80,
        // hop_length=160, win_length=400, n_fft=512
        // Refer to the companion MelExtractor.kt file.
        TODO("Implement using TarsosDSP or custom FFT")
    }

    fun close() {
        session.close()
        env.close()
    }
}
```

## 5. Audio Capture Loop (Service)

```kotlin
class WakeWordService : Service() {
    private lateinit var detector: WakeWordDetector
    private val ringBuffer = ShortArray(16000)
    private var writePos   = 0
    private var lastTriggerMs = 0L
    private val COOLDOWN_MS   = 1500L

    override fun onCreate() {
        super.onCreate()
        detector = WakeWordDetector(applicationContext)
        startCapture()
    }

    private fun startCapture() {
        Thread {
            val chunkSize = 800  // 50ms chunk
            val buf = ShortArray(chunkSize)
            audioRecord.startRecording()

            while (isRunning) {
                val read = audioRecord.read(buf, 0, chunkSize)
                if (read > 0) {
                    // Fill ring buffer
                    for (i in 0 until read) {
                        ringBuffer[writePos % 16000] = buf[i]
                        writePos++
                    }
                    // Run detection every 500ms (every 8000 samples)
                    if (writePos % 8000 == 0) {
                        val pcmFloat = FloatArray(16000) { ringBuffer[it] / 32768f }
                        val now = System.currentTimeMillis()
                        if (detector.process(pcmFloat) && now - lastTriggerMs > COOLDOWN_MS) {
                            lastTriggerMs = now
                            onWakeWordDetected()
                        }
                    }
                }
            }
        }.start()
    }

    private fun onWakeWordDetected() {
        // Broadcast intent to UI
        sendBroadcast(Intent("com.yourapp.WAKE_WORD_DETECTED"))
    }
}
```

## 6. Permissions (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE"/>

<service
    android:name=".WakeWordService"
    android:foregroundServiceType="microphone"/>
```

## 7. Model Input/Output Spec

| Property | Value |
|---|---|
| Input name | `melspectrogram` |
| Input shape | `[1, 1, 80, 100]` |
| Input dtype | `float32` |
| Output name | `logits` |
| Output shape | `[1, 2]` |
| Index 0 | probability of NOT wake word |
| Index 1 | probability of wake word |

## 8. Performance Targets

- Inference time: <20ms on a mid-range Android device
- CPU usage: <5% average
- Memory: <50MB
- Detection latency: <500ms (using 500ms step)
