package com.example.breathapp

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.breathapp.ui.theme.BreathAppTheme
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

val SERVICE_UUID: UUID = UUID.fromString("4fafc201-1fb5-459e-8fcc-c5c9c331914b")
val CHARACTERISTIC_UUID: UUID = UUID.fromString("beb5483e-36e1-4688-b7f5-ea07361b26a8")
val CHARACTERISTIC_CONFIG_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

enum class Screen { Home, Analysis, LearnMore, Settings }

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            var currentScreen by remember { mutableStateOf(Screen.Home) }
            val context = LocalContext.current
            val scope = rememberCoroutineScope()

            var connectionStatus by remember { mutableStateOf("Disconnected") }
            var bluetoothGatt by remember { mutableStateOf<BluetoothGatt?>(null) }
            val capturedData = remember { StringBuilder() }
            var dataRowCount by remember { mutableIntStateOf(0) }
            var lastReceivedData by remember { mutableStateOf("No data yet...") }

            val requestPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }

            LaunchedEffect(Unit) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    requestPermissionLauncher.launch(arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.ACCESS_FINE_LOCATION))
                }
            }

            val gattCallback = remember {
                object : BluetoothGattCallback() {
                    @SuppressLint("MissingPermission")
                    override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
                        if (newState == BluetoothProfile.STATE_CONNECTED) {
                            connectionStatus = "Connected"
                            bluetoothGatt = gatt
                            gatt.requestMtu(512)
                        } else {
                            connectionStatus = "Disconnected"
                            bluetoothGatt = null
                        }
                    }
                    @SuppressLint("MissingPermission")
                    override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
                        gatt.discoverServices()
                    }
                    @SuppressLint("MissingPermission")
                    override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
                        val char = gatt.getService(SERVICE_UUID)?.getCharacteristic(CHARACTERISTIC_UUID)
                        if (char != null) {
                            gatt.setCharacteristicNotification(char, true)
                            val desc = char.getDescriptor(CHARACTERISTIC_CONFIG_UUID)
                            if (desc != null) {
                                val value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                    gatt.writeDescriptor(desc, value)
                                } else {
                                    @Suppress("DEPRECATION")
                                    desc.value = value
                                    @Suppress("DEPRECATION")
                                    gatt.writeDescriptor(desc)
                                }
                            }
                        }
                    }
                    override fun onCharacteristicChanged(gatt: BluetoothGatt, char: BluetoothGattCharacteristic) {
                        val chunk = char.getStringValue(0)
                        capturedData.append(chunk)
                        lastReceivedData = if(chunk.length > 20) chunk.substring(0, 20) + "..." else chunk
                        dataRowCount += chunk.count { it == '\n' }
                    }
                }
            }

            @SuppressLint("MissingPermission")
            fun connectBluetooth() {
                val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as? android.bluetooth.BluetoothManager
                val scanner = bluetoothManager?.adapter?.bluetoothLeScanner
                if (scanner == null) return
                connectionStatus = "Scanning..."
                scanner.startScan(null, ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build(), object : ScanCallback() {
                    @SuppressLint("MissingPermission")
                    override fun onScanResult(callbackType: Int, result: ScanResult) {
                        if (result.device.name == "Breath_Device" || result.device.name == "ESP32_Breath") {
                            scanner.stopScan(this)
                            connectionStatus = "Found! Connecting..."
                            result.device.connectGatt(context, false, gattCallback)
                        }
                    }
                })
            }

            @SuppressLint("MissingPermission")
            fun sendCommand(cmd: String) {
                val gatt = bluetoothGatt ?: return
                val service = gatt.getService(SERVICE_UUID)
                val char = service?.getCharacteristic(CHARACTERISTIC_UUID)
                if (char != null) {
                    val bytes = cmd.toByteArray()
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        gatt.writeCharacteristic(char, bytes, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT)
                    } else {
                        @Suppress("DEPRECATION")
                        char.value = bytes
                        @Suppress("DEPRECATION")
                        char.writeType = BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
                        @Suppress("DEPRECATION")
                        gatt.writeCharacteristic(char)
                    }
                }
            }

            BreathAppTheme {
                Scaffold(
                    topBar = {
                        TopAppBar(
                            title = { Text(if(currentScreen == Screen.Settings) "Settings" else "Breath App") },
                            navigationIcon = {
                                if (currentScreen != Screen.Home) {
                                    IconButton(onClick = { currentScreen = Screen.Home }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
                                }
                            },
                            actions = {
                                IconButton(onClick = { currentScreen = Screen.Settings }) { Icon(Icons.Default.Settings, "Settings") }
                            }
                        )
                    }
                ) { innerPadding ->
                    Surface(modifier = Modifier.padding(innerPadding)) {
                        when (currentScreen) {
                            Screen.Home -> HomeScreen({ currentScreen = Screen.Analysis }, { currentScreen = Screen.LearnMore })
                            Screen.Settings -> SettingsScreen(connectionStatus) { connectBluetooth() }
                            Screen.Analysis -> AnalysisScreen(
                                connectionStatus, dataRowCount, lastReceivedData,
                                { capturedData.clear(); dataRowCount = 0; sendCommand("RUN_PYTHON") },
                                { uploadAndAnalyze(context, scope, capturedData.toString()) }
                            )
                            Screen.LearnMore -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("DICE Lab 2025") }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun AnalysisScreen(status: String, count: Int, preview: String, onCapture: () -> Unit, onAnalyze: () -> Unit) {
    var isAnalyzing by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Breath Analysis", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Card(Modifier.fillMaxWidth().padding(8.dp), colors = CardDefaults.cardColors(containerColor = Color(0xFFE3F2FD))) {
            Column(Modifier.padding(16.dp)) {
                Text("Connection: $status", fontWeight = FontWeight.Bold)
                Text("Rows: $count")
                Text("Preview: $preview", fontSize = 12.sp, color = Color.Gray)
            }
        }
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(onClick = onCapture, enabled = status.contains("Connected")) { Text("Start Capture") }
            Button(onClick = { isAnalyzing = true; onAnalyze() }, enabled = status.contains("Connected"), colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50))) { Text("Analyze Data") }
        }
        Spacer(Modifier.height(20.dp))

        GlobalAnalysisState.lastResponse?.let { result ->
            HorizontalDivider()
            Text("Analysis Results", fontSize = 22.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(vertical = 10.dp))

            // Metrics
            Card(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp), colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0))) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Session Metrics:", fontWeight = FontWeight.Bold)
                    result.metrics.forEach { (key, value) -> Text("• $key: $value") }
                }
            }

            // 1. Signal Processing
            Text("1. Signal Processing", fontSize = 18.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top=20.dp))
            result.plot_signal?.let { Text("Raw vs Filtered", fontWeight = FontWeight.Bold); Base64Image(it) }
            result.plot_peaks?.let { Text("Peak Detection", fontWeight = FontWeight.Bold); Base64Image(it) }

            // 2. PCA
            Text("2. PCA Analysis", fontSize = 18.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top=20.dp))
            result.plot_pca_l?.let { Text("Linear PCA", fontWeight = FontWeight.Bold); Base64Image(it) }
            result.plot_pca_nl?.let { Text("Non-Linear PCA", fontWeight = FontWeight.Bold); Base64Image(it) }

            // 3. Clustering
            Text("3. Clustering Analysis", fontSize = 18.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top=20.dp))
            result.plot_km?.let { Text("K-Means", fontWeight = FontWeight.Bold); Base64Image(it) }
            result.plot_opt?.let { Text("OPTICS", fontWeight = FontWeight.Bold); Base64Image(it) }

            Spacer(Modifier.height(50.dp))
        }
    }
}

@Composable
fun SettingsScreen(status: String, onConnect: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Status: $status", fontSize = 20.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(20.dp))
        Button(onClick = onConnect, enabled = !status.contains("Connected")) { Text("Connect") }
    }
}

@Composable
fun HomeScreen(onStart: () -> Unit, onLearn: () -> Unit) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Breath App", fontSize = 32.sp); Spacer(Modifier.height(20.dp))
        Button(onClick = onStart) { Text("Go to Analysis") }
        OutlinedButton(onClick = onLearn) { Text("Learn More") }
    }
}

@Composable
fun Base64Image(base64Str: String) {
    val bitmap = remember(base64Str) {
        try {
            val decodedString = Base64.decode(base64Str, Base64.DEFAULT)
            BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size)
        } catch (e: Exception) { null }
    }
    if (bitmap != null) Image(bitmap = bitmap.asImageBitmap(), contentDescription = null, modifier = Modifier.fillMaxWidth().height(200.dp).background(Color.White), contentScale = ContentScale.Fit)
}

object GlobalAnalysisState { var lastResponse by mutableStateOf<AnalysisResponse?>(null) }

fun uploadAndAnalyze(context: Context, scope: kotlinx.coroutines.CoroutineScope, data: String) {
    scope.launch {
        try {
            val file = File(context.cacheDir, "capture.csv")
            FileOutputStream(file).use { it.write(data.toByteArray()) }
            val body = MultipartBody.Part.createFormData("file", "capture.csv", file.asRequestBody("text/csv".toMediaTypeOrNull()))
            Toast.makeText(context, "Uploading...", Toast.LENGTH_SHORT).show()
            GlobalAnalysisState.lastResponse = RetrofitClient.api.uploadBreathData(body)
            Toast.makeText(context, "Done!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) { Toast.makeText(context, "Error: ${e.message}", Toast.LENGTH_LONG).show() }
    }
}