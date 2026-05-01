package com.example.breathapp

import okhttp3.MultipartBody
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

// --- UPDATED FOR 6 PLOTS ---
data class AnalysisResponse(
    val metrics: Map<String, Any>,

    // Signal Analysis
    val plot_signal: String?, // Raw vs Filtered
    val plot_peaks: String?,  // Peak Detection

    // PCA
    val plot_pca_l: String?,  // Linear
    val plot_pca_nl: String?, // Non-Linear

    // Clustering
    val plot_km: String?,     // K-Means
    val plot_opt: String?     // OPTICS
)

interface BreathApiService {
    @Multipart
    @POST("analyze")
    suspend fun uploadBreathData(
        @Part file: MultipartBody.Part
    ): AnalysisResponse
}

object RetrofitClient {
    // CHANGE THIS TO YOUR LAPTOP IP!
    private const val BASE_URL = "http://172.20.253.201:5000/"

    val api: BreathApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(BreathApiService::class.java)
    }
}
