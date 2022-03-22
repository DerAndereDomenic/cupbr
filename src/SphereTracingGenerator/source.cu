#include <iostream>
#include <fstream>

#include <CUPBR.h>
#include <CUNET.h>

#include <Models/LenGen.h>

using namespace cupbr;

#define BINS_BETA_O 50
#define BINS_BETA_I 360
#define BINS_GAMMA_I 180
#define BINS_THETA_I 180
#define BINS_PHI_I 180

struct MediumSettings
{
    float sigma;
    float phi;
    float g;
};

/*struct PathSummary
{
    uint32_t N;
    Vector3float x;
    Vector3float w;
    Vector3float X;
    Vector3float W;
};

struct FileLine
{
    float sigma;
    float g;
    float phi;
    uint32_t N;
    float cosTheta;
    float beta;
    float alpha;
    float Xx;
    float Xy;
    float Xz;
    float Wx;
    float Wy;
    float Wz;
};*/

__device__ MediumSettings generateNewSettings(uint32_t& seed)
{
    float densityRnd = Math::rnd(seed);
    float scatterAlbedoRnd = Math::rnd(seed);
    float gRnd = Math::rnd(seed);

    float density = (densityRnd * densityRnd * densityRnd) * 300.0f;
    float p = scatterAlbedoRnd * scatterAlbedoRnd * scatterAlbedoRnd;
    float scatterAlbedo = fminf(1.0f, 1.000001f - p * p);
    float g = Math::clamp(gRnd * 2.0f - 1.0f, -0.999f, 0.999f);

    return {density, scatterAlbedo, g};
}

/*__device__ float invertcdf(const float& g, const float& xi)
{
    float t = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
    return 0.5f * (1 + g * g - t * t) / g;
}

__device__ void createOrthoBasis(const Vector3float& N, Vector3float& T, Vector3float& B)
{
    float sign = N.z / fabsf(N.z);
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = Vector3float(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = Vector3float(b, sign + N.y * N.y * a, -N.y);
}

__device__ float distanceToBoundary(const Vector3float& x, const Vector3float& d)
{
    float b = 2.0f * Math::dot(x,d);
    float c = Math::dot(x,x) - 1.0f;

    float disc = b * b - 4 * c;
    if (disc <= 0)
        return 0;

    return fmaxf(0.0f, (-b + sqrtf(disc)) / 2.0f);
}

__device__ Vector3float samplePhase(const Vector3float& w, const float& g, uint32_t& seed)
{
    if(abs(g) < 0.001f)
    {
        //Random direction
        float r1 = Math::rnd(seed);
        float r2 = Math::rnd(seed) * 2.0f - 1.0f;
        float sqrR2 = r2 * r2;
        float two_pi_by_r1 = 2.0f * M_PI * r1;
        float sqrt_of_one_minus_sqrR2 = sqrtf(1.0f - sqrR2);
        float x = cosf(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
        float y = sinf(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
        float z = r2;

        Vector3float t0, t1;
        createOrthoBasis(-1.0f * w, t0, t1);
        return t0 * x + t1 * y + (-1.0f * w)*z;
    }

    float phi = Math::rnd(seed) * 2 * M_PI;
    float cosTheta = invertcdf(g, Math::rnd(seed));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    Vector3float t0, t1;
    createOrthoBasis(w, t0, t1);

    return sinTheta * sinf(phi) * t0 + sinTheta * cosf(phi) * t1 + cosTheta * w;
}

__device__ PathSummary getVPTSampleInSphere(MediumSettings& settings, uint32_t& seed)
{
    Vector3float x = 0;
    Vector3float w(0,0,1);
    Vector3float X = x;
    Vector3float W = w;
    uint32_t N = 0;
    float accum = 0;
    float importance = 1;

    while(true)
    {
        importance *= settings.phi;
        accum += importance;

        if(Math::rnd(seed) < importance / accum)
        {
            X = x;
            W = w;
        }

        w = samplePhase(w, settings.g, seed);
        ++N;

        float d = distanceToBoundary(x,w);

        float t = settings.sigma < 0.00001f ? 10000000.0f : -log(fmaxf(0.000000001f, 1.0f - Math::rnd(seed))) / settings.sigma;

        if(t>= d || isnan(t) || isinf(t))
        {
            x += w * d;
            return {N, x, w, X, W};
        }
        x += w * t;
    }
}

__global__ void generateSamples(const uint32_t N, FileLine* output)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= N)
    {
        return;
    }

    uint32_t seed = Math::tea<4>(tid, 0);

    MediumSettings settings = generateNewSettings(seed);
    PathSummary r = getVPTSampleInSphere(settings, seed);

    Vector3float zAxis = Vector3float(0,0,1);
    Vector3float xAxis = fabsf(r.x.z) > 0.999 ? Vector3float(1,0,0) : Math::normalize(Math::cross(r.x, Vector3float(0,0,1)));
    Vector3float yAxis = Math::cross(zAxis, xAxis);

    Matrix3x3float R = Math::transpose(Matrix3x3float(xAxis, yAxis, zAxis));

    Vector3float normx = R * r.x;
    Vector3float normw = R * r.w;
    Vector3float normX = R * r.X;
    Vector3float normW = R * r.W;
    
    Vector3float B(1,0,0);
    Vector3float T = Math::cross(normx, B);
    float cosTheta = normx.z;
    float beta = Math::dot(normw, T);
    float alpha = Math::dot(normw, B);

    output[tid] = 
    {
        settings.sigma,
        settings.g,
        settings.phi,
        r.N,
        cosTheta,
        beta,
        alpha,
        normX.x,
        normX.y,
        normX.z,
        normW.x,
        normW.y,
        normW.z
    };
}

void generateDatasetForTrainingCVAE()
{
    const uint32_t N = 1 << 22;
    printf("Generatirng %i samples\n", N);

    FileLine* dev_output = Memory::createDeviceArray<FileLine>(N);

    printf("Start generating samples...\n");
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    generateSamples<<<config.blocks, config.threads>>>(N, dev_output);
    cudaSafeCall(cudaDeviceSynchronize());
    printf("Start writing file...\n");

    FileLine* output = Memory::createHostArray<FileLine>(N);
    Memory::copyDevice2HostArray<FileLine>(N, dev_output, output);

    std::ofstream file;
    file.open("ScattersDataSet.ds");

    for(uint32_t i = 0; i < N; ++i)
    {
        file << output[i].sigma << ", " <<
                output[i].g << ", " <<
                output[i].phi << ", " <<
                output[i].N << ", " <<
                output[i].cosTheta << ", " <<
                output[i].beta << ", " <<
                output[i].alpha << ", " <<
                output[i].Xx << ", " <<
                output[i].Xy << ", " <<
                output[i].Xz << ", " <<
                output[i].Wx << ", " << 
                output[i].Wy << ", " <<
                output[i].Wz << "\n";
    }

    file.close();

    Memory::destroyHostArray<FileLine>(output);
    Memory::destroyDeviceObject<FileLine>(dev_output);
}

__global__ void test(const uint32_t N, cunet::LenGen lenGen, float* buffer)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();

    if (tid >= N)
        return;

    uint32_t seed = Math::tea<4>(tid, 0);

    Vector2float lenLatent = Math::sampleStdNormal2D(seed);
    float lenInput[4] = {10, 0.9, lenLatent.x, lenLatent.y};
    float lenOutput[2];
    lenGen(lenInput, lenOutput);

    float logN = fmaxf(0.0f, lenOutput[0] + Math::sampleStdNormal1D(seed) * expf(0.5*Math::clamp(lenOutput[1], -16.0f, 16.0f)));
    float n = roundf(expf(logN) + 0.49f);

    buffer[tid] = n;
}

int run()
{
    cudaSafeCall(cudaSetDevice(0));

    uint32_t N = 1 << 22;

    cunet::LenGen lenGen_host;
    cunet::LenGen* lenGen = cunet::Memory::createDeviceObject<cunet::LenGen>();

    cunet::Memory::copyHost2DeviceObject<cunet::LenGen>(&lenGen_host, lenGen);

    float* buffer = Memory::createDeviceArray<float>(N);

    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    test << <config.blocks, config.threads >> > (N, lenGen_host, buffer);
    cudaDeviceSynchronize();

    float* host_buffer = Memory::createHostArray<float>(N);
    Memory::copyDevice2HostArray<float>(N, buffer, host_buffer);
    Memory::destroyDeviceArray<float>(buffer);

    std::ofstream file;
    file.open("Evaluation.ds");

    for(uint32_t i = 0; i < N; ++i)
    {
        file << host_buffer[i] << "\n";
    }

    file.close();


    Memory::destroyHostArray<float>(host_buffer);
    //generateDatasetForTrainingCVAE();

    return 0;
}

int main()
{
    int exit = run();
    return exit;
}*/

struct PathSummary
{
    Vector3float inc_pos;
    Vector3float inc_dir;
    Vector3float out_pos;
    Vector3float out_dir;
    uint32_t num_scattering;
};

struct DataPoint
{
    float cos_beta_o;
    float sigma;
    float g;
    float num_scatters;
    float cos_theta_i;
    float phi_i;
    float cos_beta_i;
    float gamma_i;
};

__device__ float distanceToBoundary(const Vector3float& x, const Vector3float& d)
{
    float b = 2.0f * Math::dot(x,d);
    float c = Math::dot(x,x) - 1.0f;

    float disc = b * b - 4 * c;
    if (disc <= 0)
        return 0;

    return fmaxf(0.0f, (-b + sqrtf(disc)) / 2.0f);
}

__device__ Vector3float sampleHenyeyGreensteinPhase(const float& g, const Vector3float& forward, uint32_t& seed)
{
    float u1 = Math::rnd(seed);
    float u2 = Math::rnd(seed);

    float g2 = g * g;
    float d = (1.0f - g2) / (1.0f - g + 2.0f * g * u1);
    float cos_theta = Math::clamp(0.5f / g * (1.0f + g2 - d * d), -1.0f, 1.0f);

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * 3.14159f * u2;

    float x = sin_theta * cosf(phi);
    float y = sin_theta * sinf(phi);
    float z = cos_theta;

    Vector3float result = Math::normalize(Math::toLocalFrame(forward, Vector3float(x, y, z)));

    return result;
}

__host__ __device__ Vector3float sampleHemisphereUniform(uint32_t& seed, const Vector3float& N)
{
    float z = Math::rnd(seed) * 2.0f - 1.0f;
    float phi = Math::rnd(seed) * 2.0f * M_PI;

    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    Vector3float result(x, y, z);

    return Math::dot(result, N) < 0 ? -1.0f * result : result;
}

__host__ __device__ Vector3float sampleSphereUniform(uint32_t& seed)
{
    Vector3float result = sampleHemisphereUniform(seed, Vector3float(1, 0, 0));
    return Math::rnd(seed) < 0.5 ? result : -1.0f * result;
}

__global__ void generateSamples(const uint32_t num_samples, Sphere sphere, DataPoint* dataset)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();
    
    if(tid >= num_samples)
    {
        return;
    }

    uint32_t seed = Math::tea<4>(tid, 0);

    Vector3float start_pos = sampleSphereUniform(seed);
    Vector3float inc_pos = start_pos;
    Vector3float normal = Math::normalize(start_pos - sphere.position());
    Vector3float direction = -1.0f * sampleHemisphereUniform(seed, normal); // Points towards surface
    Vector3float inc_dir = direction;
    Ray ray(start_pos + 0.001f * direction, direction);

    //Volume
    MediumSettings settings = generateNewSettings(seed);
    float sigma_t = settings.sigma;
    float g = settings.g;

    Vector3float out_position;
    Vector3float out_direction;

    float num_scattering_events = 0.0f;

    while(true)
    {
        ray.traceNew(start_pos + 0.001f * direction, direction);

        float d = distanceToBoundary(ray.origin(), ray.direction());

        if (d == INFINITY)
        {
            out_position = ray.origin();
            out_direction = ray.direction();
            break;
        }

        float t = -logf(1.0f - Math::rnd(seed)) / sigma_t;

        if (t >= d)
        {
            out_position = ray.origin() + d * ray.direction();
            out_direction = ray.direction();
            break;
        }

        ++num_scattering_events;

        start_pos = ray.origin() + t * ray.direction();
        direction = sampleHenyeyGreensteinPhase(g, ray.direction(), seed);
    }

    //Calculate histogram values
    float cos_beta_o = -Math::dot(normal, inc_dir); // in [0,1] because beta_o in [0,pi/2]

    //Calculate reference coordinate system
    Vector3float lZ = inc_dir;
    float t = Math::dot(lZ, sphere.position()) - Math::dot(lZ, inc_pos);
    Vector3float lX = Math::normalize(inc_pos + t * lZ);
    Vector3float lY = Math::cross(lZ, lX);

    Matrix3x3float local2global(lX, lY, lZ);
    Matrix3x3float global2local = Math::transpose(local2global);

    Vector3float out_pos_local = global2local * out_position;
    Vector3float out_dir_local = global2local * out_direction;

    float cos_theta_i = out_pos_local.z;                        // in [-1, 1] because theta_i in [0, pi]
    float phi_i = atan2f(out_pos_local.y, out_pos_local.x);     // in [-pi, pi]

    float cos_beta_i = out_dir_local.z;                         // in [-1, 1] because beta_i in [0, pi]
    float gamma_i = atan2f(out_dir_local.y, out_dir_local.x);   // in [-pi, pi]

    dataset[tid] = {cos_beta_o, settings.sigma, settings.g, num_scattering_events, cos_theta_i, phi_i, cos_beta_i, gamma_i };
}

__global__ void convertDataSet(const uint32_t N, PathSummary* buffer, DataPoint* dataset)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= N)
    {
        return;
    }

    PathSummary summary = buffer[tid];

    float theta_dir_in = acosf(-Math::clamp(summary.inc_dir.z, -1.0f, 1.0f));

    float theta_pos_out = acosf(Math::clamp(summary.out_pos.z, -1.0f, 1.0f));
    float phi_pos_out = atan2f(summary.out_pos.y, summary.out_pos.x);

    float theta_dir_out = acosf(Math::clamp(summary.out_dir.z, -1.0f, 1.0f));
    float phi_dir_out = atan2f(summary.out_dir.y, summary.out_dir.x);
    
    //dataset[tid] = {theta_dir_in, summary.num_scattering, theta_pos_out, phi_pos_out, theta_dir_out, phi_dir_out};
}

void generateDataSet()
{
    const uint32_t N = 1 << 22;
    printf("Generating %i samples...\n", N);

    Sphere sphere(Vector3float(0, 0, 0), 1);
    
    PathSummary* buffer = Memory::createDeviceArray<PathSummary>(N);
    DataPoint* dataset = Memory::createDeviceArray<DataPoint>(N);

    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    generateSamples << <config.blocks, config.threads >> > (N, sphere, dataset);
    ::cudaSafeCall(cudaDeviceSynchronize());

    //convertDataSet << <config.blocks, config.threads >> > (N, buffer, dataset);
    //::cudaSafeCall(cudaDeviceSynchronize());

    DataPoint* host_dataset = Memory::createHostArray<DataPoint>(N);
    Memory::copyDevice2HostArray<DataPoint>(N, dataset, host_dataset);
    Memory::destroyDeviceArray<DataPoint>(dataset);

    std::ofstream file("SphereScatters_Ver6.ds", std::ios::out | std::ios::binary);
    file.write((char*)host_dataset, sizeof(DataPoint) * N);
    file.close();

    /*for(uint32_t i = 0; i < N; ++i)
    {
        file << host_dataset[i].inc_pos.x << ", " <<
            host_dataset[i].inc_pos.y << ", " <<
            host_dataset[i].inc_pos.z << ", " <<
            host_dataset[i].num_scattering << ", " <<
            host_dataset[i].out_pos.x << ", " <<
            host_dataset[i].out_pos.y << ", " <<
            host_dataset[i].out_pos.z << ", " <<
            host_dataset[i].out_dir.x << ", " <<
            host_dataset[i].out_dir.y << ", " <<
            host_dataset[i].out_dir.z << "\n";
    }*/

    file.close();

    Memory::destroyHostArray<DataPoint>(host_dataset);
    Memory::destroyDeviceArray<PathSummary>(buffer);
}

__global__ void gsdf_kernel(const Sphere sphere, const uint32_t N, float* histogram_x, float* histogram_w)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();
    
    if(tid >= N)
    {
        return;
    }

    uint32_t seed = Math::tea<4>(tid, 0);

    Vector3float start_pos = sampleSphereUniform(seed);
    Vector3float inc_pos = start_pos;
    Vector3float normal = Math::normalize(start_pos - sphere.position());
    Vector3float direction = -1.0f * sampleHemisphereUniform(seed, start_pos); // Points towards surface
    Vector3float inc_dir = direction;
    Ray ray(start_pos + 0.001f * direction, direction);

    //Volume
    float sigma_s = 3.0f;
    float sigma_a = 0.0f;
    float sigma_t = sigma_s + sigma_a;
    float g = 0.6f;

    Vector3float out_position;
    Vector3float out_direction;

    int32_t num_scatters = 0;

    while(true)
    {
        ray.traceNew(start_pos + 0.001f * direction, direction);

        float d = distanceToBoundary(ray.origin(), ray.direction());

        if (d == INFINITY)
        {
            out_position = ray.origin();
            out_direction = ray.direction();
            break;
        }

        float t = -logf(1.0f - Math::rnd(seed)) / sigma_t;

        if (t >= d)
        {
            out_position = ray.origin() + d * ray.direction();
            out_direction = ray.direction();
            break;
        }
        ++num_scatters;

        start_pos = ray.origin() + t * ray.direction();
        direction = sampleHenyeyGreensteinPhase(g, ray.direction(), seed);
    }

    if (num_scatters == 0)
        return;

    //Calculate histogram values
    float cos_beta_o = -Math::dot(normal, inc_dir); // in [0,1] because beta_o in [0,pi/2]

    //Calculate reference coordinate system
    Vector3float lZ = inc_dir;
    float t = Math::dot(lZ, sphere.position()) - Math::dot(lZ, inc_pos);
    Vector3float lX = Math::normalize(inc_pos + t * lZ);
    Vector3float lY = Math::cross(lZ, lX);

    Matrix3x3float local2global(lX, lY, lZ);
    Matrix3x3float global2local = Math::transpose(local2global);

    Vector3float out_pos_local = global2local * out_position;
    Vector3float out_dir_local = global2local * out_direction;

    float cos_beta_i = out_dir_local.z;                         // in [-1, 1] because beta_i in [0, pi]
    float gamma_i = atan2f(out_dir_local.y, out_dir_local.x);   // in [-pi, pi]

    float cos_theta_i = out_pos_local.z;                        // in [-1, 1] because theta_i in [0, pi]
    float phi_i = atan2f(out_pos_local.y, out_pos_local.x);     // in [-pi, pi]

    // Calculate bins
    float pif = static_cast<float>(M_PI);
    int32_t bin_beta_o = static_cast<int32_t>(Math::clamp(cos_beta_o, 0.0f, 0.9999f) * BINS_BETA_O);
    int32_t bin_beta_i = static_cast<int32_t>(Math::clamp((cos_beta_i + 1.0f) / 2.0f, 0.0f, 0.9999f) * BINS_BETA_I);
    int32_t bin_gamma_i = static_cast<int32_t>(Math::clamp((gamma_i + pif) / (2.0f * pif), 0.0f, 0.9999f) * BINS_GAMMA_I);
    int32_t bin_theta_i = static_cast<int32_t>(Math::clamp((cos_theta_i + 1.0f) / 2.0f, 0.0f, 0.9999f) * BINS_THETA_I);
    int32_t bin_phi_i = static_cast<int32_t>(Math::clamp((phi_i + pif) / (2.0f * pif), 0.0f, 0.9999f) * BINS_PHI_I);

    int32_t index_position = bin_beta_o + BINS_BETA_O * bin_theta_i + BINS_BETA_O * BINS_THETA_I * bin_phi_i;
    int32_t index_direction = bin_beta_o + BINS_BETA_O * bin_beta_i + BINS_BETA_O * BINS_BETA_I * bin_gamma_i;

    atomicAdd(&histogram_x[index_position], 1.0f);
    atomicAdd(&histogram_w[index_direction], 1.0f);
}

void generateGSDF()
{
    const uint32_t N = 1 << 31;
    printf("Generating %u samples...\n", N);

    float* histogram_x = Memory::createDeviceArray<float>(BINS_BETA_O * BINS_PHI_I * BINS_THETA_I);
    float* histogram_w = Memory::createDeviceArray<float>(BINS_BETA_O * BINS_BETA_I * BINS_GAMMA_I);

    Sphere sphere(Vector3float(0), 1);

    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    gsdf_kernel << <config.blocks, config.threads >> > (sphere, N, histogram_x, histogram_w);
    ::cudaSafeCall(cudaDeviceSynchronize());

    float* pdf_x = Memory::createHostArray<float>(BINS_BETA_O * BINS_PHI_I * BINS_THETA_I);
    float* pdf_w = Memory::createHostArray<float>(BINS_BETA_O * BINS_BETA_I * BINS_GAMMA_I);
    Memory::copyDevice2HostArray<float>(BINS_BETA_O * BINS_PHI_I * BINS_THETA_I, histogram_x, pdf_x);
    Memory::copyDevice2HostArray<float>(BINS_BETA_O * BINS_BETA_I * BINS_GAMMA_I, histogram_w, pdf_w);

    printf("Normalize position...\n");

    for(uint32_t bin_beta_o = 0; bin_beta_o < BINS_BETA_O; ++bin_beta_o)
    {
        float normalization = 0.0f;
        for(uint32_t bin_theta_i = 0; bin_theta_i < BINS_THETA_I; ++bin_theta_i)
        {
            for(uint32_t bin_phi_i = 0; bin_phi_i < BINS_PHI_I; ++bin_phi_i)
            {
                normalization += pdf_x[bin_beta_o + BINS_BETA_O * bin_theta_i + BINS_BETA_O * BINS_THETA_I * bin_phi_i];
            }
        }

        for(uint32_t bin_theta_i = 0; bin_theta_i < BINS_THETA_I; ++bin_theta_i)
        {
            for(uint32_t bin_phi_i = 0; bin_phi_i < BINS_PHI_I; ++bin_phi_i)
            {
                pdf_x[bin_beta_o + BINS_BETA_O * bin_theta_i + BINS_BETA_O * BINS_THETA_I * bin_phi_i] /= normalization;
            }
        }
    }

    printf("Normalize direction...\n");

    for(uint32_t bin_beta_o = 0; bin_beta_o < BINS_BETA_O; ++bin_beta_o)
    {
        float normalization = 0.0f;
        for(uint32_t bin_beta_i = 0; bin_beta_i < BINS_BETA_I; ++bin_beta_i)
        {
            for(uint32_t bin_gamma_i = 0; bin_gamma_i < BINS_GAMMA_I; ++bin_gamma_i)
            {
                normalization += pdf_w[bin_beta_o + BINS_BETA_O * bin_beta_i + BINS_BETA_O * BINS_BETA_I * bin_gamma_i];
            }
        }

        for(uint32_t bin_beta_i = 0; bin_beta_i < BINS_BETA_I; ++bin_beta_i)
        {
            for(uint32_t bin_gamma_i = 0; bin_gamma_i < BINS_GAMMA_I; ++bin_gamma_i)
            {
                pdf_w[bin_beta_o + BINS_BETA_O * bin_beta_i + BINS_BETA_O * BINS_BETA_I * bin_gamma_i] /= normalization;
            }
        }
    }

    printf("Writing to file...\n");
    std::ofstream pdf_x_file("pdf_x.gsdf", std::ios::out | std::ios::binary);
    pdf_x_file.write((char*)pdf_x, sizeof(float) * BINS_BETA_O * BINS_PHI_I * BINS_THETA_I);
    pdf_x_file.close();

    std::ofstream pdf_w_file("pdf_w.gsdf", std::ios::out | std::ios::binary);
    pdf_w_file.write((char*)pdf_w, sizeof(float) * BINS_BETA_O * BINS_GAMMA_I * BINS_BETA_I);
    pdf_w_file.close();

    printf("Done!\n");

    Memory::destroyHostArray<float>(pdf_x);
    Memory::destroyHostArray<float>(pdf_w);
    Memory::destroyDeviceArray<float>(histogram_w);
    Memory::destroyDeviceArray<float>(histogram_x);
}

#define BINS_INC_DIR 100
#define BINS_N 100

__global__ void kernel_generateTestHistograms(const uint32_t N, float* histogram_n, float* histogram_x, float* histogram_w)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();
    
    if(tid >= N)
    {
        return;
    }

    uint32_t seed = Math::tea<4>(tid, 0);

    Vector3float start_pos = sampleSphereUniform(seed);
    Vector3float inc_pos = start_pos;
    Vector3float direction = -1.0f * sampleHemisphereUniform(seed, start_pos); // Points towards surface
    Vector3float inc_dir = direction;
    Ray ray(start_pos + 0.001f * direction, direction);

    //Volume
    float sigma_s = 3.0f;
    float sigma_a = 0.0f;
    float sigma_t = sigma_s + sigma_a;
    float g = 0.6f;

    Vector3float out_position;
    Vector3float out_direction;

    float num_scattering_events = 0.0f;

    while(true)
    {
        ray.traceNew(start_pos + 0.001f * direction, direction);

        float d = distanceToBoundary(ray.origin(), ray.direction());

        if (d == INFINITY)
        {
            out_position = ray.origin();
            out_direction = ray.direction();
            break;
        }

        float t = -logf(1.0f - Math::rnd(seed)) / sigma_t;

        if (t >= d)
        {
            out_position = ray.origin() + d * ray.direction();
            out_direction = ray.direction();
            break;
        }

        ++num_scattering_events;

        start_pos = ray.origin() + t * ray.direction();
        direction = sampleHenyeyGreensteinPhase(g, ray.direction(), seed);
    }

    if (num_scattering_events >= 100 || num_scattering_events == 0)
        return;

    //Calculate histogram values
    float cos_beta_o = -Math::dot(inc_pos, inc_dir); // in [0,1] because beta_o in [0,pi/2]

    //Calculate reference coordinate system
    Vector3float lZ = inc_dir;
    float t = - Math::dot(lZ, inc_pos);
    Vector3float lX = Math::normalize(inc_pos + t * lZ);
    Vector3float lY = Math::cross(lZ, lX);

    Matrix3x3float local2global(lX, lY, lZ);
    Matrix3x3float global2local = Math::transpose(local2global);

    Vector3float out_pos_local = global2local * out_position;
    Vector3float out_dir_local = global2local * out_direction;

    float cos_beta_i = out_dir_local.z;                         // in [-1, 1] because beta_i in [0, pi]
    float gamma_i = atan2f(out_dir_local.y, out_dir_local.x);   // in [-pi, pi]

    float cos_theta_i = out_pos_local.z;                        // in [-1, 1] because theta_i in [0, pi]
    float phi_i = atan2f(out_pos_local.y, out_pos_local.x);     // in [-pi, pi]

    // Calculate bins
    float pif = static_cast<float>(M_PI);
    int32_t bin_beta_o = static_cast<int32_t>(Math::clamp(cos_beta_o, 0.0f, 0.9999f) * BINS_INC_DIR);
    int32_t bin_n = static_cast<int32_t>(num_scattering_events);
    int32_t bin_beta_i = static_cast<int32_t>(Math::clamp((cos_beta_i + 1.0f) / 2.0f, 0.0f, 0.9999f) * BINS_BETA_I);
    int32_t bin_gamma_i = static_cast<int32_t>(Math::clamp((gamma_i + pif) / (2.0f * pif), 0.0f, 0.9999f) * BINS_GAMMA_I);
    int32_t bin_theta_i = static_cast<int32_t>(Math::clamp((cos_theta_i + 1.0f) / 2.0f, 0.0f, 0.9999f) * BINS_THETA_I);
    int32_t bin_phi_i = static_cast<int32_t>(Math::clamp((phi_i + pif) / (2.0f * pif), 0.0f, 0.9999f) * BINS_PHI_I);

    int32_t index_n = bin_beta_o + BINS_INC_DIR * bin_n;
    int32_t index_x = bin_beta_o + BINS_INC_DIR * bin_n + BINS_INC_DIR * BINS_N * bin_theta_i + BINS_INC_DIR * BINS_N * BINS_THETA_I * bin_phi_i;
    int32_t index_w = bin_beta_o + BINS_INC_DIR * bin_n + BINS_INC_DIR * BINS_N * bin_beta_i + BINS_INC_DIR * BINS_N * BINS_BETA_I * bin_gamma_i;

    atomicAdd(&histogram_n[index_n], 1.0f);
    atomicAdd(&histogram_x[index_x], 1.0f);
    atomicAdd(&histogram_w[index_w], 1.0f);
}

void generateTestHistograms()
{
    const uint32_t N = 1 << 31;
    printf("Generating %i samples...\n", N);

    float* histogram_n = Memory::createDeviceArray<float>(BINS_INC_DIR * BINS_N);
    float* histogram_x = Memory::createDeviceArray<float>(BINS_INC_DIR * BINS_N * BINS_THETA_I * BINS_PHI_I);
    float* histogram_w = Memory::createDeviceArray<float>(BINS_INC_DIR * BINS_N * BINS_BETA_I * BINS_GAMMA_I);

    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    kernel_generateTestHistograms << <config.blocks, config.threads >> > (N, histogram_n, histogram_x, histogram_w);
    ::cudaSafeCall(cudaDeviceSynchronize());

    float* pdf_n = Memory::createHostArray<float>(BINS_INC_DIR * BINS_N);
    float* pdf_x = Memory::createHostArray<float>(BINS_INC_DIR * BINS_N * BINS_THETA_I * BINS_PHI_I);
    float* pdf_w = Memory::createHostArray<float>(BINS_INC_DIR * BINS_N * BINS_BETA_I * BINS_GAMMA_I);
    Memory::copyDevice2HostArray<float>(BINS_INC_DIR * BINS_N, histogram_n, pdf_n);
    Memory::copyDevice2HostArray<float>(BINS_INC_DIR * BINS_N * BINS_THETA_I * BINS_PHI_I, histogram_x, pdf_x);
    Memory::copyDevice2HostArray<float>(BINS_INC_DIR * BINS_N * BINS_BETA_I * BINS_GAMMA_I, histogram_w, pdf_w);

    printf("Normalize n...\n");

    for(uint32_t bin_beta_o = 0; bin_beta_o < BINS_INC_DIR; ++bin_beta_o)
    {
        float normalization = 0.0f;
        for(uint32_t bin_n = 0; bin_n < BINS_N; ++bin_n)
        {
            normalization += pdf_n[bin_beta_o + BINS_INC_DIR * bin_n];
        }

        for(uint32_t bin_n = 0; bin_n < BINS_N; ++bin_n)
        {
            pdf_n[bin_beta_o + BINS_INC_DIR * bin_n] /= normalization;
        }
    }

    printf("Normalize x...\n");
    for(uint32_t bin_beta_o = 0; bin_beta_o < BINS_INC_DIR; ++bin_beta_o)
    {
        for(uint32_t bin_n = 0; bin_n < BINS_N; ++bin_n)
        {
            float normalization = 0.0f;
            for(uint32_t bin_theta_i = 0; bin_theta_i < BINS_THETA_I; ++bin_theta_i)
            {
                for(uint32_t bin_phi_i = 0; bin_phi_i < BINS_PHI_I; ++bin_phi_i)
                {
                    normalization += pdf_x[bin_beta_o + BINS_INC_DIR * bin_n + BINS_INC_DIR * BINS_N * bin_theta_i + BINS_INC_DIR * BINS_N * BINS_THETA_I * bin_phi_i];
                }
            }

            if (Math::safeFloatEqual(normalization, 0.0f))
                continue;

            for(uint32_t bin_theta_i = 0; bin_theta_i < BINS_THETA_I; ++bin_theta_i)
            {
                for(uint32_t bin_phi_i = 0; bin_phi_i < BINS_PHI_I; ++bin_phi_i)
                {
                    pdf_x[bin_beta_o + BINS_INC_DIR * bin_n + BINS_INC_DIR * BINS_N * bin_theta_i + BINS_INC_DIR * BINS_N * BINS_THETA_I * bin_phi_i] /= normalization;
                }
            }
        }
    }

    printf("Normalize w...\n");
    for(uint32_t bin_beta_o = 0; bin_beta_o < BINS_INC_DIR; ++bin_beta_o)
    {
        for(uint32_t bin_n = 0; bin_n < BINS_N; ++bin_n)
        {
            float normalization = 0.0f;
            for(uint32_t bin_beta_i = 0; bin_beta_i < BINS_BETA_I; ++bin_beta_i)
            {
                for(uint32_t bin_gamma_i = 0; bin_gamma_i < BINS_PHI_I; ++bin_gamma_i)
                {
                    normalization += pdf_w[bin_beta_o + BINS_INC_DIR * bin_n + BINS_INC_DIR * BINS_N * bin_beta_i + BINS_INC_DIR * BINS_N * BINS_BETA_I * bin_gamma_i];
                }
            }

            if (Math::safeFloatEqual(normalization, 0.0f))
                continue;

            for(uint32_t bin_beta_i = 0; bin_beta_i < BINS_BETA_I; ++bin_beta_i)
            {
                for(uint32_t bin_gamma_i = 0; bin_gamma_i < BINS_PHI_I; ++bin_gamma_i)
                {
                    pdf_w[bin_beta_o + BINS_INC_DIR * bin_n + BINS_INC_DIR * BINS_N * bin_beta_i + BINS_INC_DIR * BINS_N * BINS_BETA_I * bin_gamma_i] /= normalization;
                }
            }
        }
    }

    printf("Writing to file...\n");
    std::ofstream pdf_n_file("pdf_n_test.gsdf", std::ios::out | std::ios::binary);
    pdf_n_file.write((char*)pdf_n, sizeof(float) * BINS_INC_DIR * BINS_N);
    pdf_n_file.close();

    std::ofstream pdf_x_file("pdf_x_test.gsdf", std::ios::out | std::ios::binary);
    pdf_x_file.write((char*)pdf_x, sizeof(float) * BINS_INC_DIR * BINS_N * BINS_THETA_I * BINS_PHI_I);
    pdf_x_file.close();

    std::ofstream pdf_w_file("pdf_w_test.gsdf", std::ios::out | std::ios::binary);
    pdf_w_file.write((char*)pdf_w, sizeof(float) * BINS_INC_DIR * BINS_N * BINS_BETA_I * BINS_GAMMA_I);
    pdf_w_file.close();

    printf("Done!\n");

    Memory::destroyHostArray<float>(pdf_n);
    Memory::destroyHostArray<float>(pdf_x);
    Memory::destroyHostArray<float>(pdf_w);
    Memory::destroyDeviceArray<float>(histogram_n);
    Memory::destroyDeviceArray<float>(histogram_x);
    Memory::destroyDeviceArray<float>(histogram_w);
}

struct FeatureVector
{
    float g;
    float effective_albedo;
    float coefficients[20];
    Vector3float out_pos;
};

inline __host__ __device__ float MAD(const float& albedo_p, const float& albedo_eff, const float& g)
{
    return 0.25f * g + 0.25f * albedo_p + albedo_eff;
}

__global__ void mc_kernel(const uint32_t N, const MediumSettings med, PathSummary* out_points)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();
    
    if(tid >= N)
    {
        return;
    }

    uint32_t seed = Math::tea<4>(tid, 0);

    Vector3float start_pos = sampleSphereUniform(seed);
    Vector3float inc_pos = start_pos;
    Vector3float direction = -1.0f * sampleHemisphereUniform(seed, start_pos); // Points towards surface
    Vector3float inc_dir = direction;
    Ray ray(start_pos + 0.001f * direction, direction);

    //Volume
    //float effective_albedo = Math::rnd(seed);
    float single_scatter_albedo = med.phi;   //no absorption for now (1.0f - expf(-8.0f * effective_albedo)) / (1.0f - expf(-8.0f));
    float g = med.g;                            //Fixed g for now Math::rnd(seed);
    float sigma_t = med.sigma;

    Vector3float out_position;

    while(true)
    {
        ray.traceNew(start_pos + 0.001f * direction, direction);

        float d = distanceToBoundary(ray.origin(), ray.direction());

        if (d == INFINITY)
        {
            out_position = ray.origin();
            break;
        }

        float t = -logf(1.0f - Math::rnd(seed)) / sigma_t;

        if (t >= d)
        {
            out_position = ray.origin() + d * ray.direction();
            break;
        }

        start_pos = ray.origin() + t * ray.direction();
        direction = sampleHenyeyGreensteinPhase(g, ray.direction(), seed);
    }

    out_points[tid].out_pos = out_position;
    out_points[tid].inc_pos = inc_pos;
    out_points[tid].inc_dir = inc_dir;
}

__device__ void addSample(float A[20][20], float n[20], Vector3float& bi, Vector3float& normal, const float& weight)
{
    float x = bi.x;
    float y = bi.y;
    float z = bi.z;

    float Pi[20] = {1.0, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z, x*x*x, x*x*y, x*x*z, x*y*y, x*y*z, x*z*z, y*y*y, y*y*z, y*z*z, z*z*z};
    float grad_Pi[20][20] =
    {
        {0, 1, 0, 0, 2 * x, y, z, 0, 0, 0, 3 * x * x, 2 * x * y, 2 * x * z, y * y, y * z, z * z, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, x, 0, 2 * y, z, 0, 0, x * x, 0, x * 2 * y, x * z, 0, 3 * y * y, 2 * y * z, z * z, 0},
        {0, 0, 0, 1, 0, 0, x, 0, y, 2 * z, 0, 0, x * x, 0, x * y, x * 2 * z, 0, y * y, y * 2 * z, 3 * z * z}
    };
    
    for(uint32_t i = 0; i < 20; ++i)
    {
        for(uint32_t j = 0; j < 20; ++j)
        {
            A[i][j] += weight * Pi[i] * Pi[j];
        }
    }

    for(uint32_t k = 0; k < 3; ++k)
    {
        for(uint32_t i = 0; i < 20; ++i)
        {
            n[i] += weight * grad_Pi[k][i] * normal[k];
            for(uint32_t j = 0; j < 20; ++j)
            {
                A[i][j] += weight * grad_Pi[k][i] * grad_Pi[k][j];
            }
        }
    }

}

__device__ void solve(float A[20][20], float n[20], float x[20])
{
    for(uint32_t i = 0; i < 20; ++i)
    {
        //Search max element
        float maxEl = fabsf(A[i][i]);
        uint32_t maxRow = i;
        for(uint32_t k = i+1;k < 20; ++k)
        {
            if(fabsf(A[k][i]) > maxEl)
            {
                maxEl = fabsf(A[k][i]);
                maxRow = k;
            }
        }
            
        /*if(Math::safeFloatEqual(maxEl, 0))
        {
            *solved = false;
            return;
        }*/
            
        //Swap row
        for(uint32_t k = i; k < 20; ++k)
        {
            float tmp = A[maxRow][k];
            A[maxRow][k] = A[i][k];
            A[i][k] = tmp;
        }
        
        //Swap in solution vector
        float tmp = n[maxRow];
        n[maxRow] = n[i];
        n[i] = tmp;
            
        //Make zeros
        for(uint32_t k = i+1; k < 20; ++k)
        {
            float c = -A[k][i]/A[i][i];
            for(uint32_t j = i; j < 20; ++j)
            {
                A[k][j] = i == j ? 0.0f : A[k][j] + c * A[i][j];
            }
            
            n[k] += c * n[i];
        }
    }
        
    //Backsubstitution
    for(int i = 20-1; i >= 0; --i)
    {
        x[i] = n[i]/A[i][i];
        for(int k = i-1; k>=0; --k)
        {
            n[k] -= A[k][i]*x[i];
        }
    }
}

__global__ void feature_kernel(const uint32_t N, 
                               const uint32_t m, 
                               const float sigma_n,
                               const Vector3float* samples, 
                               const PathSummary* summary, 
                               FeatureVector* features)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= N)
    {
        return;
    }


    float mu = 0.0001f;

    float A[20][20] = { 0, };
    float n[20] = { 0, };

    float cosa = -summary[tid].inc_dir.z;
    float sina = sqrtf(fmaxf(0.0f, 1.0f - cosa * cosa));
    float mcosa = 1.0f - cosa;

    Vector3float n_ = Math::normalize(Vector3float(-summary[tid].inc_dir.y, summary[tid].inc_dir.x, 0.0f));

    Vector3float col1 = Vector3float(
        n_.x*n_.x*mcosa + cosa,
        n_.y*n_.x*mcosa + n_.z*sina,
        n_.z*n_.x*mcosa - n_.y*sina
    );

    Vector3float col2 = Vector3float(
        n_.x*n_.y*mcosa - n_.z*sina,
        n_.y*n_.y*mcosa + cosa,
        n_.z*n_.y*mcosa + n_.x*sina
    );

    Vector3float col3 = Vector3float(
        n_.x*n_.z*mcosa + n_.y*sina,
        n_.y*n_.z*mcosa - n_.x*sina,
        n_.z*n_.z*mcosa + cosa
    );

    //TODO Soon™
    Matrix3x3float R(col1, col2, col3);
    if(Math::safeFloatEqual(summary[tid].inc_dir.z, 1.0f, 1e-8))
    {
        printf("Identity\n");
        R = Matrix3x3float(Vector3float(1, 0, 0), Vector3float(0, 1, 0), Vector3float(0, 0, 1));
    }

    features[tid].out_pos = R*(summary[tid].out_pos - summary[tid].inc_pos)/sigma_n;
    features[tid].g = 0.6f;
    features[tid].effective_albedo = 1.0f;

    for(uint32_t i = 0; i < m; ++i)
    {
        Vector3float bi = R * (samples[i] - summary[tid].inc_pos)/sigma_n;
        Vector3float normal = R * samples[i];
        float weight = expf(-Math::dot(bi, bi) / 2.0f);

        addSample(A, n, bi, normal, weight);
    }


    //Regularization
    for(uint32_t i = 0; i < 20; ++i)
    {
        A[i][i] += sqrtf(mu);
    }

    solve(A, n, features[tid].coefficients);

}

void generatePolyDataset()
{
    const uint32_t N = 1 << 24;

    printf("Generating %u samples...\n", N);

    MediumSettings med;
    med.g = 0.6f;
    med.phi = 1.0f;
    med.sigma = 3.0f;

    float sigma_t_reduced = (1.0f - med.g) * med.sigma;

    float sigma_n = 2.0f * MAD(med.phi, med.phi, med.g)/sigma_t_reduced;
    uint32_t m = std::max(1024u, static_cast<uint32_t>(2.0f * static_cast<float>(M_PI) / (sigma_n * sigma_n)));

    Vector3float* h_sample_points = Memory::createHostArray<Vector3float>(m);
    Vector3float* d_sample_points = Memory::createDeviceArray<Vector3float>(m);
    PathSummary* d_summary = Memory::createDeviceArray<PathSummary>(N);
    FeatureVector* d_features = Memory::createDeviceArray<FeatureVector>(N);
    FeatureVector* h_features = Memory::createHostArray<FeatureVector>(N);

    uint32_t seed = Math::tea<4>(434234, 48558641523);
    for(uint32_t i = 0; i < m; ++i)
    {
        h_sample_points[i] = sampleSphereUniform(seed);
    }

    Memory::copyHost2DeviceArray<Vector3float>(m, h_sample_points, d_sample_points);
    Memory::destroyHostArray<Vector3float>(h_sample_points);

    printf("Compute feature vectors...\n");
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    mc_kernel << <config.blocks, config.threads >> > (N, med, d_summary);
    cudaSafeCall(cudaDeviceSynchronize());

    feature_kernel << <config.blocks, config.threads >> > (N,
                                                           m,
                                                           sigma_n,
                                                           d_sample_points,
                                                           d_summary,
                                                           d_features);
    cudaSafeCall(cudaDeviceSynchronize());

    Memory::copyDevice2HostArray<FeatureVector>(N, d_features, h_features);
    Memory::destroyDeviceArray<FeatureVector>(d_features);

    printf("Writing to file...\n");

    std::ofstream descriptor_file("Descriptors.ds", std::ios::out | std::ios::binary);
    descriptor_file.write((char*)h_features, sizeof(FeatureVector) * N);
    descriptor_file.close();

    Memory::destroyHostArray<FeatureVector>(h_features);
    Memory::destroyDeviceArray<Vector3float>(d_sample_points);
    Memory::destroyDeviceArray<PathSummary>(d_summary);
}


int run()
{
    cudaSafeCall(cudaSetDevice(0));

    //generateDataSet();

    //generateGSDF();

    //generateTestHistograms();

    generatePolyDataset();

    return 0;
}

int main()
{
    int exit = run();
    return exit;
}