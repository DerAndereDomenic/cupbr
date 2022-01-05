#include <iostream>
#include <fstream>

#include <CUPBR.h>

using namespace cupbr;

/*struct MediumSettings
{
    float sigma;
    float phi;
    float g;
};

struct PathSummary
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
};

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

__device__ float invertcdf(const float& g, const float& xi)
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
        x += w * d;
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

    Vector3float normx = r.x.x * xAxis + r.x.y * yAxis + r.x.z * zAxis;
    Vector3float normw = r.w.x * xAxis + r.w.y * yAxis + r.w.z * zAxis;
    Vector3float normX = r.X.x * xAxis + r.X.y * yAxis + r.X.z * zAxis;
    Vector3float normW = r.W.x * xAxis + r.W.y * yAxis + r.W.z * zAxis;
    
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

int run()
{
    cudaSafeCall(cudaSetDevice(0));

    generateDatasetForTrainingCVAE();

    return 0;
}

int main()
{
    int exit = run();
    return exit;
}*/

struct PathSummary
{
    Vector3float inc_dir;
    uint32_t num_scattering;
};

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

__device__ Vector3float sampleHemisphereUniform(uint32_t& seed, const Vector3float& N)
{
    float z = Math::rnd(seed) * 2.0f - 1.0f;
    float phi = Math::rnd(seed) * 2.0f * M_PI;

    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    Vector3float result(x, y, z);

    return Math::dot(result, N) < 0 ? -1.0f * result : result;
}

__global__ void generateSamples(const uint32_t num_samples, Sphere sphere, PathSummary* buffer)
{
    const uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= num_samples)
    {
        return;
    }

    uint32_t seed = Math::tea<4>(tid, 1);

    Vector3float start_pos = Vector3float(0.99, 0, 0); // Just inside the sphere
    Vector3float direction = sampleHemisphereUniform(seed, Vector3float(1, 0, 0));
    Ray ray(start_pos, direction);
    PathSummary summary = { direction, 0 };

    //Volume
    float sigma_s = 10.0f;
    float sigma_a = 0.0f;
    float sigma_t = sigma_s + sigma_a;
    float g = 0.9f;

    Vector3float path[100];
    path[0] = start_pos;

    while(true)
    {
        Vector3float inc_dir = -1.0f * ray.direction();

        Vector3float new_direction = sampleHenyeyGreensteinPhase(g, inc_dir, seed);
        ray.traceNew(start_pos + 0.001f * new_direction, new_direction);

        LocalGeometry geom = sphere.computeRayIntersection(ray);

        if (geom.depth == INFINITY)
        {
            //printf("%f %f %f\n", start_pos.x, start_pos.y, start_pos.z);
            break;
        }

        float t = -logf(1.0f - Math::rnd(seed)) / sigma_t;

        if (t >= geom.depth)
        {
            //printf("sample\n");
            break;
        }

        ++summary.num_scattering;
        start_pos = start_pos + t * ray.direction();
        //path[summary.num_scattering] = start_pos;
    }

    //for(int i = 0; i < summary.num_scattering + 1; ++i)
    //{
    //    printf("%f %f %f\n", path[i].x, path[i].y, path[i].z);
    //}

    buffer[tid] = summary;
}

void generateDataSet()
{
    const uint32_t N = 1 << 20;
    printf("Generating %i samples...\n", N);

    Sphere sphere(Vector3float(0, 0, 0), 1);
    
    PathSummary* buffer = Memory::createDeviceArray<PathSummary>(N);

    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(N);
    generateSamples << <config.blocks, config.threads >> > (N, sphere, buffer);
    cudaSafeCall(cudaDeviceSynchronize());

    PathSummary* host_buffer = Memory::createHostArray<PathSummary>(N);
    Memory::copyDevice2HostArray<PathSummary>(N, buffer, host_buffer);
    Memory::destroyDeviceArray<PathSummary>(buffer);

    std::ofstream file;
    file.open("ScattersDataSet.ds");

    for(uint32_t i = 0; i < N; ++i)
    {
        file << host_buffer[i].inc_dir.x << ", " <<
                host_buffer[i].inc_dir.y << ", " <<
                host_buffer[i].inc_dir.z << ", " <<
                host_buffer[i].num_scattering << "\n";
    }

    file.close();

    Memory::destroyHostArray<PathSummary>(host_buffer);
}

int run()
{
    cudaSafeCall(cudaSetDevice(0));

    generateDataSet();

    return 0;
}

int main()
{
    int exit = run();
    return exit;
}