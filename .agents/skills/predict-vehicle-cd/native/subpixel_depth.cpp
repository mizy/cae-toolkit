// First-hit orthographic point samples, at explicit positions on a shifted sensor.
// Vertex coordinates and barycentric tolerances use the training renderer's units.
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" void sample_depth(const float* points, const std::int32_t* faces,
                             std::int64_t triangles, float* depth, int width,
                             int factor, int anchor, double camera_x, double camera_y,
                             double sample_x, double sample_y) {
    const double bx = 0.5 + sample_x - camera_x;
    const double by = 0.5 + sample_y - camera_y;
    for (std::int64_t i = 0; i < triangles; ++i) {
        const float* a = points + 3LL * faces[3 * i];
        const float* b = points + 3LL * faces[3 * i + 1];
        const float* c = points + 3LL * faces[3 * i + 2];
        const float x0 = std::min({a[0], b[0], c[0]}), x1 = std::max({a[0], b[0], c[0]});
        const float y0 = std::min({a[1], b[1], c[1]}), y1 = std::max({a[1], b[1], c[1]});
        const float tolerance = 2e-5f * std::max(x1 - x0, y1 - y0) + 1e-5f;
        const int left = std::max(0, int(std::ceil((x0 - bx - tolerance) * factor + anchor)));
        const int right = std::min(width - 1, int(std::floor((x1 - bx + tolerance) * factor + anchor)));
        const int top = std::max(0, int(std::ceil((y0 - by - tolerance) * factor + anchor)));
        const int bottom = std::min(width - 1, int(std::floor((y1 - by + tolerance) * factor + anchor)));
        const float denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        if (left > right || top > bottom || std::abs(denom) < 1e-8f) continue;
        for (int y = top; y <= bottom; ++y) {
            const float gy = float(double(y - anchor) / factor + by);
            for (int x = left; x <= right; ++x) {
                const float gx = float(double(x - anchor) / factor + bx);
                const float u = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / denom;
                const float v = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / denom;
                const float w = 1.f - u - v;
                if (u >= -1e-5f && v >= -1e-5f && w >= -1e-5f) {
                    const float z = u * a[2] + v * b[2] + w * c[2];
                    float& closest = depth[y * width + x];
                    closest = std::min(closest, z);
                }
            }
        }
    }
}
