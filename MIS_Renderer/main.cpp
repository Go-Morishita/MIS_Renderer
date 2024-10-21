//
//  main.cpp
//  gl3d_hello_world
//
//  Created by Yonghao Yue on 2019/09/28.
//  Copyright © 2019 Yonghao Yue. All rights reserved.
//

/*
* 2024/10/07
* このコードは2種類のサンプリング方法でやってるから（屈折光と直接の照明効果？）
* MISでそれを統合する。
* computeDirectLighting()（直接光照明の計算）
* computeDiffuseReflection()（拡散反射の計算）
* 直接光と拡散反射で測度を統一する必要がある。？
* 直接光と拡散反射でそれぞれ異なる乱数を使っているがそれを統一する必要がある。
* 被ったら0を返すようになっているけどそれはゼロの重みをかけるように変更する。OK？
* 
* 測度統一
* バランスヒューリスティック関数の実装
*/

#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_DONT_VECTORIZE

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include <unistd.h>
#else
#include <GL/freeglut.h>
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <iostream>

#include "Camera.h"
#include "Jpeg.h"
#include "TriMesh.h"

#include "GLPreview.h"
#include "random.h"

#include "optimal_heuristic.h"

enum MouseMode
{
    MM_CAMERA,
    MM_LIGHT,
};

struct RayTracingInternalData
{
    int nextPixel_i;
    int nextPixel_j;
};

struct RayHit
{
    double t;
    double alpha;
    double beta;
    int mesh_idx; // < -1: no intersection, -1: area light, >= 0: object_mesh
    int primitive_idx; // < 0: no intersection
    bool isFront;
};

const double __FAR__ = 1.0e33;
const int MAX_RAY_DEPTH = 8;

const int g_FilmWidth = 640;
const int g_FilmHeight = 480;
float* g_FilmBuffer = nullptr;
float* g_AccumulationBuffer = nullptr;
int* g_CountBuffer = nullptr;
GLuint g_FilmTexture = 0;

RayTracingInternalData g_RayTracingInternalData;

bool g_DrawFilm = true;

int width = 640;
int height = 480;
int nSamplesPerPixel = 1;

MouseMode g_MouseMode = MM_CAMERA;
int g_MM_LIGHT_idx = 0;
int mx, my;

double g_FrameSize_WindowSize_Scale_x = 1.0;
double g_FrameSize_WindowSize_Scale_y = 1.0;

Camera g_Camera;
std::vector<AreaLight> g_AreaLights;

Object g_Obj;

Eigen::Vector3d computeShading(const Ray& in_Ray, const RayHit& in_RayHit, const Object& in_Object, const std::vector<AreaLight>& in_AreaLights);

void initAreaLights()
{
    AreaLight light1;
    light1.pos << -1.2, 1.2, 1.2;
    light1.arm_u << 1.0, 0.0, 0.0;
    light1.arm_v = -light1.pos.cross(light1.arm_u);
    light1.arm_v.normalize();
    light1.arm_u = light1.arm_u * 0.3;
    light1.arm_v = light1.arm_v * 0.2;

    //light1.color << 1.0, 0.8, 0.3;
    light1.color << 1.0, 1.0, 1.0;
    //light1.intensity = 64.0;
    light1.intensity = 48.0;

    AreaLight light2;
    light2.pos << 1.2, 1.2, 0.0;
    light2.arm_u << 1.0, 0.0, 0.0;
    light2.arm_v = -light2.pos.cross(light2.arm_u);
    light2.arm_v.normalize();
    light2.arm_u = light2.arm_u * 0.3;
    light2.arm_v = light2.arm_v * 0.2;

    //light2.color << 0.3, 0.3, 1.0;
    light2.color << 1.0, 1.0, 1.0;
    //light2.intensity = 64.0;
    light2.intensity = 30.0;

    g_AreaLights.push_back(light1);
    g_AreaLights.push_back(light2);
}

void resetFilm()
{
    memset(g_AccumulationBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    memset(g_CountBuffer, 0, sizeof(int) * g_FilmWidth * g_FilmHeight);
}

void initFilm()
{
    g_FilmBuffer = (float*)malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_AccumulationBuffer = (float*)malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_CountBuffer = (int*)malloc(sizeof(int) * g_FilmWidth * g_FilmHeight);
    resetFilm();

    glGenTextures(1, &g_FilmTexture);
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_FilmWidth, g_FilmHeight, 0, GL_RGB, GL_FLOAT, g_FilmBuffer);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void updateFilm()
{
    for (int i = 0; i < g_FilmWidth * g_FilmHeight; i++)
    {
        if (g_CountBuffer[i] > 0)
        {
            g_FilmBuffer[i * 3] = g_AccumulationBuffer[i * 3] / g_CountBuffer[i];
            g_FilmBuffer[i * 3 + 1] = g_AccumulationBuffer[i * 3 + 1] / g_CountBuffer[i];
            g_FilmBuffer[i * 3 + 2] = g_AccumulationBuffer[i * 3 + 2] / g_CountBuffer[i];
        }
        else
        {
            g_FilmBuffer[i * 3] = 0.0;
            g_FilmBuffer[i * 3 + 1] = 0.0;
            g_FilmBuffer[i * 3 + 2] = 0.0;
        }
    }

    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_FilmWidth, g_FilmHeight, GL_RGB, GL_FLOAT, g_FilmBuffer);
}

void clearRayTracedResult()
{
    g_RayTracingInternalData.nextPixel_i = -1;
    g_RayTracingInternalData.nextPixel_j = 0;

    memset(g_FilmBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
}

void stepToNextPixel(RayTracingInternalData& io_data)
{
    io_data.nextPixel_i++;
    if (io_data.nextPixel_i >= g_FilmWidth)
    {
        io_data.nextPixel_i = 0;
        io_data.nextPixel_j++;

        if (io_data.nextPixel_j >= g_FilmHeight)
        {
            io_data.nextPixel_j = 0;
        }
    }
}

void rayTriangleIntersect(const TriMesh& in_Mesh, const int in_Triangle_idx, const Ray& in_Ray, RayHit& out_Result)
{
    out_Result.t = __FAR__;

    const Eigen::Vector3d v1 = in_Mesh.vertices[in_Mesh.triangles[in_Triangle_idx](0)];
    const Eigen::Vector3d v2 = in_Mesh.vertices[in_Mesh.triangles[in_Triangle_idx](1)];
    const Eigen::Vector3d v3 = in_Mesh.vertices[in_Mesh.triangles[in_Triangle_idx](2)];

    Eigen::Vector3d triangle_normal = (v1 - v3).cross(v2 - v3);
    triangle_normal.normalize();

    bool isFront = true;

    const double denominator = triangle_normal.dot(in_Ray.d);
    if (denominator >= 0.0)
        isFront = false;

    const double t = triangle_normal.dot(v3 - in_Ray.o) / denominator;

    if (t <= 0.0)
        return;

    const Eigen::Vector3d x = in_Ray.o + t * in_Ray.d;

    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = v1 - v3; A.col(1) = v2 - v3;

    Eigen::Matrix2d ATA = A.transpose() * A;
    const Eigen::Vector2d b = A.transpose() * (x - v3);

    const Eigen::Vector2d alpha_beta = ATA.inverse() * b;

    if (alpha_beta.x() < 0.0 || 1.0 < alpha_beta.x() || alpha_beta.y() < 0.0 || 1.0 < alpha_beta.y() || 1.0 - alpha_beta.x() - alpha_beta.y() < 0.0 || 1.0 < 1.0 - alpha_beta.x() - alpha_beta.y()) return;

    out_Result.t = t;
    out_Result.alpha = alpha_beta.x();
    out_Result.beta = alpha_beta.y();
    out_Result.isFront = isFront;
}

void rayAreaLightIntersect(const std::vector<AreaLight>& in_AreaLights, const int in_Light_idx, const Ray& in_Ray, RayHit& out_Result)
{
    out_Result.t = __FAR__;

    const Eigen::Vector3d pos = in_AreaLights[in_Light_idx].pos;
    const Eigen::Vector3d arm_u = in_AreaLights[in_Light_idx].arm_u;
    const Eigen::Vector3d arm_v = in_AreaLights[in_Light_idx].arm_v;

    Eigen::Vector3d light_normal = arm_u.cross(arm_v);
    light_normal.normalize();

    bool isFront = true;

    const double denominator = light_normal.dot(in_Ray.d);
    if (denominator >= 0.0)
        isFront = false;

    const double t = light_normal.dot(pos - in_Ray.o) / denominator;

    if (t <= 0.0)
        return;

    const Eigen::Vector3d x = in_Ray.o + t * in_Ray.d;

    // rescale uv coordinates such that they reside in [-1, 1], when the hit point is inside of the light.
    const double u = (x - pos).dot(arm_u) / arm_u.squaredNorm();
    const double v = (x - pos).dot(arm_v) / arm_v.squaredNorm();

    if (u < -1.0 || 1.0 < u || v < -1.0 || 1.0 < v) return;

    out_Result.t = t;
    out_Result.alpha = u;
    out_Result.beta = v;
    out_Result.isFront = isFront;
}

void rayTracing(const Object& in_Object, const std::vector<AreaLight>& in_AreaLights, const Ray& in_Ray, RayHit& io_Hit)
{
    double t_min = __FAR__;
    double alpha_I = 0.0, beta_I = 0.0;
    int mesh_idx = -99; int primitive_idx = -1;
    bool isFront = true;

    for (int m = 0; m < in_Object.meshes.size(); m++)
    {
        for (int k = 0; k < in_Object.meshes[m].triangles.size(); k++)
        {
            if (m == in_Ray.prev_mesh_idx && k == in_Ray.prev_primitive_idx) continue;

            RayHit temp_hit;
            rayTriangleIntersect(in_Object.meshes[m], k, in_Ray, temp_hit);
            if (temp_hit.t < t_min)
            {
                t_min = temp_hit.t;
                alpha_I = temp_hit.alpha;
                beta_I = temp_hit.beta;
                mesh_idx = m;
                primitive_idx = k;
                isFront = temp_hit.isFront;
            }
        }
    }

    for (int l = 0; l < in_AreaLights.size(); l++)
    {
        if (-1 == in_Ray.prev_mesh_idx && l == in_Ray.prev_primitive_idx) continue;

        RayHit temp_hit;
        rayAreaLightIntersect(in_AreaLights, l, in_Ray, temp_hit);
        if (temp_hit.t < t_min)
        {
            t_min = temp_hit.t;
            alpha_I = temp_hit.alpha;
            beta_I = temp_hit.beta;
            mesh_idx = -1;
            primitive_idx = l;
            isFront = temp_hit.isFront;
        }
    }

    io_Hit.t = t_min;
    io_Hit.alpha = alpha_I;
    io_Hit.beta = beta_I;
    io_Hit.mesh_idx = mesh_idx;
    io_Hit.primitive_idx = primitive_idx;
    io_Hit.isFront = isFront;
}

Eigen::Vector3d sampleRandomPoint(const AreaLight& in_Light)
{
    const double r1 = 2.0 * randomMT() - 1.0;
    const double r2 = 2.0 * randomMT() - 1.0;
    return in_Light.pos + r1 * in_Light.arm_u + r2 * in_Light.arm_v;
}

// randを変更
Eigen::Vector3d computeDirectLighting(const std::vector<AreaLight>& in_AreaLights, const Eigen::Vector3d& in_x, const Eigen::Vector3d& in_n, const Eigen::Vector3d& in_w_eye, const RayHit& in_ray_hit, const Object& in_Object, const Material& in_Material, const int depth)
{
    Eigen::Vector3d direct_light_contribution = Eigen::Vector3d::Zero();

    for (int i = 0; i < in_AreaLights.size(); i++)
    {
        const Eigen::Vector3d p_light = sampleRandomPoint(in_AreaLights[i]);
        Eigen::Vector3d w_L = p_light - in_x;
        const double dist = w_L.norm();
        w_L.normalize();

        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = n_light.norm() * 4.0;
        n_light.normalize();
        const double cosT_l = n_light.dot(-w_L);
        if (cosT_l <= 0.0) continue;

        // shadow test
        Ray ray; ray.o = in_x; ray.d = w_L; ray.depth = depth + 1;
        ray.prev_mesh_idx = in_ray_hit.mesh_idx; ray.prev_primitive_idx = in_ray_hit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, ray, rh);
        if (rh.mesh_idx < 0 && rh.primitive_idx == i)
        {
            // diffuse
            const double cos_theta = std::max<double>(0.0, w_L.dot(in_n));
            direct_light_contribution += area * in_AreaLights[i].color.cwiseProduct(in_Material.kd) * in_AreaLights[i].intensity * cos_theta * cosT_l / (M_PI * dist * dist);
        }
    }

    return direct_light_contribution;
}

Eigen::Vector3d computeRayHitNormal(const Object& in_Object, const RayHit& in_Hit)
{
    const int v1_idx = in_Object.meshes[in_Hit.mesh_idx].triangles[in_Hit.primitive_idx](0);
    const int v2_idx = in_Object.meshes[in_Hit.mesh_idx].triangles[in_Hit.primitive_idx](1);
    const int v3_idx = in_Object.meshes[in_Hit.mesh_idx].triangles[in_Hit.primitive_idx](2);

    const Eigen::Vector3d n1 = in_Object.meshes[in_Hit.mesh_idx].vertex_normals[v1_idx];
    const Eigen::Vector3d n2 = in_Object.meshes[in_Hit.mesh_idx].vertex_normals[v2_idx];
    const Eigen::Vector3d n3 = in_Object.meshes[in_Hit.mesh_idx].vertex_normals[v3_idx];

    const double gamma = 1.0 - in_Hit.alpha - in_Hit.beta;
    Eigen::Vector3d n = in_Hit.alpha * n1 + in_Hit.beta * n2 + gamma * n3;
    n.normalize();

    if (!in_Hit.isFront) n = -n;

    return n;
}

// randを変更
Eigen::Vector3d computeDiffuseReflection(const Eigen::Vector3d& in_x, const Eigen::Vector3d& in_n, const Eigen::Vector3d& in_w_eye, const RayHit& in_ray_hit, const Object& in_Object, const Material& in_Material, const std::vector<AreaLight>& in_AreaLights, const int depth, bool& hit_light)
{
    Eigen::Vector3d bn = in_Object.meshes[in_ray_hit.mesh_idx].vertices[in_Object.meshes[in_ray_hit.mesh_idx].triangles[in_ray_hit.primitive_idx].x()] - in_Object.meshes[in_ray_hit.mesh_idx].vertices[in_Object.meshes[in_ray_hit.mesh_idx].triangles[in_ray_hit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double theta = acos(sqrt(randomMT()));
    const double phi = randomMT() * 2.0 * M_PI;

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d w_L = _dx * bn + _dy * in_n + _dz * cn;
    w_L.normalize();

    Ray ray;
    ray.o = in_x; ray.d = w_L; ray.depth = depth + 1;
    ray.prev_mesh_idx = in_ray_hit.mesh_idx; ray.prev_primitive_idx = in_ray_hit.primitive_idx;

    RayHit new_ray_hit;
    rayTracing(in_Object, in_AreaLights, ray, new_ray_hit);
    
    // もし光源にぶつかったら hit_light を true にする。

    if (new_ray_hit.mesh_idx >= 0 && new_ray_hit.primitive_idx >= 0)
    {
        hit_light = false;
    }else{
        hit_light = true;
    }

    return computeShading(ray, new_ray_hit, in_Object, in_AreaLights);
}

Eigen::Vector3d computeReflection(const Eigen::Vector3d& in_x, const Eigen::Vector3d& in_n, const Eigen::Vector3d& in_w_eye, const RayHit& in_ray_hit, const Object& in_Object, const Material& in_Material, const std::vector<AreaLight>& in_AreaLights, const int depth)
{
    const double e_dot_n = in_w_eye.dot(in_n);
    Eigen::Vector3d w_L = 2.0 * in_n * e_dot_n - in_w_eye;
    w_L.normalize();

    Ray ray;
    ray.o = in_x; ray.d = w_L; ray.depth = depth + 1;
    ray.prev_mesh_idx = in_ray_hit.mesh_idx; ray.prev_primitive_idx = in_ray_hit.primitive_idx;

    RayHit new_ray_hit;
    rayTracing(in_Object, in_AreaLights, ray, new_ray_hit);

    if (new_ray_hit.primitive_idx >= 0)
    {
        return computeShading(ray, new_ray_hit, in_Object, in_AreaLights);
    }

    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d computeRefraction(const Eigen::Vector3d& in_x, const Eigen::Vector3d& in_n, const Eigen::Vector3d& in_w_eye, const RayHit& in_ray_hit, const Object& in_Object, const Material& in_Material, const std::vector<AreaLight>& in_AreaLights, const int depth)
{
    const double e_dot_n = in_w_eye.dot(in_n);
    const double eta = in_ray_hit.isFront ? in_Material.eta : 1.0 / in_Material.eta;

    const double inside_sqrt = 1.0 - eta * eta * (1.0 - e_dot_n * e_dot_n);
    if (inside_sqrt < 0.0)
    {
        return computeReflection(in_x, in_n, in_w_eye, in_ray_hit, in_Object, in_Material, in_AreaLights, depth);
    }
    else
    {
        Eigen::Vector3d w_t = -in_n * sqrt(inside_sqrt) - eta * (in_w_eye - e_dot_n * in_n);
        w_t.normalize();

        Ray ray;
        ray.o = in_x; ray.d = w_t; ray.depth = depth + 1;
        ray.prev_mesh_idx = in_ray_hit.mesh_idx; ray.prev_primitive_idx = in_ray_hit.primitive_idx;

        RayHit new_ray_hit;
        rayTracing(in_Object, in_AreaLights, ray, new_ray_hit);

        if (new_ray_hit.primitive_idx >= 0)
        {
            return computeShading(ray, new_ray_hit, in_Object, in_AreaLights);
        }

        return Eigen::Vector3d::Zero();
    }
}

Eigen::Vector3d sampleRandomDirection(const Eigen::Vector3d& in_n) {

    const double theta = acos(sqrt(randomMT()));
    const double phi = randomMT() * 2.0 * M_PI;

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d direction = Eigen::Vector3d(_dx, _dy, _dz);
    direction.normalize();

    return direction;
}

/* オプティマルヒューリスティック */
double optimal_weights(const double& p, const double& p1, const double& p2)
{
    return 0;
}

/* パワーヒューリスティック */
double power_weights(const double& p, const double& p1, const double& p2)
{
    return pow(p, 2) / (pow(p1, 2) + pow(p2, 2));
}


/* バランスヒューリスティック */
double balance_weights(const double& p, const double& p1, const double& p2)
{
    return p / (p1 + p2);
}


Eigen::Vector3d computeMISLighting(const std::vector<AreaLight>& in_AreaLights, const Eigen::Vector3d& in_x, const Eigen::Vector3d& in_n, const Eigen::Vector3d& in_w_eye, const RayHit& in_ray_hit, const Object& in_Object, const Material& in_Material, const int depth, const double kd)
{
    // 測度の統一
    Eigen::Vector3d sampled_direction = sampleRandomDirection(in_n);

    Eigen::Vector3d direct_lighting = computeDirectLighting(in_AreaLights, in_x, in_n, in_w_eye, in_ray_hit, in_Object, in_Material, depth);

    // 光源に当たったかの判定用
    bool hit_light = false;
    Eigen::Vector3d diffuse_reflection = computeDiffuseReflection(in_x, in_n, in_w_eye, in_ray_hit, in_Object, in_Material, in_AreaLights, depth, hit_light) / kd;

    // 直接寄与のPDF
    double p_direct = 0.0;
    for (int i = 0; i < in_AreaLights.size(); i++) {
        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        double area = n_light.norm() * 4.0;
        if (area > 0.0) {
            p_direct += 1.0 / area;
        }
    }

    // 拡散反射のPDF
    double cos_theta = std::max(0.0, in_n.dot(sampled_direction.normalized()));
    double p_diffuse = cos_theta / M_PI;

    double weight_direct = 0.0;
    double weight_diffuse = 0.0;

    if (hit_light == true)
    {
        weight_direct = 1.0;
        weight_diffuse = 0.0;
    }
    else
    {
        weight_direct = balance_weights(p_direct, p_direct, p_diffuse);

        weight_diffuse = balance_weights(p_diffuse, p_direct, p_diffuse);
    }
    
    
    printf("weight_direct:%f\n", weight_direct);
    printf("weight_diffuse:%f\n", weight_diffuse);

    printf("weight_direct + weight_diffuse:%f\n", weight_direct + weight_diffuse);

    return weight_direct * direct_lighting + weight_diffuse * diffuse_reflection;
}


/* 変更 */
Eigen::Vector3d computeShading(const Ray& in_Ray, const RayHit& in_RayHit, const Object& in_Object, const std::vector<AreaLight>& in_AreaLights)
{
    if (in_RayHit.mesh_idx < 0)
    {
        if (!in_RayHit.isFront)
            return Eigen::Vector3d::Zero();

        return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
    }

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

    const double kd_max = in_Object.meshes[in_RayHit.mesh_idx].material.kd.maxCoeff();

    return computeMISLighting(in_AreaLights, x, n, -in_Ray.d, in_RayHit, in_Object, in_Object.meshes[in_RayHit.mesh_idx].material, in_Ray.depth, kd_max);
}

void shadeNextPixel()
{
    stepToNextPixel(g_RayTracingInternalData);

    const int pixel_flat_idx = g_RayTracingInternalData.nextPixel_j * g_FilmWidth + g_RayTracingInternalData.nextPixel_i;

    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    for (int k = 0; k < nSamplesPerPixel; k++)
    {
        double p_x = (g_RayTracingInternalData.nextPixel_i + randomMT()) / g_FilmWidth;
        double p_y = (g_RayTracingInternalData.nextPixel_j + randomMT()) / g_FilmHeight;

        Ray ray; ray.depth = 0;
        g_Camera.screenView(p_x, p_y, ray);
        ray.prev_mesh_idx = -99; ray.prev_primitive_idx = -1;

        RayHit ray_hit;
        rayTracing(g_Obj, g_AreaLights, ray, ray_hit);

        if (ray_hit.primitive_idx >= 0)
        {
            I += computeShading(ray, ray_hit, g_Obj, g_AreaLights);
        }
    }

    g_AccumulationBuffer[pixel_flat_idx * 3] += I.x();
    g_AccumulationBuffer[pixel_flat_idx * 3 + 1] += I.y();
    g_AccumulationBuffer[pixel_flat_idx * 3 + 2] += I.z();
    g_CountBuffer[pixel_flat_idx] += nSamplesPerPixel;
}

void idle()
{
#ifdef __APPLE__
    //usleep( 1000*1000 / 60 );
    usleep(1000 / 60);
#else
    Sleep(1000.0 / 60.0);
#endif

    for (int i = 0; i < g_FilmWidth * g_FilmHeight; i++) shadeNextPixel();
    updateFilm();

    glutPostRedisplay();
}

void mouseDrag(int x, int y)
{
    int _dx = x - mx, _dy = y - my;
    mx = x; my = y;

    double dx = double(_dx) / double(width);
    double dy = -double(_dy) / double(height);

    if (g_MouseMode == MM_CAMERA)
    {
        double scale = 2.0;

        g_Camera.rotateCameraInLocalFrameFixLookAt(dx * scale);
        resetFilm();
        updateFilm();
        glutPostRedisplay();
    }
    else if (g_MouseMode == MM_LIGHT && g_MM_LIGHT_idx < g_AreaLights.size())
    {
        double scale = 1.0;

        g_AreaLights[g_MM_LIGHT_idx].pos += dx * scale * g_Camera.getXVector() + dy * scale * g_Camera.getYVector();
        resetFilm();
        updateFilm();
        glutPostRedisplay();
    }
}

void mouseDown(int x, int y)
{
    mx = x; my = y;
}

void mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
        mouseDown(x, y);
}

void key(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'C':
    case 'c':
        g_MouseMode = MM_CAMERA;
        break;
    case '1':
    case '2':
    case '3':
        g_MouseMode = MM_LIGHT;
        g_MM_LIGHT_idx = key - '1';
        break;
    case 'f':
    case 'F':
        g_DrawFilm = !g_DrawFilm;
        glutPostRedisplay();
        break;
    }
}

void display()
{
    glViewport(0, 0, width * g_FrameSize_WindowSize_Scale_x, height * g_FrameSize_WindowSize_Scale_y);

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    projection_and_modelview(g_Camera, width, height);

    glEnable(GL_DEPTH_TEST);

    drawFloor();

    computeGLShading(g_Obj, g_AreaLights);
    drawObject(g_Obj);

    drawLights(g_AreaLights);

    if (g_DrawFilm)
        drawFilm(g_Camera, g_FilmTexture);

    glDisable(GL_DEPTH_TEST);

    glutSwapBuffers();
}

void resize(int w, int h)
{
    width = w;
    height = h;
}

void saveJpeg(const char* filename, int width, int height, unsigned char* imageData)
{
    FILE* outfile;
    errno_t err = fopen_s(&outfile, filename, "wb");
    if (err != 0)
    {
        fprintf(stderr, "Error: could not open %s for writing.\n", filename);
        return;
    }

    // Create a jpeg compression object
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Setup error handling
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Specify the destination file
    jpeg_stdio_dest(&cinfo, outfile);

    // Set parameters for compression
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3; // Number of color components (RGB)
    cinfo.in_color_space = JCS_RGB; // Color space (RGB)

    // Set default compression parameters
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 85, TRUE); // Quality of 85 is usually good

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    // Write scanlines one by one
    while (cinfo.next_scanline < cinfo.image_height)
    {
        unsigned char* row_pointer = &imageData[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    // Finish compression
    jpeg_finish_compress(&cinfo);

    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

void saveScreenshot(const char* filename)
{
    int screenshot_width = g_FilmWidth;
    int screenshot_height = g_FilmHeight;
    std::vector<unsigned char> pixels(screenshot_width * screenshot_height * 3);

    glReadPixels(0, 0, screenshot_width, screenshot_height, GL_RGB, GL_UNSIGNED_BYTE, &pixels[0]);

    // Flip the image vertically (OpenGL stores pixels from bottom-left)
    for (int y = 0; y < screenshot_height / 2; ++y)
    {
        for (int x = 0; x < screenshot_width * 3; ++x)
        {
            std::swap(pixels[y * screenshot_width * 3 + x], pixels[(screenshot_height - y - 1) * screenshot_width * 3 + x]);
        }
    }

    // Save the pixel data to a JPEG file
    saveJpeg(filename, screenshot_width, screenshot_height, &pixels[0]);
}

// Callback function for the timer
void timer(int value)
{
    if (value == 0)
    {
        saveScreenshot("screenshot.jpg");
        std::cout << "Screenshot saved as screenshot.jpg" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    g_Camera.setEyePoint(Eigen::Vector3d{ 0.0, 1.0, 4.0 });
    g_Camera.lookAt(Eigen::Vector3d{ 0.0, 0.5, 0.0 }, Eigen::Vector3d{ 0.0, 1.0, 0.0 });
    initAreaLights();

    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutCreateWindow("Hello world!!");

    // With retina display, frame buffer size is twice the window size.
    // Viewport size should be set on the basis of the frame buffer size, rather than the window size.
    // g_FrameSize_WindowSize_Scale_x and g_FrameSize_WindowSize_Scale_y account for this factor.
    GLint dims[4] = { 0 };
    glGetIntegerv(GL_VIEWPORT, dims);
    g_FrameSize_WindowSize_Scale_x = double(dims[2]) / double(width);
    g_FrameSize_WindowSize_Scale_y = double(dims[3]) / double(height);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseDrag);
    glutKeyboardFunc(key);

    initFilm();
    resetFilm();
    clearRayTracedResult();
    loadObj("box2.obj", g_Obj);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glutTimerFunc(20000, timer, 0);

    glutMainLoop();
    return 0;
}
