// src/dxvk/rtx_render/rtx_fork_precipitation.cpp
//
// Fork-owned file. Implementation of the weather-driven precipitation particle
// system. See rtx_fork_precipitation.h for the design rationale and for how
// this relates to the NFS Carbon RTX mod's game-side rain.
//
// Structure of this file:
//   1. Procedural drop texture   — a soft radial droplet built on the CPU, so
//                                  the effect ships no assets and works in any
//                                  game the runtime is dropped into.
//   2. Emitter quad              — a unit quad registered as an external mesh.
//   3. Parameter resolution      — blended weather options -> fall direction,
//                                  speed, lifetime, spawn distance.
//   4. Descriptor build + dwell  — RtxParticleSystemDesc, adopted on a timer.
//   5. Per-frame submit          — one hidden external draw carrying the desc.
//   6. ImGui                     — the global (non-preset) knobs.

#include "rtx_fork_precipitation.h"

#include "rtx_fork_hooks.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_camera.h"
#include "rtx_options.h"
#include "rtx_particle_system.h"
#include "rtx_texture.h"
#include "rtx_imgui.h"
#include "imgui/imgui.h"

#include "../dxvk_buffer.h"
#include "../dxvk_device.h"
#include "../dxvk_image.h"
#include "../dxvk_util.h"

#include "../../util/log/log.h"
#include "../../util/util_global_time.h"
#include "../../util/util_string.h"
#include "../../util/xxHash/xxhash.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace dxvk { namespace fork_precipitation {

  namespace {

    // Fixed identities for the emitter mesh + material. The asset replacer keys
    // its external tables on these, and the particle manager folds the material
    // hash into its own system key, so both must be stable for the lifetime of
    // the process (a changing material hash forks the particle system exactly
    // like a changing descriptor would).
    constexpr uint64_t kMeshHandleValue     = 0x50524543495F4D53ull;  // "PRECI_MS"
    constexpr uint64_t kMaterialHandleValue = 0x50524543495F4D54ull;  // "PRECI_MT"
    constexpr uint64_t kDropTextureHash     = 0x50524543495F5458ull;  // "PRECI_TX"

    // Drop texture: small on purpose. It is a smooth falloff with no detail to
    // resolve, it is drawn at a few pixels across, and there can be tens of
    // thousands of them on screen — a big texture would only cost cache.
    constexpr uint32_t kDropTextureSize = 64;
    constexpr uint32_t kDropTextureMips = 7;  // 64,32,16,8,4,2,1

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kDegToRad = kPi / 180.0f;

    // Vertex layout for the emitter quad. Matches remixapi_HardcodedVertex so
    // the RasterBuffer strides/offsets below read exactly like the external-mesh
    // builder in rtx_remix_api.cpp.
    struct EmitterVertex {
      float    position[3];
      float    normal[3];
      float    texcoord[2];
      uint32_t color;  // B8G8R8A8_UNORM
    };

    float saturate(float v) {
      return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }

    // (fork - 2026-07-26) The previous "glow" emissive stand-in and its
    // sun-elevation gating were removed here: the drops' darkness was traced
    // to the froxel radiance cache containing no skylight at all, and the fix
    // moved to the correct layer - the resolver's particle lighting
    // approximation now adds a real sky-view-LUT ambient term for materials
    // flagged sky-lit (see ensureMaterial below, resolve.slangh, and the
    // per-preset skyLight field).

    Vector3 safeNormalize(const Vector3& v, const Vector3& fallback) {
      const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
      if (!(lenSq > 1e-12f)) {
        return fallback;
      }
      const float invLen = 1.0f / std::sqrt(lenSq);
      return Vector3(v.x * invLen, v.y * invLen, v.z * invLen);
    }

    // Any orthonormal pair perpendicular to n. Branch picks the world axis least
    // aligned with n so the cross product never degenerates.
    void buildTangentBasis(const Vector3& n, Vector3& outT1, Vector3& outT2) {
      const Vector3 reference = (std::fabs(n.z) < 0.9f) ? Vector3(0.0f, 0.0f, 1.0f)
                                                        : Vector3(1.0f, 0.0f, 0.0f);
      outT1 = safeNormalize(cross(reference, n), Vector3(1.0f, 0.0f, 0.0f));
      outT2 = cross(n, outT1);
    }

    // ---------------------------------------------------------------------
    // buildDropTexturePixels
    //
    // A radially symmetric droplet: opaque-ish core, smooth falloff to zero at
    // the rim, RGB left at white so the per-particle colour (which the particle
    // manager modulates in via VertexColor0) is the only tint. Keeping RGB at
    // 1.0 also sidesteps the UNORM-vs-sRGB question entirely — white is white
    // in both encodings.
    //
    // With motion trails enabled the runtime stretches only the CENTRE of the
    // sprite and preserves its edges, so this same round drop becomes a rain
    // streak with soft caps; with trails off it reads as a snowflake.
    //
    // Returns a tightly packed mip chain (level 0 first), box-filtered so
    // distant precipitation doesn't shimmer.
    // ---------------------------------------------------------------------
    std::vector<uint8_t> buildDropTexturePixels() {
      std::vector<uint8_t> mip0(kDropTextureSize * kDropTextureSize * 4);

      for (uint32_t y = 0; y < kDropTextureSize; ++y) {
        for (uint32_t x = 0; x < kDropTextureSize; ++x) {
          // Pixel centre in [-1, 1].
          const float u = (static_cast<float>(x) + 0.5f) / kDropTextureSize * 2.0f - 1.0f;
          const float v = (static_cast<float>(y) + 0.5f) / kDropTextureSize * 2.0f - 1.0f;
          const float r = std::sqrt(u * u + v * v);

          // smoothstep(1 -> 0) over the radius, squared for a tighter core.
          const float t = saturate(1.0f - r);
          const float smooth = t * t * (3.0f - 2.0f * t);
          const float alpha = smooth * smooth;

          uint8_t* px = &mip0[(y * kDropTextureSize + x) * 4];
          px[0] = 255;
          px[1] = 255;
          px[2] = 255;
          px[3] = static_cast<uint8_t>(saturate(alpha) * 255.0f + 0.5f);
        }
      }

      std::vector<uint8_t> chain;
      chain.reserve(mip0.size() * 2);
      chain.insert(chain.end(), mip0.begin(), mip0.end());

      // Box-filter down the chain. Source is always the previous level, which
      // lives at the tail of `chain` at the time we read it.
      const uint8_t* src = chain.data();
      uint32_t srcSize = kDropTextureSize;
      size_t srcOffset = 0;

      for (uint32_t mip = 1; mip < kDropTextureMips; ++mip) {
        const uint32_t dstSize = std::max(1u, srcSize / 2);
        const size_t dstOffset = chain.size();
        chain.resize(dstOffset + static_cast<size_t>(dstSize) * dstSize * 4);

        // resize() may reallocate, so re-derive the source pointer after it.
        src = chain.data() + srcOffset;
        uint8_t* dst = chain.data() + dstOffset;

        for (uint32_t y = 0; y < dstSize; ++y) {
          for (uint32_t x = 0; x < dstSize; ++x) {
            for (uint32_t c = 0; c < 4; ++c) {
              const uint32_t x0 = std::min(x * 2, srcSize - 1);
              const uint32_t y0 = std::min(y * 2, srcSize - 1);
              const uint32_t x1 = std::min(x * 2 + 1, srcSize - 1);
              const uint32_t y1 = std::min(y * 2 + 1, srcSize - 1);
              const uint32_t sum =
                  src[(y0 * srcSize + x0) * 4 + c] + src[(y0 * srcSize + x1) * 4 + c] +
                  src[(y1 * srcSize + x0) * 4 + c] + src[(y1 * srcSize + x1) * 4 + c];
              dst[(y * dstSize + x) * 4 + c] = static_cast<uint8_t>((sum + 2) / 4);
            }
          }
        }

        srcOffset = dstOffset;
        srcSize = dstSize;
      }

      return chain;
    }

  }  // anonymous namespace

  // ---------------------------------------------------------------------------
  // Singleton. One emitter identity exists per process (see the fixed handles
  // above), so the system is a singleton rather than a CommonDeviceObject.
  // ---------------------------------------------------------------------------
  PrecipitationSystem& PrecipitationSystem::get() {
    static PrecipitationSystem s_instance;
    return s_instance;
  }

  // ---------------------------------------------------------------------------
  // ensureTexture — create + upload the procedural drop texture.
  //
  // Mirrors fork_hooks::createTexture (rtx_fork_api_entry.cpp): device-local
  // image, host-visible staging buffer, per-mip copyBufferToImage, transition to
  // SHADER_READ_ONLY_OPTIMAL. We are already on the render thread with a live
  // DxvkContext, so the commands are recorded inline instead of via EmitCs.
  //
  // STAGING LIFETIME: the staging Rc<DxvkBuffer> is a local that dies when this
  // function returns, which is safe — DxvkContext::copyBufferToImage calls
  // m_cmd->trackResource<DxvkAccess::Read>(srcBuffer), so the command list holds
  // its own reference until the GPU has executed the copy. (The API's
  // createTexture only captures the buffer in its EmitCs lambda to carry it to
  // the CS thread; the lambda reference likewise dies at record time and the
  // same trackResource keeps the buffer alive for execution.)
  // ---------------------------------------------------------------------------
  bool PrecipitationSystem::ensureTexture(RtxContext& ctx) {
    if (m_dropImageView != nullptr) {
      return true;
    }

    const std::vector<uint8_t> pixels = buildDropTexturePixels();

    DxvkImageCreateInfo imageInfo = {};
    imageInfo.type        = VK_IMAGE_TYPE_2D;
    imageInfo.format      = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.flags       = 0;
    imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.extent      = { kDropTextureSize, kDropTextureSize, 1 };
    imageInfo.numLayers   = 1;
    imageInfo.mipLevels   = kDropTextureMips;
    imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.layout      = VK_IMAGE_LAYOUT_UNDEFINED;

    Rc<DxvkImage> image = ctx.getDevice()->createImage(
      imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXMaterialTexture,
      "RTX Precipitation - drop");

    if (image == nullptr) {
      Logger::err("[RTX Precipitation] failed to create drop texture image");
      return false;
    }

    DxvkBufferCreateInfo stagingInfo = {};
    stagingInfo.size   = pixels.size();
    stagingInfo.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    stagingInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;

    Rc<DxvkBuffer> stagingBuffer = ctx.getDevice()->createBuffer(
      stagingInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "RTX Precipitation - drop staging");

    if (stagingBuffer == nullptr) {
      Logger::err("[RTX Precipitation] failed to create drop texture staging buffer");
      return false;
    }

    auto stagingSlice = DxvkBufferSlice { stagingBuffer };
    std::memcpy(stagingSlice.mapPtr(0), pixels.data(), pixels.size());

    DxvkImageViewCreateInfo viewInfo = {};
    viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format    = imageInfo.format;
    viewInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT;
    viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel  = 0;
    viewInfo.numLevels = kDropTextureMips;
    viewInfo.minLayer  = 0;
    viewInfo.numLayers = 1;

    Rc<DxvkImageView> imageView = ctx.getDevice()->createImageView(image, viewInfo);
    if (imageView == nullptr) {
      Logger::err("[RTX Precipitation] failed to create drop texture view");
      return false;
    }

    ctx.changeImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkDeviceSize offset = 0;
    for (uint32_t mip = 0; mip < kDropTextureMips; ++mip) {
      VkImageSubresourceLayers subresource = {};
      subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.mipLevel       = mip;
      subresource.baseArrayLayer = 0;
      subresource.layerCount     = 1;

      const VkExtent3D   extent  = util::computeMipLevelExtent(imageInfo.extent, mip);
      const VkDeviceSize mipSize = util::computeImageDataSize(imageInfo.format, extent);

      ctx.copyBufferToImage(image, subresource, VkOffset3D { 0, 0, 0 }, extent,
                            stagingBuffer, offset, 0, 0);
      offset += mipSize;
    }

    ctx.changeImageLayout(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // The texture table keys on the image hash (see TextureRef::getImageHash),
    // so stamp our own rather than leaving it at whatever the data hash gives.
    image->setHash(kDropTextureHash);

    m_dropImage = image;
    m_dropImageView = imageView;
    return true;
  }

  // ---------------------------------------------------------------------------
  // ensureMaterial — register the drop material with the asset replacer under
  // the fixed material handle.
  //
  // Deliberately neutral: white albedo, fully opaque, legacy alpha state. All
  // of the weather-driven look (colour, opacity) rides on the per-particle
  // colour, which RtxParticleSystemManager::submitDrawState modulates against
  // this material via VertexColor0. That keeps the material — and therefore its
  // hash, and therefore the particle system identity — constant across weather
  // changes.
  //
  // Because it goes through the external-material table it is also replaceable:
  // a mod can author a USD material against the material hash and get bespoke
  // raindrops without touching the runtime.
  // ---------------------------------------------------------------------------
  bool PrecipitationSystem::ensureMaterial(RtxContext& ctx) {
    if (m_materialRegistered) {
      return true;
    }
    if (m_dropImageView == nullptr) {
      return false;
    }

    // Record what we actually registered so refreshDesc can detect live edits
    // of the material options (registration is otherwise once-per-process and
    // later edits would silently do nothing).
    m_materialRoughness = saturate(roughness());
    m_materialMetallic = saturate(metallic());

    OpaqueMaterialData opaque;
    opaque.setAlbedoOpacityTexture(TextureRef { m_dropImageView, kDropTextureHash });
    opaque.setAlbedoConstant(Vector3(1.0f, 1.0f, 1.0f));
    opaque.setOpacityConstant(1.0f);
    opaque.setRoughnessConstant(m_materialRoughness);
    opaque.setMetallicConstant(m_materialMetallic);

    // Sky-lit particle flag (fork - 2026-07-26). Particle-category surfaces
    // are shaded by the resolver's opacity lighting approximation (albedo x
    // froxel volumetric radiance), and the froxel integrator carries no sky
    // term - which left rain black under overcast daylight. This flag opts
    // the drop material into the resolver's sky-view-LUT ambient term
    // (resolve.slangh, gated by rtx.weather.precipitation.appearance.
    // per-preset skyLight field via cb.particleSkyAmbientScale). Safe for THIS system
    // specifically because the spawn-time shelter probe guarantees every
    // living drop saw open sky at birth.
    opaque.setSkyLitParticle(true);
    // Alpha/blend state comes from the emitter draw call (which we set up in
    // submit()), matching how the NFSC mod drives its spawner material.
    opaque.setUseLegacyAlphaState(true);

    auto handle = reinterpret_cast<remixapi_MaterialHandle>(kMaterialHandleValue);
    ctx.getSceneManager().getAssetReplacer()->makeMaterialWithTexturePreload(
      ctx, handle, MaterialData { opaque });

    m_materialRegistered = true;
    return true;
  }

  // ---------------------------------------------------------------------------
  // ensureMesh — build and register the emitter quad.
  //
  // A UNIT quad (±1 in local XY, normal +Z). Everything physical — radius,
  // orientation, position — lives in the per-frame transform, so changing the
  // spawn radius never has to rebuild geometry.
  //
  // Winding matters more than usual here. The spawn shader derives the emission
  // direction as cross(normalize(p1-p0), normalize(p2-p0)) and uses it
  // UNNORMALIZED to scale initial velocity, so both triangles must be wound off
  // perpendicular unit-length edges or the two halves of the quad would emit at
  // different speeds. Indices {0,1,2} and {3,2,1} both give exactly +Z.
  // ---------------------------------------------------------------------------
  bool PrecipitationSystem::ensureMesh(RtxContext& ctx) {
    if (m_meshRegistered) {
      return true;
    }

    const EmitterVertex vertices[4] = {
      { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f }, 0xFFFFFFFFu },
      { {  1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f }, 0xFFFFFFFFu },
      { { -1.0f,  1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f }, 0xFFFFFFFFu },
      { {  1.0f,  1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f }, 0xFFFFFFFFu },
    };
    const uint32_t indices[6] = { 0, 1, 2, 3, 2, 1 };

    auto allocBuffer = [&ctx](size_t sizeInBytes, const char* name) -> Rc<DxvkBuffer> {
      DxvkBufferCreateInfo bufferInfo = {};
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                       | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      bufferInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                        | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      bufferInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      bufferInfo.size = align(sizeInBytes, CACHE_LINE_SIZE);
      return ctx.getDevice()->createBuffer(
        bufferInfo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::RTXBuffer,
        name);
    };

    m_quadVertexBuffer = allocBuffer(sizeof(vertices), "RTX Precipitation - emitter VB");
    m_quadIndexBuffer  = allocBuffer(sizeof(indices),  "RTX Precipitation - emitter IB");
    if (m_quadVertexBuffer == nullptr || m_quadIndexBuffer == nullptr) {
      Logger::err("[RTX Precipitation] failed to allocate emitter geometry buffers");
      m_quadVertexBuffer = nullptr;
      m_quadIndexBuffer = nullptr;
      return false;
    }

    auto vertexSlice = DxvkBufferSlice { m_quadVertexBuffer };
    auto indexSlice  = DxvkBufferSlice { m_quadIndexBuffer };
    std::memcpy(vertexSlice.mapPtr(0), vertices, sizeof(vertices));
    std::memcpy(indexSlice.mapPtr(0), indices, sizeof(indices));

    RasterGeometry geometry;
    geometry.externalMaterial = reinterpret_cast<remixapi_MaterialHandle>(kMaterialHandleValue);
    geometry.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    geometry.cullMode = VK_CULL_MODE_NONE;
    geometry.frontFace = VK_FRONT_FACE_CLOCKWISE;
    geometry.vertexCount = 4;
    geometry.indexCount = 6;
    geometry.positionBuffer = RasterBuffer { vertexSlice, offsetof(EmitterVertex, position), sizeof(EmitterVertex), VK_FORMAT_R32G32B32_SFLOAT };
    geometry.normalBuffer   = RasterBuffer { vertexSlice, offsetof(EmitterVertex, normal),   sizeof(EmitterVertex), VK_FORMAT_R32G32B32_SFLOAT };
    geometry.texcoordBuffer = RasterBuffer { vertexSlice, offsetof(EmitterVertex, texcoord), sizeof(EmitterVertex), VK_FORMAT_R32G32_SFLOAT };
    geometry.color0Buffer   = RasterBuffer { vertexSlice, offsetof(EmitterVertex, color),    sizeof(EmitterVertex), VK_FORMAT_B8G8R8A8_UNORM };
    geometry.indexBuffer    = RasterBuffer { indexSlice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32 };
    geometry.boundingBox.minPos = Vector3(-1.0f, -1.0f, 0.0f);
    geometry.boundingBox.maxPos = Vector3( 1.0f,  1.0f, 0.0f);

    // Fixed hashes: the geometry never changes, and a stable identity keeps the
    // emitter matching to the same RtInstance frame over frame.
    geometry.hashes[HashComponents::Indices]            = kMeshHandleValue ^ 0x01ull;
    geometry.hashes[HashComponents::VertexPosition]     = kMeshHandleValue ^ 0x02ull;
    geometry.hashes[HashComponents::VertexTexcoord]     = kMeshHandleValue ^ 0x03ull;
    geometry.hashes[HashComponents::GeometryDescriptor] = kMeshHandleValue ^ 0x04ull;
    geometry.hashes[HashComponents::VertexLayout]       = kMeshHandleValue ^ 0x05ull;
    geometry.hashes.precombine();

    std::vector<RasterGeometry> submeshes;
    submeshes.push_back(std::move(geometry));

    ctx.getSceneManager().getAssetReplacer()->registerExternalMesh(
      reinterpret_cast<remixapi_MeshHandle>(kMeshHandleValue), std::move(submeshes));

    m_meshRegistered = true;
    return true;
  }

  bool PrecipitationSystem::ensureResources(RtxContext& ctx) {
    return ensureTexture(ctx) && ensureMaterial(ctx) && ensureMesh(ctx);
  }

  // ---------------------------------------------------------------------------
  // resolveParams — blended weather options -> the geometry of the fall.
  //
  // Wind: the horizontal drift borrows the atmosphere's cloud wind so the rain
  // slants the same way the sky is moving. cloudWindSpeed is stored in km/s and
  // the direction is the same (cos, sin) convention the cloud advection uses
  // (RtxAtmosphere::advanceCloudMotion), mapped onto the scene's right/forward
  // axes.
  // ---------------------------------------------------------------------------
  PrecipitationSystem::Params PrecipitationSystem::resolveParams() {
    Params params;

    const Vector3 up      = SceneManager::getSceneUp();
    const Vector3 forward = SceneManager::getSceneForward();
    const Vector3 right   = SceneManager::calculateSceneRight();

    const float fallSpeedMs = std::max(fallSpeed(), 0.05f);

    const float windAngleRad = RtxOptions::cloudWindDirection() * kDegToRad;
    const float windSpeedMs  = std::max(RtxOptions::cloudWindSpeed(), 0.0f) * 1000.0f;  // km/s -> m/s
    const float horizontalMs = windSpeedMs * std::max(windResponse(), 0.0f);

    const Vector3 windDir = right * std::cos(windAngleRad) + forward * std::sin(windAngleRad);

    // Travel = straight down at fallSpeed, pushed sideways by the wind.
    const Vector3 velocity = (up * -fallSpeedMs) + (windDir * horizontalMs);
    const float speedMs = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);

    params.fallDirection = safeNormalize(velocity, up * -1.0f);
    params.speedMetersPerSec = std::max(speedMs, 0.05f);

    // How far along the travel direction the spawn plane sits so that it is
    // spawnHeightMeters ABOVE the camera. Clamped so a near-horizontal fall
    // (huge wind, tiny fall speed) can't push the plane to infinity.
    const float downComponent = std::max(-dot(params.fallDirection, up), 0.15f);
    const float heightM = std::max(spawnHeightMeters(), 0.5f);
    params.spawnDistanceWorld = (heightM / downComponent) * RtxOptions::getMeterToWorldUnitScale();

    // Lifetime: enough to fall from the plane to fallMarginMeters below the
    // camera. This is what recycles the budget, so it directly sets density.
    const float travelM = (heightM + std::max(fallMarginMeters(), 0.0f)) / downComponent;
    params.timeToLiveSec = std::max(travelM / params.speedMetersPerSec, 0.05f);

    return params;
  }

  // ---------------------------------------------------------------------------
  // buildDesc — the RtxParticleSystemDesc for the current look.
  //
  // Unit notes, because the particle system mixes conventions:
  //   * velocities / sizes / forces are in CENTIMETRES (the shader multiplies
  //     them by rtx.sceneScale itself),
  //   * positions (attractor, and our emitter transform) are in world units.
  //
  // Drag/gravity coupling: the evolve shader integrates
  //   v = (v + up * gravity * dt) * (1 - drag * dt)
  // so the steady state is gravity/drag, and gravity acts on the VERTICAL axis
  // only. Setting gravity = -fallSpeed * drag (the vertical component of the
  // spawn velocity, NOT the total speed, which also contains the wind push)
  // makes the spawn velocity the terminal velocity too: particles neither
  // accelerate nor stall, and turbulence kicks decay back toward the fall speed.
  // That is the difference between snow (drag + turbulence: flutters, then
  // resumes falling) and rain (no drag: dead straight).
  //
  // KNOWN CONSEQUENCE: there is no horizontal force to balance drag against, so
  // drag also bleeds the wind push off over a particle's flight with time
  // constant 1/drag. That is why the wind-driven presets (blizzard, sandstorm)
  // use LOW drag with high turbulence rather than the reverse — high drag would
  // straighten them to a vertical fall within a second of spawning.
  // ---------------------------------------------------------------------------
  RtxParticleSystemDesc PrecipitationSystem::buildDesc(const Params& params) {
    RtxParticleSystemDesc desc;

    const float speedCmS = params.speedMetersPerSec * 100.0f;
    const float verticalSpeedCmS = std::max(fallSpeed(), 0.05f) * 100.0f;
    const float dragCoeff = std::max(drag(), 0.0f);
    const int   budget = std::max(maxParticles(), 1);

    desc.maxNumParticles = static_cast<uint32_t>(budget);

    // Population = spawnRate * lifetime, so this saturates the budget at
    // intensity 1 and scales linearly below it.
    //
    // CAPPED below the budget: RtxParticleSystemManager::spawnParticles treats
    // spawnRatePerSecond >= maxNumParticles as "constant population" mode and
    // re-initializes EVERY particle EVERY frame (spawnParticleCount = max, the
    // whole buffer re-runs the spawn kernel) — precipitation would freeze into
    // a static sheet at the spawn plane. budget/ttl crosses that threshold
    // whenever ttl < 1 s (fast fall speed + a small spawn volume), so clamp
    // with margin. The only cost is a slightly under-filled budget for such
    // extreme configurations.
    desc.spawnRatePerSecond = std::min(
      saturate(intensity()) * static_cast<float>(budget) / params.timeToLiveSec,
      0.9f * static_cast<float>(budget));
    desc.spawnBurstDuration = 0.0f;

    desc.minTimeToLive = params.timeToLiveSec;
    desc.maxTimeToLive = params.timeToLiveSec;

    desc.initialVelocityFromNormal = speedCmS;
    desc.initialVelocityConeAngleDegrees = std::max(spreadDegrees(), 0.0f);
    // The emitter is glued to the camera; inheriting its motion would fling
    // every drop sideways whenever the player moves.
    desc.initialVelocityFromMotion = 0.0f;

    desc.dragCoefficient = dragCoeff;
    desc.gravityForce = -verticalSpeedCmS * dragCoeff;

    const float turbulenceCmS2 = std::max(turbulence(), 0.0f) * 100.0f;
    desc.useTurbulence = turbulenceCmS2 > 0.0f ? 1 : 0;
    desc.turbulenceForce = turbulenceCmS2;
    desc.turbulenceFrequency = std::max(turbulenceFrequency(), 1e-4f);

    // Appearance overrides (fork - 2026-07-26): global user-taste multipliers
    // stacked on the blended per-preset look - see the option block in the
    // header for why these exist (the blender owns the per-preset fields, so
    // only a separate layer can stick while weather is active).
    const float trail = std::max(streak() * std::max(streakScale(), 0.0f), 0.0f);
    desc.enableMotionTrail = trail > 0.01f ? 1 : 0;
    desc.motionTrailMultiplier = std::max(trail, 0.01f);
    // Motion trails already imply velocity alignment; without them we want the
    // drop to stay camera-facing and un-rotated (snow reads better that way
    // than spinning).
    desc.alignParticlesToVelocity = desc.enableMotionTrail ? 1 : 0;

    // Billboard choice matters a lot for how "anchored" precipitation reads
    // (fork — 2026-07-25, fixing "the rain changes direction when I turn").
    //
    // FaceCamera_Spherical puts the billboard plane in the CAMERA plane
    // (basisRight/basisUp = camera right/up). The motion trail then stretches
    // along the world velocity PROJECTED into that plane, so any horizontal
    // component of the fall gets re-projected as the camera yaws and the
    // apparent fall direction sweeps around with the view.
    //
    // FaceCamera_UpAxisLocked pins basisUp to world up and only yaws the quad
    // to face the viewer, so a falling streak stays locked to the world
    // vertical no matter where the camera looks. That is what streaked
    // precipitation wants.
    //
    // Round, un-streaked particles (snow) have no orientation to betray, and
    // spherical gives them a better silhouette when looking up or down, so they
    // keep it.
    desc.billboardType = desc.enableMotionTrail
      ? ParticleBillboardType::FaceCamera_UpAxisLocked
      : ParticleBillboardType::FaceCamera_Spherical;
    desc.spriteSheetMode = ParticleSpriteSheetMode::UseMaterialSpriteSheet;
    desc.spriteSheetRows = 1;
    desc.spriteSheetCols = 1;
    desc.randomFlipAxis = ParticleRandomFlipAxis::None;
    desc.initialRotationDeviationDegrees = 0.0f;

    // Spawn-time TLAS occlusion: the mechanism that actually makes interiors
    // work, since it is view-independent (see the option description and the
    // traceSpawnOcclusion comment in particle_system_common.h).
    desc.traceSpawnOcclusion = occludeUnderCover() ? 1 : 0;

    desc.enableCollisionDetection = enableCollision() ? 1 : 0;
    desc.collisionMode = ParticleCollisionMode::Kill;
    desc.collisionThickness = std::max(collisionThickness(), 0.0f);
    desc.collisionRestitution = 0.0f;

    desc.hideEmitter = debugShowEmitter() ? 0 : 1;
    desc.useSpawnTexcoords = 0;
    desc.restrictVelocityX = 0;
    desc.restrictVelocityY = 0;
    desc.restrictVelocityZ = 0;

    desc.attractorPosition = Vector3(0.0f, 0.0f, 0.0f);
    desc.attractorForce = 0.0f;
    desc.attractorRadius = 0.0f;

    // Safety clamp. Turbulence integrates into velocity with only drag to
    // oppose it, so a low-drag/high-turbulence preset could otherwise fling
    // particles arbitrarily fast. 3x the intended speed leaves plenty of room
    // for gusting while bounding the worst case. (Per world axis; a zero or
    // negative entry would mean unclamped.)
    const float velocityClampCmS = std::max(speedCmS * 3.0f, 100.0f);
    desc.maxVelocity = { vec3(velocityClampCmS, velocityClampCmS, velocityClampCmS) };

    // Animated curves: two identical keyframes (spawn, end-of-life) so size
    // and colour hold constant over each particle's life. The min/max PAIR of
    // curves is where per-drop randomness lives: the GPU samples the
    // animation texture between the min row and the max row with the
    // particle's stable randSeed (computeDataRow, randomizeAcrossTwoRows), so
    // spreading min/max apart by the variance options gives every drop its
    // own size/opacity, fixed for its lifetime, at zero extra cost.
    const float width = std::max(dropWidth() * std::max(widthScale(), 0.0f), 0.01f);
    const float length = std::max(dropLength() * std::max(lengthScale(), 0.0f), 0.01f);
    const float sizeVar = std::min(std::max(sizeVariance(), 0.0f), 0.9f);
    const vec2 sizeLo(width * (1.0f - sizeVar), length * (1.0f - sizeVar));
    const vec2 sizeHi(width * (1.0f + sizeVar), length * (1.0f + sizeVar));
    desc.minSize = { sizeLo, sizeLo };
    desc.maxSize = { sizeHi, sizeHi };

    const Vector3 presetColor = color();
    const Vector3 tintOverride = tintColor();
    const Vector3 tint(saturate(presetColor.x * std::max(tintOverride.x, 0.0f)),
                       saturate(presetColor.y * std::max(tintOverride.y, 0.0f)),
                       saturate(presetColor.z * std::max(tintOverride.z, 0.0f)));
    const float alpha = saturate(opacity() * std::max(opacityScale(), 0.0f));
    const float alphaVar = std::min(std::max(opacityVariance(), 0.0f), 0.9f);
    const vec4 rgbaLo(tint.x, tint.y, tint.z, saturate(alpha * (1.0f - alphaVar)));
    const vec4 rgbaHi(tint.x, tint.y, tint.z, saturate(alpha * (1.0f + alphaVar)));
    desc.minColor = { rgbaLo, rgbaLo };
    desc.maxColor = { rgbaHi, rgbaHi };

    desc.minRotationSpeed = { 0.0f, 0.0f };
    desc.maxRotationSpeed = { 0.0f, 0.0f };

    return desc;
  }

  // ---------------------------------------------------------------------------
  // refreshDesc — dwell-gated adoption of a new descriptor.
  //
  // See the DESCRIPTOR STABILITY note in the header: the particle manager keys
  // systems on the descriptor hash, so adopting a freshly computed descriptor
  // every frame during a weather blend would allocate a full particle system per
  // frame. Holding each descriptor for descUpdateIntervalMs bounds that to a
  // handful of systems per transition, and the orphans crossfade out on their
  // own (no new spawns, existing particles finish their lifetime, the manager
  // evicts once the last one dies).
  //
  // The dwell is FLOORED at maxTimeToLive/6. Orphans live for their remaining
  // particle lifetime (prepareForNextFrame erases a system maxTimeToLive after
  // its last spawn), so the number of systems alive at once during a continuous
  // blend is ~lifetime/dwell + 1; a flat 750 ms dwell with slow-falling snow
  // (~20 s lifetime, ~7.5 MB of buffers per system at the default budget) would
  // stack ~27 systems = ~200 MB of transient VRAM plus 27 simulate dispatch
  // groups per frame. The floor bounds that at ~7 systems for ANY preset while
  // leaving fast-lifetime rain at the configured interval. descUpdateIntervalMs
  // = 0 disables the dwell AND the floor (debugging only).
  //
  // Material refresh rides the same dwell: re-registering the drop material
  // changes its MaterialData hash, and submitExternalDraw stamps that hash onto
  // the emitter draw call (setHashOverride), which is exactly the hash the
  // particle manager folds into the system key — i.e. a material edit forks the
  // system identity just like a descriptor edit and would allocate a system per
  // frame if applied while a slider is being dragged.
  // ---------------------------------------------------------------------------
  const RtxParticleSystemDesc& PrecipitationSystem::refreshDesc(RtxContext& ctx, const Params& params) {
    const uint64_t nowMs = GlobalTime::get().absoluteTimeMs();
    uint64_t intervalMs = static_cast<uint64_t>(std::max(descUpdateIntervalMs(), 0));
    if (intervalMs > 0 && m_activeDesc.has_value()) {
      const uint64_t lifetimeFloorMs = static_cast<uint64_t>(m_activeDesc->maxTimeToLive * (1000.0f / 6.0f));
      intervalMs = std::max(intervalMs, lifetimeFloorMs);
    }

    const bool firstUse = !m_activeDesc.has_value();
    const bool dwellElapsed = nowMs >= m_activeDescTimeMs + intervalMs;

    if (firstUse || dwellElapsed) {
      // Live material edits (roughness / metallic): re-register under the same
      // handle. The registration is otherwise once-per-process, so without this
      // the two options silently did nothing after the first frame. The old
      // system keeps its own MaterialData COPY (ParticleSystem::materialData)
      // and crossfades out; the forked identity picks the new material up on
      // this frame's fetchParticleSystem.
      if (m_materialRegistered &&
          (saturate(roughness()) != m_materialRoughness || saturate(metallic()) != m_materialMetallic)) {
        ctx.getSceneManager().getAssetReplacer()->destroyExternalMaterial(
          reinterpret_cast<remixapi_MaterialHandle>(kMaterialHandleValue));
        m_materialRegistered = false;
        ensureMaterial(ctx);
      }

      RtxParticleSystemDesc next = buildDesc(params);
      // Only pay for a new system if something actually moved. A settled
      // weather state produces a bit-identical descriptor every time, so this
      // is the common case once a blend finishes.
      if (firstUse || next.calcHash() != m_activeDesc->calcHash()) {
        m_activeDesc = std::move(next);
        m_activeDescTimeMs = nowMs;
      } else {
        // Re-arm the timer anyway so a settled state isn't re-hashing every
        // frame once the interval has passed.
        m_activeDescTimeMs = nowMs;
      }
    }

    return *m_activeDesc;
  }

  // ---------------------------------------------------------------------------
  // buildEmitterTransform — place and orient the spawn plane.
  //
  // Local +Z maps to the fall direction (which is what makes
  // initialVelocityFromNormal launch drops downwind), local XY spans the plane
  // scaled to spawnRadiusMeters. The plane sits spawnDistanceWorld back along
  // the travel direction from the camera, so drops converge on the viewer
  // instead of falling behind them in a crosswind.
  //
  // DELIBERATELY INDEPENDENT OF CAMERA ORIENTATION (fork — 2026-07-25). This
  // used to add a bias along the camera's flattened forward vector to push the
  // volume into view. That tied the spawn plane to where the player was
  // LOOKING: panning swung the whole volume around them, and because the spawn
  // shader smears new particles between the emitter's previous and current
  // transform, a turn dragged freshly spawned drops along the swing. Only the
  // camera's POSITION may move the emitter; nothing here may read its rotation.
  // A 20 m radius disc centred on the viewer already covers the view in every
  // direction, so the bias was buying very little for that cost.
  // ---------------------------------------------------------------------------
  Matrix4 PrecipitationSystem::buildEmitterTransform(RtxContext& ctx, const Params& params) {
    const RtCamera& camera = ctx.getSceneManager().getCamera();
    const Vector3 cameraPos = camera.getPosition();

    const Vector3 origin = cameraPos - params.fallDirection * params.spawnDistanceWorld;

    Vector3 tangent, bitangent;
    buildTangentBasis(params.fallDirection, tangent, bitangent);

    const float radiusWorld = std::max(spawnRadiusMeters(), 0.1f) * RtxOptions::getMeterToWorldUnitScale();

    return Matrix4 {
      Vector4(tangent.x   * radiusWorld, tangent.y   * radiusWorld, tangent.z   * radiusWorld, 0.0f),
      Vector4(bitangent.x * radiusWorld, bitangent.y * radiusWorld, bitangent.z * radiusWorld, 0.0f),
      Vector4(params.fallDirection.x, params.fallDirection.y, params.fallDirection.z, 0.0f),
      Vector4(origin.x, origin.y, origin.z, 1.0f)
    };
  }

  // ---------------------------------------------------------------------------
  // submit — the per-frame entry point.
  // ---------------------------------------------------------------------------
  void PrecipitationSystem::submit(RtxContext& ctx) {
    const bool active = enable() && intensity() > 0.0f && maxParticles() > 0;

    if (!active) {
      if (m_wasActive) {
        // Stop submitting: existing particles finish their lifetime and the
        // manager evicts the system. Forget the descriptor so re-enabling
        // starts a fresh one immediately rather than waiting out the dwell.
        m_activeDesc.reset();
        m_wasActive = false;
      }
      return;
    }

    // Camera-anchored, so there is nothing sensible to do before the main
    // camera has been established for this frame.
    if (!ctx.getSceneManager().getCamera().isValid(ctx.getDevice()->getCurrentFrameId())) {
      return;
    }

    // The singleton outlives any DxvkDevice. If the device changed since the
    // resources were built (device recreation, a second D3D9 device in the
    // process), every cached Rc<> belongs to the OLD device and — worse — the
    // registered-flags say "done" while the NEW device's asset replacer has
    // never seen the mesh/material handles, which would make submitExternalDraw
    // log "External mesh has no submeshes" every frame forever. Drop everything
    // and rebuild against the current device.
    if (m_resourceDevice != ctx.getDevice().ptr()) {
      m_dropImage = nullptr;
      m_dropImageView = nullptr;
      m_quadVertexBuffer = nullptr;
      m_quadIndexBuffer = nullptr;
      m_materialRegistered = false;
      m_meshRegistered = false;
      m_activeDesc.reset();
      m_resourceDevice = ctx.getDevice().ptr();
    }

    if (!ensureResources(ctx)) {
      // ensureResources logs its own failure; don't spam per frame.
      return;
    }

    const Params params = resolveParams();
    const RtxParticleSystemDesc& desc = refreshDesc(ctx, params);

    DrawCallState drawCall {};
    drawCall.cameraType = CameraType::Main;
    // transformData / materialData are private to DrawCallState; the friended
    // hook fills them in (same arrangement the Remix API uses for its own
    // hand-built external draws).
    fork_hooks::precipitationEmitterDrawCall(drawCall, buildEmitterTransform(ctx, params));

    // No categories on the emitter itself: it is hidden, and the particle
    // manager stamps InstanceCategories::Particle onto the geometry it
    // generates. (This field only feeds the external-draw identity hash anyway;
    // the categories that reach the instance come from the draw call.)
    const CategoryFlags categories {};

    // commitExternalGeometryToRT takes ownership via unique_ptr (upstream
    // changed the signature; picked up in the 2026-07-29 sync).
    auto state = std::make_unique<ExternalDrawState>(ExternalDrawState {
      std::move(drawCall),
      reinterpret_cast<remixapi_MeshHandle>(kMeshHandleValue),
      CameraType::Main,
      categories,
      /* doubleSided */ true,
      desc,
      {}
    });

    ctx.commitExternalGeometryToRT(std::move(state));
    m_wasActive = true;
  }

  // ---------------------------------------------------------------------------
  // showImguiSettings — the global knobs. Per-preset look values are generated
  // into the Weather Preset Editor automatically by the weather field table, so
  // this panel deliberately covers only what is NOT per-preset.
  // ---------------------------------------------------------------------------
  void PrecipitationSystem::showImguiSettings() {
    if (!ImGui::TreeNode("Precipitation (global)")) {
      return;
    }

    ImGui::TextDisabled("Rain / snow particles. Look and amount are per-preset\n"
                        "(Weather Preset Editor -> Precipitation); these are the\n"
                        "budget and spawn-volume knobs shared by every preset.");

    // Built once: the panel no longer sits under Weather, so the preset
    // coupling is not implied by position any more and has to be said.
    static const std::string s_enableTooltip =
        std::string(enableObject().getDescription()) +
        "\n\nMaster switch only. How hard it rains is driven by the active "
        "weather preset's precipitation intensity, so with this enabled and a "
        "clear preset active you will still see nothing.";

    RemixGui::Checkbox("Enable Precipitation", &enableObject());
    RemixGui::SetTooltipToLastWidgetOnHover(s_enableTooltip.c_str());

    // Per-preset look values (intensity, fall speed, drop size, color, sky
    // light, ...) are edited in the Weather Preset Editor like every other
    // Numos weather field - the "Live values" tree that used to mirror them
    // here was retired (fork - 2026-07-26) to keep this panel consistent with
    // how the other features present their controls: preset fields live in
    // the preset editor, only the shared non-weather knobs live here.
    ImGui::TextDisabled("Per-preset look values (amount, motion, drop look, sky\n"
                        "light) are edited in the Weather Preset Editor above.");

    // Appearance overrides (fork - 2026-07-26). Unlike the tree above, these
    // are NOT per-preset fields, so the weather blender never touches them:
    // they are the look knobs that actually stick while a preset is active.
    if (ImGui::TreeNode("Appearance overrides (stack on presets)")) {
      ImGui::TextDisabled("Tint and multipliers applied ON TOP of the per-preset look,\n"
                          "so they hold even while an active weather preset drives the\n"
                          "look fields every frame. Saved like any other option. Edits\n"
                          "apply on the descriptor dwell (~1s) and crossfade in over\n"
                          "one particle lifetime.");

      RemixGui::ColorEdit3("Tint", &tintColorObject());
      RemixGui::SetTooltipToLastWidgetOnHover(tintColorObject().getDescription());

      RemixGui::DragFloat("Opacity x", &opacityScaleObject(), 0.01f, 0.0f, 4.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(opacityScaleObject().getDescription());

      RemixGui::DragFloat("Drop Width x", &widthScaleObject(), 0.01f, 0.0f, 10.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(widthScaleObject().getDescription());

      RemixGui::DragFloat("Drop Length x", &lengthScaleObject(), 0.01f, 0.0f, 10.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(lengthScaleObject().getDescription());

      RemixGui::DragFloat("Motion Streak x", &streakScaleObject(), 0.01f, 0.0f, 10.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(streakScaleObject().getDescription());

      RemixGui::DragFloat("Size Variance", &sizeVarianceObject(), 0.01f, 0.0f, 0.9f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(sizeVarianceObject().getDescription());

      RemixGui::DragFloat("Opacity Variance", &opacityVarianceObject(), 0.01f, 0.0f, 0.9f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(opacityVarianceObject().getDescription());

      ImGui::TreePop();
    }

    RemixGui::DragInt("Particle Budget", &maxParticlesObject(), 100.0f, 0, 200000);
    RemixGui::SetTooltipToLastWidgetOnHover(maxParticlesObject().getDescription());

    RemixGui::DragFloat("Spawn Radius (m)", &spawnRadiusMetersObject(), 0.25f, 1.0f, 200.0f, "%.1f");
    RemixGui::SetTooltipToLastWidgetOnHover(spawnRadiusMetersObject().getDescription());

    RemixGui::DragFloat("Spawn Height (m)", &spawnHeightMetersObject(), 0.25f, 1.0f, 200.0f, "%.1f");
    RemixGui::SetTooltipToLastWidgetOnHover(spawnHeightMetersObject().getDescription());

    RemixGui::DragFloat("Fall Margin (m)", &fallMarginMetersObject(), 0.25f, 0.0f, 100.0f, "%.1f");
    RemixGui::SetTooltipToLastWidgetOnHover(fallMarginMetersObject().getDescription());

    RemixGui::DragFloat("Emission Spread (deg)", &spreadDegreesObject(), 0.1f, 0.0f, 45.0f, "%.1f");
    RemixGui::SetTooltipToLastWidgetOnHover(spreadDegreesObject().getDescription());

    RemixGui::DragFloat("Turbulence Frequency", &turbulenceFrequencyObject(), 0.001f, 0.0001f, 1.0f, "%.4f");
    RemixGui::SetTooltipToLastWidgetOnHover(turbulenceFrequencyObject().getDescription());

    ImGui::Separator();

    RemixGui::Checkbox("Stop Under Cover (ray traced)", &occludeUnderCoverObject());
    RemixGui::SetTooltipToLastWidgetOnHover(occludeUnderCoverObject().getDescription());

    // Read-only occlusion diagnostics (fork - 2026-07-25). Added after the
    // feature failed silently on its first in-game test: every CPU/GPU link
    // was provably correct from source, and what was actually wrong (the spawn
    // plane sitting BELOW the sheltering geometry because of the scene-scale /
    // world-unit mismatch) was invisible without numbers. These four counters
    // turn "occlusion doesn't work" into a measurement:
    //   rays == 0                -> the trace is not running at all (option off,
    //                               TLAS missing, or no spawns this frame);
    //   rays > 0, kills+hits == 0 while standing under a roof
    //                            -> the trace runs but the roof is not
    //                               reachable in the TLAS under
    //                               OBJECT_MASK_OPAQUE (not submitted, culled,
    //                               or non-opaque geometry);
    //   relocated > 0 under partial cover (doorway, porch), with rain still
    //   visible in the open      -> the shelter-relocation fix is doing its
    //                               job (drops reroll to open sky instead of
    //                               dying, so cover no longer starves the
    //                               whole spawn volume);
    //   shelter kills > 0        -> drops for which EVERY reroll found cover:
    //                               the signature of a genuinely enclosed
    //                               space. Under partial cover this should
    //                               stay near zero now - if it climbs there,
    //                               the relocation attempts are all landing
    //                               under the same roof (volume too small
    //                               relative to the cover).
    if (occludeUnderCover()) {
      const auto diag = RtxParticleSystemManager::getSpawnTraceDiagnostics();
      const float planeUnits = std::max(spawnHeightMeters(), 0.5f) * RtxOptions::getMeterToWorldUnitScale();

      ImGui::TextDisabled("Occlusion diagnostics (readback is a few frames old):");
      if (!diag.tlasValid) {
        ImGui::Text("  Scene TLAS: MISSING - trace skipped");
      } else {
        ImGui::Text("  Scene TLAS: available");
      }
      ImGui::Text("  Spawn rays: %u   shelter kills: %u   landing hits: %u   relocated: %u",
                  diag.raysTraced, diag.shelterKills, diag.landingHits, diag.relocations);
      ImGui::Text("  Spawn plane: +%.0f world units above camera (%.1f m at rtx.sceneScale = %.3g)",
                  planeUnits, spawnHeightMeters(), RtxOptions::sceneScale());
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Sanity-check this against the game's real unit scale. Remix converts authored meters via\n"
        "meterToWorldUnitScale = 100 * rtx.sceneScale; if the game's world uses more units per real meter\n"
        "than that (Fallout 4: ~70 units/m vs 10 units/m at sceneScale 0.1), every meter-authored\n"
        "precipitation dimension is proportionally smaller in the real world - including this spawn-plane\n"
        "height. Cover ABOVE the spawn plane is handled by the upward shelter probe, so occlusion still\n"
        "works; but the visible rain volume itself scales with these numbers.");
    }

    RemixGui::Checkbox("Collide With World (screen-space)", &enableCollisionObject());
    RemixGui::SetTooltipToLastWidgetOnHover(enableCollisionObject().getDescription());

    RemixGui::DragFloat("Collision Thickness (cm)", &collisionThicknessObject(), 0.5f, 0.0f, 500.0f, "%.1f");
    RemixGui::SetTooltipToLastWidgetOnHover(collisionThicknessObject().getDescription());

    ImGui::Separator();

    RemixGui::DragFloat("Drop Roughness", &roughnessObject(), 0.01f, 0.0f, 1.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover(roughnessObject().getDescription());

    RemixGui::DragFloat("Drop Metallic", &metallicObject(), 0.01f, 0.0f, 1.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover(metallicObject().getDescription());

    RemixGui::DragInt("Descriptor Update Interval (ms)", &descUpdateIntervalMsObject(), 10.0f, 0, 10000);
    RemixGui::SetTooltipToLastWidgetOnHover(descUpdateIntervalMsObject().getDescription());

    RemixGui::Checkbox("Debug: Show Emitter Plane", &debugShowEmitterObject());
    RemixGui::SetTooltipToLastWidgetOnHover(debugShowEmitterObject().getDescription());

    ImGui::TreePop();
  }

} }  // namespace dxvk::fork_precipitation


// ---------------------------------------------------------------------------
// fork_hooks bodies
// ---------------------------------------------------------------------------
namespace dxvk { namespace fork_hooks {

  // Called once per frame from RtxContext::injectRTX, immediately before
  // SceneManager::prepareSceneData (which runs the particle simulation). No-op
  // unless a weather preset has actually asked for precipitation.
  void submitPrecipitation(class RtxContext& ctx) {
    fork_precipitation::PrecipitationSystem::get().submit(ctx);
  }

  // Fills in the precipitation emitter's draw call. Friended by DrawCallState
  // for transformData / materialData, mirroring how
  // RemixAPIPrivateAccessor::toRtDrawState builds the API's external draws.
  //
  // The blend state here is what makes the DROPS translucent, not the emitter:
  // RtxParticleSystemManager::submitDrawState copies this LegacyMaterialData
  // onto the generated particle geometry's draw call. It is the runtime-side
  // equivalent of the InstanceInfoBlendEXT the NFSC mod chains onto its spawner
  // instance. Colour/alpha are modulated against VertexColor0 so the
  // per-particle colour (which carries the weather-driven tint and opacity)
  // reaches the surface.
  void precipitationEmitterDrawCall(DrawCallState& drawCall, const Matrix4& objectToWorld) {
    drawCall.transformData.objectToWorld = objectToWorld;
    drawCall.transformData.textureTransform = Matrix4 {};
    drawCall.transformData.texgenMode = TexGenMode::None;

    drawCall.materialData.alphaTestEnabled = false;
    drawCall.materialData.alphaTestReferenceValue = 0;
    drawCall.materialData.alphaTestCompareOp = VK_COMPARE_OP_ALWAYS;
    drawCall.materialData.blendMode.enableBlending = true;
    drawCall.materialData.blendMode.colorSrcFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    drawCall.materialData.blendMode.colorDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    drawCall.materialData.blendMode.colorBlendOp = VK_BLEND_OP_ADD;
    drawCall.materialData.blendMode.alphaSrcFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    drawCall.materialData.blendMode.alphaDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    drawCall.materialData.blendMode.alphaBlendOp = VK_BLEND_OP_ADD;
    drawCall.materialData.blendMode.writeMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    drawCall.materialData.textureColorArg1Source = RtTextureArgSource::Texture;
    drawCall.materialData.textureColorArg2Source = RtTextureArgSource::VertexColor0;
    drawCall.materialData.textureColorOperation = DxvkRtTextureOperation::Modulate;
    drawCall.materialData.textureAlphaArg1Source = RtTextureArgSource::Texture;
    drawCall.materialData.textureAlphaArg2Source = RtTextureArgSource::VertexColor0;
    drawCall.materialData.textureAlphaOperation = DxvkRtTextureOperation::Modulate;
    drawCall.materialData.isVertexColorBakedLighting = false;
  }

  // Renders the global precipitation controls inside the weather panel.
  void showPrecipitationUI() {
    fork_precipitation::PrecipitationSystem::showImguiSettings();
  }

} }  // namespace dxvk::fork_hooks
