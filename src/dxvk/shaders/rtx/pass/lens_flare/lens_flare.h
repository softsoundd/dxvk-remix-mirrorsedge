/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include "rtx/utility/shader_types.h"

// The fast, paraxial ghosts.
#define LENS_FLARE_GHOSTS_INPUT         0
#define LENS_FLARE_SUN_VISIBILITY_INPUT 1
#define LENS_FLARE_COLOR_INPUT_OUTPUT   10

// The ray traced ghosts, whose passes search for each grid's domain, trace it, give it its irradiance, then rasterise
// the grids.
#define LENS_FLARE_TRACE_SURFACES_INPUT         0
#define LENS_FLARE_TRACE_INSTANCES_INPUT        1
#define LENS_FLARE_TRACE_DOMAINS_INPUT_OUTPUT   10
#define LENS_FLARE_TRACE_VERTICES_INPUT_OUTPUT  11

#define LENS_FLARE_RASTER_VERTICES_INPUT       0
#define LENS_FLARE_RASTER_INSTANCES_INPUT      1
#define LENS_FLARE_RASTER_SUN_VISIBILITY_INPUT 2

// The traced ghosts drawn at half and quarter resolution, added to the output.
#define LENS_FLARE_COMPOSITE_HALF_INPUT         0
#define LENS_FLARE_COMPOSITE_QUARTER_INPUT      1
#define LENS_FLARE_COMPOSITE_COLOR_INPUT_OUTPUT 2

#define LENS_FLARE_MAX_GHOSTS 64
#define LENS_FLARE_TILE_SIZE  16
// The most clear apertures that clip a fast ghost.
#define LENS_FLARE_PASS_DISCS 4
#define LENS_FLARE_WAVELENGTHS 8
#define LENS_FLARE_MAX_SURFACES 32
#define LENS_FLARE_TRACE_GROUP_SIZE 64
// The search traces LENS_FLARE_SEARCH_SAMPLES rays along the sun's meridian on the entrance plane, one per thread of
// its group, then LENS_FLARE_SEARCH_LINES lines of LENS_FLARE_SEARCH_LINE_SAMPLES rays across what they find.
#define LENS_FLARE_SEARCH_SAMPLES 512
#define LENS_FLARE_SEARCH_LINES 16
#define LENS_FLARE_SEARCH_LINE_SAMPLES 32

#define LENS_FLARE_COATING_NONE         0
#define LENS_FLARE_COATING_SINGLE_LAYER 1
#define LENS_FLARE_COATING_MULTILAYER   2

// An instance that draws all of its light over its own grid, one wavelength of a ghost split at its edge band, which
// draws its own light only along the ghost's edges, or that ghost's middle wavelength, its carrier, which also draws
// every wavelength's light past the band.
#define LENS_FLARE_INSTANCE_WHOLE   0
#define LENS_FLARE_INSTANCE_BANDED  1
#define LENS_FLARE_INSTANCE_CARRIER 2

// A traced vertex's ray reaches the sensor, or the lens loses it. A lost ray beside traced ones may take the place
// their trend gives it, so that the triangles across the lost rays' boundary are drawn, and a lost ray beside those a
// place beyond, so that the boundary's light fades out before the drawn triangles end.
#define LENS_FLARE_VERTEX_LOST           0
#define LENS_FLARE_VERTEX_TRACED         1
#define LENS_FLARE_VERTEX_EXTENDED       2
#define LENS_FLARE_VERTEX_EXTENDED_OUTER 3
// The loss margin of a ray lost in a way that gives no margin, such as landing out of range.
#define LENS_FLARE_NO_MARGIN -16.0f

// The wavelengths of a split ghost are consecutive instances, its carrier this far into them.
#define LENS_FLARE_CARRIER_OFFSET (LENS_FLARE_WAVELENGTHS / 2)
// The least edge band and hand-off, in the grid's cells, so that every wavelength's triangles resolve them alike.
#define LENS_FLARE_BAND_CELLS     2.0f
#define LENS_FLARE_HAND_OFF_CELLS 4.0f

#define LENS_FLARE_DEBUG_VIEW_NONE         0
#define LENS_FLARE_DEBUG_VIEW_FLARE_ONLY   1
#define LENS_FLARE_DEBUG_VIEW_GHOST_BOUNDS 2

// A ghost at one wavelength, in the frame of the sun's meridian, x along it and y across it. A pixel's sensor point p
// maps back to the entrance plane at e = (p - (sensorOffset, 0)) * entranceFromSensor, as fitted to the ghost's real
// rays, and on to the stop paraxially at e * stopFromEntrance + (stopOffset, 0), all in mm.
struct LensFlareGhostSample {
  vec2 entranceFromSensor;
  float sensorOffset;
  // The mean of entranceFromSensor's scales, which carries the edges' blend onto the entrance plane.
  float entrancePerSensor;

  // The ghost's radiance at this wavelength per unit of sun illuminance, weighted into the channels.
  vec3 radiance;
  float stopFromEntrance;

  float stopOffset;
  uint pad0;
  uint pad1;
  uint pad2;
};

// One ghost, the sun's light reflected twice inside the lens, sampled at sampleCount wavelengths.
struct LensFlareGhost {
  // Circle in aspect corrected NDC that contains the ghost this frame, for culling.
  vec2 boundCenter;
  float boundRadius;
  // Width the ghost's edges blend over, in mm on the sensor: how far they move from one wavelength to the next,
  // which joins the samples into a continuous fringe, or across the sun's disc, every point of which casts the ghost.
  float edgeSpread;

  uint sampleCount;
  uint passDiscCount;
  uint pad1;
  uint pad2;

  // The clear apertures on the ghost's path that cut it for the sun's angle, as their paraxial images on the entrance
  // plane: each disc's centre along the sun's meridian and its radius, in mm.
  vec2 passDiscs[LENS_FLARE_PASS_DISCS];

  LensFlareGhostSample samples[LENS_FLARE_WAVELENGTHS];
};

struct LensFlareArgs {
  uvec2 imageSize;
  vec2 invImageSize;

  // Angle of the sun's rays entering the lens, and the sensor's mm per unit of aspect corrected NDC. The sensor's
  // image is inverted on the screen, which is why both are negated against the screen.
  vec2 sunRayAngle;
  float sensorScale;
  float aspectRatio;

  // The sun's illuminance at the camera, times the intensity.
  vec3 sourceIlluminance;
  uint ghostCount;

  // The iris's corner radius at the stop, in mm.
  float stopRadius;
  uint apertureBlades;
  float apertureCurvature;
  float apertureRotation;

  uint debugView;
  // The iris blades' spreads off their regular places (see lens_aperture.slangh).
  float apertureDistanceJitter;
  float apertureAngleJitter;
  uint pad0;
};

// A surface of the lens for the ray traced ghosts, its vertex z, radius (0 for flat) and clear radius in mm.
struct LensFlareSurface {
  float z;
  float radius;
  float clearRadius;
  // The index at the d line of the glass of a coated air to glass surface, which the multilayer's inner layer
  // matches, or 0 for a bare cemented surface.
  float coatedGlassIndex;

  // The index after the surface at each wavelength sample, then at the wavelength the lens is focused at.
  vec4 indexAfter[3];
};

// One ghost at one wavelength. Its rays are traced in the frame of the sun's meridian, the plane through the axis
// and the sun, in which the lens's symmetry keeps what passes symmetric about the meridian. A search along the
// meridian and across it finds the domain the passing rays span across the sun's disc, which a fine grid then traces.
struct LensFlareTraceInstance {
  uint first;
  uint second;
  uint wavelength;
  // What the ghost's passes through the surfaces between its reflections let through, at normal incidence.
  float transmission;

  // The segment of the meridian on the entrance plane the search spans: its middle and half length, in mm.
  float searchCenter;
  float searchHalfSize;
  // One of LENS_FLARE_INSTANCE_*.
  uint kind;
  // The ghost's first instance. All the wavelengths of a split ghost trace one shared grid.
  uint ghostFirst;

  // The channel weights of the instance's wavelength, or for a ghost traced at one wavelength, its whole spectrum
  // relative to that wavelength's reflectance.
  vec3 weight;
  // For a split ghost, how far inside the carrier's edges each wavelength draws all of its own light, and over how
  // much further it hands its light to the carrier, in pixels, each at least LENS_FLARE_*_CELLS of the grid's cells.
  float bandInner;

  float bandWidth;
  // How far the ghost's edges blur across the sun's disc, in mm on the sensor.
  float edgeBlur;
  uint pad1;
  uint pad2;
};

// The fine grid's corner and spacing on the entrance plane in the meridian's frame, in mm. No ray passes when the
// spacing is 0.
struct LensFlareTraceDomain {
  vec2 origin;
  vec2 spacing;
};

struct LensFlareTracedVertex {
  // Where the ray meets the sensor, in mm.
  vec2 sensor;
  // Where it crossed the stop, in units of the iris's corner radius.
  vec2 stop;

  // The largest of its heights over the clear radius of each surface it met. Past 1 a surface blocked it.
  float clipRatio;
  // The product of its two reflections' reflectance at their true incidence.
  float reflectance;
  // One of LENS_FLARE_VERTEX_*.
  uint valid;
  // The irradiance it brings the sensor per unit of sun illuminance: the light of the grid's triangles around it
  // over their area on the sensor.
  float irradiance;

  // How the sensor point and the stop coordinate move from the sun's centre to its limb, its edge scaled by the
  // edges' softness, along the meridian (xy) and across it (zw), for the ray's fixed entrance point.
  vec4 sensorFromSun;
  vec4 stopFromSun;
  // The same for the clip ratio.
  vec2 clipFromSun;
  // How much the stop coordinate and the clip ratio change from the ray to its neighbours on the grid.
  float stopPerCell;
  float clipPerCell;

  // For a carrier, how far its ray lies through the hand-off, 0 at the band's inner end and 1 where the carrier takes
  // over, and how that grows per mm on the sensor, so that each wavelength can find it at its own ray.
  float handOff;
  vec2 handOffGradient;
  // How far the ray is from being lost, the least of its margins against each way the lens can lose it, each of which
  // falls through 0 where the ray would be lost. Negative once lost, and LENS_FLARE_NO_MARGIN for a loss with no such
  // margin. The lost rays' boundary is an edge of its own, beside the clip ratio's.
  float lossMargin;

  // How the loss margin changes from the sun's centre to its limb, along the meridian and across it, and the most it
  // changes to the ray's neighbours on the grid.
  vec2 lossFromSun;
  float lossPerCell;
  uint pad0;
};

struct LensFlareTraceArgs {
  // The direction the sun's rays tilt in on the entrance plane, the meridian's, and their angle to the axis.
  vec2 meridian;
  float sunAngle;
  uint surfaceCount;

  uint stopIndex;
  float sensorZ;
  uint coating;
  float coatingWavelengthNm;

  float sensorReflectance;
  float irisCornerRadius;
  // Vertices per side of each fine grid.
  uint gridSize;
  uint instanceCount;

  // An output pixel's size on the sensor in mm.
  float pixelMm;
  // The sun's angular radius in the lens's angles, and the scale on the blur it gives the ghosts' edges.
  float sunRadius;
  float edgeSoftness;
  uint apertureBlades;

  // The wavelength samples, then the focus wavelength, in nm.
  vec4 wavelengthsNm[3];

  float apertureCurvature;
  float apertureRotation;
  // The iris blades' spreads off their regular places (see lens_aperture.slangh).
  float apertureDistanceJitter;
  float apertureAngleJitter;
};

struct LensFlareRasterArgs {
  // The sensor's mm per unit of aspect corrected NDC.
  float sensorScale;
  float aspectRatio;
  uint gridSize;
  // A pixel of the target's size on the sensor in mm.
  float pixelMm;

  // The sun's illuminance at the camera, times the intensity and the radiance the main path gives a unit of sensor
  // irradiance: f^2 over the entrance pupil's area.
  vec3 sourceRadiance;
  uint apertureBlades;

  float apertureCurvature;
  float apertureRotation;
  // The target's size in pixels. It spans the output's extent, at full, half or quarter resolution.
  vec2 imageSize;

  // The first of the instances the draw rasterises, all of which are drawn at the target's resolution.
  uint firstInstance;
  // The iris blades' spreads off their regular places (see lens_aperture.slangh), and the distance of its farthest
  // corner from the centre in units of its regular corner radius.
  float apertureDistanceJitter;
  float apertureAngleJitter;
  float apertureCornerReach;
};

struct LensFlareCompositeArgs {
  uvec2 imageSize;
  vec2 invImageSize;

  // Whether any ghost was drawn at half and at quarter resolution.
  uint drawHalf;
  uint drawQuarter;
  uint pad0;
  uint pad1;
};
