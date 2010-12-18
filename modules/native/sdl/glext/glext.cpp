/*
   FALCON - The Falcon Programming Language.
   FILE: sdlopengl.cpp

   The SDL binding support module - OpenGL extension.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Tue, 12 Aug 2009 00:06:56 +1100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The sdl module - main file.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include "version.h"
#include "glext_ext.h"
#include "glext_mod.h"

#include <SDL.h>
#include <SDL_opengl.h>

/#--*
   @module sdlopengl sdlopengl
   @brief OpenGL extensions for the Falcon SDL module.

   This module wraps the OpenGL extensions for SDL. Namely, this module
   is meant to allow the falcon user access to the openGL API so that
   3D rendering can be performed.
   

   @beginmodule sdlopengl
*/


FALCON_MODULE_DECL
{
   Falcon::Module *self = new Falcon::Module();
   self->name( "sdlgl" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the main SDL module.
   self->addDepend( "sdl" );

   
   {
      Falcon::Symbol *c_sdlopenglext = self->addClass( "glext" );

   }
   {
      Falcon::Symbol *c_sdlglext = self->addClass( "GLExt" );
      self->addClassProperty( c_sdlglext, "GLEXT_VERSION" ).setInteger( GL_GLEXT_VERSION );

      #ifndef GL_VERSION_1_2
      self->addClassProperty( c_sdlglext, "UNSIGNED_BYTE_3_3_2" ).setInteger( GL_UNSIGNED_BYTE_3_3_2 );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_4_4_4_4" ).setInteger( GL_UNSIGNED_SHORT_4_4_4_4 );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_5_5_5_1" ).setInteger( GL_UNSIGNED_SHORT_5_5_5_1 );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_8_8_8_8" ).setInteger( GL_UNSIGNED_INT_8_8_8_8 );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_10_10_10_2" ).setInteger( GL_UNSIGNED_INT_10_10_10_2 );
      self->addClassProperty( c_sdlglext, "RESCALE_NORMAL" ).setInteger( GL_RESCALE_NORMAL );
      self->addClassProperty( c_sdlglext, "TEXTURE_BINDING_3D" ).setInteger( GL_TEXTURE_BINDING_3D );
      self->addClassProperty( c_sdlglext, "PACK_SKIP_IMAGES" ).setInteger( GL_PACK_SKIP_IMAGES );
      self->addClassProperty( c_sdlglext, "PACK_IMAGE_HEIGHT" ).setInteger( GL_PACK_IMAGE_HEIGHT );
      self->addClassProperty( c_sdlglext, "UNPACK_SKIP_IMAGES" ).setInteger( GL_UNPACK_SKIP_IMAGES );
      self->addClassProperty( c_sdlglext, "UNPACK_IMAGE_HEIGHT" ).setInteger( GL_UNPACK_IMAGE_HEIGHT );
      self->addClassProperty( c_sdlglext, "TEXTURE_3D" ).setInteger( GL_TEXTURE_3D );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_3D" ).setInteger( GL_PROXY_TEXTURE_3D );
      self->addClassProperty( c_sdlglext, "TEXTURE_DEPTH" ).setInteger( GL_TEXTURE_DEPTH );
      self->addClassProperty( c_sdlglext, "TEXTURE_WRAP_R" ).setInteger( GL_TEXTURE_WRAP_R );
      self->addClassProperty( c_sdlglext, "MAX_3D_TEXTURE_SIZE" ).setInteger( GL_MAX_3D_TEXTURE_SIZE );
      self->addClassProperty( c_sdlglext, "UNSIGNED_BYTE_2_3_3_REV" ).setInteger( GL_UNSIGNED_BYTE_2_3_3_REV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_5_6_5" ).setInteger( GL_UNSIGNED_SHORT_5_6_5 );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_5_6_5_REV" ).setInteger( GL_UNSIGNED_SHORT_5_6_5_REV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_4_4_4_4_REV" ).setInteger( GL_UNSIGNED_SHORT_4_4_4_4_REV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_1_5_5_5_REV" ).setInteger( GL_UNSIGNED_SHORT_1_5_5_5_REV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_8_8_8_8_REV" ).setInteger( GL_UNSIGNED_INT_8_8_8_8_REV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_2_10_10_10_REV" ).setInteger( GL_UNSIGNED_INT_2_10_10_10_REV );
      self->addClassProperty( c_sdlglext, "BGR" ).setInteger( GL_BGR );
      self->addClassProperty( c_sdlglext, "BGRA" ).setInteger( GL_BGRA );
      self->addClassProperty( c_sdlglext, "MAX_ELEMENTS_VERTICES" ).setInteger( GL_MAX_ELEMENTS_VERTICES );
      self->addClassProperty( c_sdlglext, "MAX_ELEMENTS_INDICES" ).setInteger( GL_MAX_ELEMENTS_INDICES );
      self->addClassProperty( c_sdlglext, "CLAMP_TO_EDGE" ).setInteger( GL_CLAMP_TO_EDGE );
      self->addClassProperty( c_sdlglext, "TEXTURE_MIN_LOD" ).setInteger( GL_TEXTURE_MIN_LOD );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_LOD" ).setInteger( GL_TEXTURE_MAX_LOD );
      self->addClassProperty( c_sdlglext, "TEXTURE_BASE_LEVEL" ).setInteger( GL_TEXTURE_BASE_LEVEL );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_LEVEL" ).setInteger( GL_TEXTURE_MAX_LEVEL );
      self->addClassProperty( c_sdlglext, "LIGHT_MODEL_COLOR_CONTROL" ).setInteger( GL_LIGHT_MODEL_COLOR_CONTROL );
      self->addClassProperty( c_sdlglext, "SINGLE_COLOR" ).setInteger( GL_SINGLE_COLOR );
      self->addClassProperty( c_sdlglext, "SEPARATE_SPECULAR_COLOR" ).setInteger( GL_SEPARATE_SPECULAR_COLOR );
      self->addClassProperty( c_sdlglext, "SMOOTH_POINT_SIZE_RANGE" ).setInteger( GL_SMOOTH_POINT_SIZE_RANGE );
      self->addClassProperty( c_sdlglext, "SMOOTH_POINT_SIZE_GRANULARITY" ).setInteger( GL_SMOOTH_POINT_SIZE_GRANULARITY );
      self->addClassProperty( c_sdlglext, "SMOOTH_LINE_WIDTH_RANGE" ).setInteger( GL_SMOOTH_LINE_WIDTH_RANGE );
      self->addClassProperty( c_sdlglext, "SMOOTH_LINE_WIDTH_GRANULARITY" ).setInteger( GL_SMOOTH_LINE_WIDTH_GRANULARITY );
      self->addClassProperty( c_sdlglext, "ALIASED_POINT_SIZE_RANGE" ).setInteger( GL_ALIASED_POINT_SIZE_RANGE );
      self->addClassProperty( c_sdlglext, "ALIASED_LINE_WIDTH_RANGE" ).setInteger( GL_ALIASED_LINE_WIDTH_RANGE );
      #endif

      #ifndef GL_ARB_imaging
      self->addClassProperty( c_sdlglext, "CONSTANT_COLOR" ).setInteger( GL_CONSTANT_COLOR );
      self->addClassProperty( c_sdlglext, "ONE_MINUS_CONSTANT_COLOR" ).setInteger( GL_ONE_MINUS_CONSTANT_COLOR );
      self->addClassProperty( c_sdlglext, "CONSTANT_ALPHA" ).setInteger( GL_CONSTANT_ALPHA );
      self->addClassProperty( c_sdlglext, "ONE_MINUS_CONSTANT_ALPHA" ).setInteger( GL_ONE_MINUS_CONSTANT_ALPHA );
      self->addClassProperty( c_sdlglext, "BLEND_COLOR" ).setInteger( GL_BLEND_COLOR );
      self->addClassProperty( c_sdlglext, "FUNC_ADD" ).setInteger( GL_FUNC_ADD );
      self->addClassProperty( c_sdlglext, "MIN" ).setInteger( GL_MIN );
      self->addClassProperty( c_sdlglext, "MAX" ).setInteger( GL_MAX );
      self->addClassProperty( c_sdlglext, "BLEND_EQUATION" ).setInteger( GL_BLEND_EQUATION );
      self->addClassProperty( c_sdlglext, "FUNC_SUBTRACT" ).setInteger( GL_FUNC_SUBTRACT );
      self->addClassProperty( c_sdlglext, "FUNC_REVERSE_SUBTRACT" ).setInteger( GL_FUNC_REVERSE_SUBTRACT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_1D" ).setInteger( GL_CONVOLUTION_1D );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_2D" ).setInteger( GL_CONVOLUTION_2D );
      self->addClassProperty( c_sdlglext, "SEPARABLE_2D" ).setInteger( GL_SEPARABLE_2D );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_BORDER_MODE" ).setInteger( GL_CONVOLUTION_BORDER_MODE );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_FILTER_SCALE" ).setInteger( GL_CONVOLUTION_FILTER_SCALE );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_FILTER_BIAS" ).setInteger( GL_CONVOLUTION_FILTER_BIAS );
      self->addClassProperty( c_sdlglext, "REDUCE" ).setInteger( GL_REDUCE );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_FORMAT" ).setInteger( GL_CONVOLUTION_FORMAT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_WIDTH" ).setInteger( GL_CONVOLUTION_WIDTH );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_HEIGHT" ).setInteger( GL_CONVOLUTION_HEIGHT );
      self->addClassProperty( c_sdlglext, "MAX_CONVOLUTION_WIDTH" ).setInteger( GL_MAX_CONVOLUTION_WIDTH );
      self->addClassProperty( c_sdlglext, "MAX_CONVOLUTION_HEIGHT" ).setInteger( GL_MAX_CONVOLUTION_HEIGHT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_RED_SCALE" ).setInteger( GL_POST_CONVOLUTION_RED_SCALE );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_GREEN_SCALE" ).setInteger( GL_POST_CONVOLUTION_GREEN_SCALE );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_BLUE_SCALE" ).setInteger( GL_POST_CONVOLUTION_BLUE_SCALE );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_ALPHA_SCALE" ).setInteger( GL_POST_CONVOLUTION_ALPHA_SCALE );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_RED_BIAS" ).setInteger( GL_POST_CONVOLUTION_RED_BIAS );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_GREEN_BIAS" ).setInteger( GL_POST_CONVOLUTION_GREEN_BIAS );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_BLUE_BIAS" ).setInteger( GL_POST_CONVOLUTION_BLUE_BIAS );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_ALPHA_BIAS" ).setInteger( GL_POST_CONVOLUTION_ALPHA_BIAS );
      self->addClassProperty( c_sdlglext, "HISTOGRAM" ).setInteger( GL_HISTOGRAM );
      self->addClassProperty( c_sdlglext, "PROXY_HISTOGRAM" ).setInteger( GL_PROXY_HISTOGRAM );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_WIDTH" ).setInteger( GL_HISTOGRAM_WIDTH );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_FORMAT" ).setInteger( GL_HISTOGRAM_FORMAT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_RED_SIZE" ).setInteger( GL_HISTOGRAM_RED_SIZE );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_GREEN_SIZE" ).setInteger( GL_HISTOGRAM_GREEN_SIZE );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_BLUE_SIZE" ).setInteger( GL_HISTOGRAM_BLUE_SIZE );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_ALPHA_SIZE" ).setInteger( GL_HISTOGRAM_ALPHA_SIZE );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_LUMINANCE_SIZE" ).setInteger( GL_HISTOGRAM_LUMINANCE_SIZE );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_SINK" ).setInteger( GL_HISTOGRAM_SINK );
      self->addClassProperty( c_sdlglext, "MINMAX" ).setInteger( GL_MINMAX );
      self->addClassProperty( c_sdlglext, "MINMAX_FORMAT" ).setInteger( GL_MINMAX_FORMAT );
      self->addClassProperty( c_sdlglext, "MINMAX_SINK" ).setInteger( GL_MINMAX_SINK );
      self->addClassProperty( c_sdlglext, "TABLE_TOO_LARGE" ).setInteger( GL_TABLE_TOO_LARGE );
      self->addClassProperty( c_sdlglext, "COLOR_MATRIX" ).setInteger( GL_COLOR_MATRIX );
      self->addClassProperty( c_sdlglext, "COLOR_MATRIX_STACK_DEPTH" ).setInteger( GL_COLOR_MATRIX_STACK_DEPTH );
      self->addClassProperty( c_sdlglext, "MAX_COLOR_MATRIX_STACK_DEPTH" ).setInteger( GL_MAX_COLOR_MATRIX_STACK_DEPTH );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_RED_SCALE" ).setInteger( GL_POST_COLOR_MATRIX_RED_SCALE );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_GREEN_SCALE" ).setInteger( GL_POST_COLOR_MATRIX_GREEN_SCALE );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_BLUE_SCALE" ).setInteger( GL_POST_COLOR_MATRIX_BLUE_SCALE );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_ALPHA_SCALE" ).setInteger( GL_POST_COLOR_MATRIX_ALPHA_SCALE );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_RED_BIAS" ).setInteger( GL_POST_COLOR_MATRIX_RED_BIAS );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_GREEN_BIAS" ).setInteger( GL_POST_COLOR_MATRIX_GREEN_BIAS );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_BLUE_BIAS" ).setInteger( GL_POST_COLOR_MATRIX_BLUE_BIAS );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_ALPHA_BIAS" ).setInteger( GL_POST_COLOR_MATRIX_ALPHA_BIAS );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE" ).setInteger( GL_COLOR_TABLE );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_COLOR_TABLE" ).setInteger( GL_POST_CONVOLUTION_COLOR_TABLE );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_COLOR_TABLE" ).setInteger( GL_POST_COLOR_MATRIX_COLOR_TABLE );
      self->addClassProperty( c_sdlglext, "PROXY_COLOR_TABLE" ).setInteger( GL_PROXY_COLOR_TABLE );
      self->addClassProperty( c_sdlglext, "PROXY_POST_CONVOLUTION_COLOR_TABLE" ).setInteger( GL_PROXY_POST_CONVOLUTION_COLOR_TABLE );
      self->addClassProperty( c_sdlglext, "PROXY_POST_COLOR_MATRIX_COLOR_TABLE" ).setInteger( GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_SCALE" ).setInteger( GL_COLOR_TABLE_SCALE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_BIAS" ).setInteger( GL_COLOR_TABLE_BIAS );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_FORMAT" ).setInteger( GL_COLOR_TABLE_FORMAT );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_WIDTH" ).setInteger( GL_COLOR_TABLE_WIDTH );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_RED_SIZE" ).setInteger( GL_COLOR_TABLE_RED_SIZE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_GREEN_SIZE" ).setInteger( GL_COLOR_TABLE_GREEN_SIZE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_BLUE_SIZE" ).setInteger( GL_COLOR_TABLE_BLUE_SIZE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_ALPHA_SIZE" ).setInteger( GL_COLOR_TABLE_ALPHA_SIZE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_LUMINANCE_SIZE" ).setInteger( GL_COLOR_TABLE_LUMINANCE_SIZE );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_INTENSITY_SIZE" ).setInteger( GL_COLOR_TABLE_INTENSITY_SIZE );
      self->addClassProperty( c_sdlglext, "CONSTANT_BORDER" ).setInteger( GL_CONSTANT_BORDER );
      self->addClassProperty( c_sdlglext, "REPLICATE_BORDER" ).setInteger( GL_REPLICATE_BORDER );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_BORDER_COLOR" ).setInteger( GL_CONVOLUTION_BORDER_COLOR );
      #endif

      #ifndef GL_VERSION_1_3
      self->addClassProperty( c_sdlglext, "TEXTURE0" ).setInteger( GL_TEXTURE0 );
      self->addClassProperty( c_sdlglext, "TEXTURE1" ).setInteger( GL_TEXTURE1 );
      self->addClassProperty( c_sdlglext, "TEXTURE2" ).setInteger( GL_TEXTURE2 );
      self->addClassProperty( c_sdlglext, "TEXTURE3" ).setInteger( GL_TEXTURE3 );
      self->addClassProperty( c_sdlglext, "TEXTURE4" ).setInteger( GL_TEXTURE4 );
      self->addClassProperty( c_sdlglext, "TEXTURE5" ).setInteger( GL_TEXTURE5 );
      self->addClassProperty( c_sdlglext, "TEXTURE6" ).setInteger( GL_TEXTURE6 );
      self->addClassProperty( c_sdlglext, "TEXTURE7" ).setInteger( GL_TEXTURE7 );
      self->addClassProperty( c_sdlglext, "TEXTURE8" ).setInteger( GL_TEXTURE8 );
      self->addClassProperty( c_sdlglext, "TEXTURE9" ).setInteger( GL_TEXTURE9 );
      self->addClassProperty( c_sdlglext, "TEXTURE10" ).setInteger( GL_TEXTURE10 );
      self->addClassProperty( c_sdlglext, "TEXTURE11" ).setInteger( GL_TEXTURE11 );
      self->addClassProperty( c_sdlglext, "TEXTURE12" ).setInteger( GL_TEXTURE12 );
      self->addClassProperty( c_sdlglext, "TEXTURE13" ).setInteger( GL_TEXTURE13 );
      self->addClassProperty( c_sdlglext, "TEXTURE14" ).setInteger( GL_TEXTURE14 );
      self->addClassProperty( c_sdlglext, "TEXTURE15" ).setInteger( GL_TEXTURE15 );
      self->addClassProperty( c_sdlglext, "TEXTURE16" ).setInteger( GL_TEXTURE16 );
      self->addClassProperty( c_sdlglext, "TEXTURE17" ).setInteger( GL_TEXTURE17 );
      self->addClassProperty( c_sdlglext, "TEXTURE18" ).setInteger( GL_TEXTURE18 );
      self->addClassProperty( c_sdlglext, "TEXTURE19" ).setInteger( GL_TEXTURE19 );
      self->addClassProperty( c_sdlglext, "TEXTURE20" ).setInteger( GL_TEXTURE20 );
      self->addClassProperty( c_sdlglext, "TEXTURE21" ).setInteger( GL_TEXTURE21 );
      self->addClassProperty( c_sdlglext, "TEXTURE22" ).setInteger( GL_TEXTURE22 );
      self->addClassProperty( c_sdlglext, "TEXTURE23" ).setInteger( GL_TEXTURE23 );
      self->addClassProperty( c_sdlglext, "TEXTURE24" ).setInteger( GL_TEXTURE24 );
      self->addClassProperty( c_sdlglext, "TEXTURE25" ).setInteger( GL_TEXTURE25 );
      self->addClassProperty( c_sdlglext, "TEXTURE26" ).setInteger( GL_TEXTURE26 );
      self->addClassProperty( c_sdlglext, "TEXTURE27" ).setInteger( GL_TEXTURE27 );
      self->addClassProperty( c_sdlglext, "TEXTURE28" ).setInteger( GL_TEXTURE28 );
      self->addClassProperty( c_sdlglext, "TEXTURE29" ).setInteger( GL_TEXTURE29 );
      self->addClassProperty( c_sdlglext, "TEXTURE30" ).setInteger( GL_TEXTURE30 );
      self->addClassProperty( c_sdlglext, "TEXTURE31" ).setInteger( GL_TEXTURE31 );
      self->addClassProperty( c_sdlglext, "ACTIVE_TEXTURE" ).setInteger( GL_ACTIVE_TEXTURE );
      self->addClassProperty( c_sdlglext, "CLIENT_ACTIVE_TEXTURE" ).setInteger( GL_CLIENT_ACTIVE_TEXTURE );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_UNITS" ).setInteger( GL_MAX_TEXTURE_UNITS );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_MODELVIEW_MATRIX" ).setInteger( GL_TRANSPOSE_MODELVIEW_MATRIX );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_PROJECTION_MATRIX" ).setInteger( GL_TRANSPOSE_PROJECTION_MATRIX );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_TEXTURE_MATRIX" ).setInteger( GL_TRANSPOSE_TEXTURE_MATRIX );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_COLOR_MATRIX" ).setInteger( GL_TRANSPOSE_COLOR_MATRIX );
      self->addClassProperty( c_sdlglext, "MULTISAMPLE" ).setInteger( GL_MULTISAMPLE );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_COVERAGE" ).setInteger( GL_SAMPLE_ALPHA_TO_COVERAGE );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_ONE" ).setInteger( GL_SAMPLE_ALPHA_TO_ONE );
      self->addClassProperty( c_sdlglext, "SAMPLE_COVERAGE" ).setInteger( GL_SAMPLE_COVERAGE );
      self->addClassProperty( c_sdlglext, "SAMPLE_BUFFERS" ).setInteger( GL_SAMPLE_BUFFERS );
      self->addClassProperty( c_sdlglext, "SAMPLES" ).setInteger( GL_SAMPLES );
      self->addClassProperty( c_sdlglext, "SAMPLE_COVERAGE_VALUE" ).setInteger( GL_SAMPLE_COVERAGE_VALUE );
      self->addClassProperty( c_sdlglext, "SAMPLE_COVERAGE_INVERT" ).setInteger( GL_SAMPLE_COVERAGE_INVERT );
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_BIT" ).setInteger( GL_MULTISAMPLE_BIT );
      self->addClassProperty( c_sdlglext, "NORMAL_MAP" ).setInteger( GL_NORMAL_MAP );
      self->addClassProperty( c_sdlglext, "REFLECTION_MAP" ).setInteger( GL_REFLECTION_MAP );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP" ).setInteger( GL_TEXTURE_CUBE_MAP );
      self->addClassProperty( c_sdlglext, "TEXTURE_BINDING_CUBE_MAP" ).setInteger( GL_TEXTURE_BINDING_CUBE_MAP );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_X" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_X );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_X" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_X );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_Y" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_Y );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_Y" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_Z" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_Z );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_Z" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_CUBE_MAP" ).setInteger( GL_PROXY_TEXTURE_CUBE_MAP );
      self->addClassProperty( c_sdlglext, "MAX_CUBE_MAP_TEXTURE_SIZE" ).setInteger( GL_MAX_CUBE_MAP_TEXTURE_SIZE );
      self->addClassProperty( c_sdlglext, "COMPRESSED_ALPHA" ).setInteger( GL_COMPRESSED_ALPHA );
      self->addClassProperty( c_sdlglext, "COMPRESSED_LUMINANCE" ).setInteger( GL_COMPRESSED_LUMINANCE );
      self->addClassProperty( c_sdlglext, "COMPRESSED_LUMINANCE_ALPHA" ).setInteger( GL_COMPRESSED_LUMINANCE_ALPHA );
      self->addClassProperty( c_sdlglext, "COMPRESSED_INTENSITY" ).setInteger( GL_COMPRESSED_INTENSITY );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGB" ).setInteger( GL_COMPRESSED_RGB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGBA" ).setInteger( GL_COMPRESSED_RGBA );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPRESSION_HINT" ).setInteger( GL_TEXTURE_COMPRESSION_HINT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPRESSED_IMAGE_SIZE" ).setInteger( GL_TEXTURE_COMPRESSED_IMAGE_SIZE );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPRESSED" ).setInteger( GL_TEXTURE_COMPRESSED );
      self->addClassProperty( c_sdlglext, "NUM_COMPRESSED_TEXTURE_FORMATS" ).setInteger( GL_NUM_COMPRESSED_TEXTURE_FORMATS );
      self->addClassProperty( c_sdlglext, "COMPRESSED_TEXTURE_FORMATS" ).setInteger( GL_COMPRESSED_TEXTURE_FORMATS );
      self->addClassProperty( c_sdlglext, "CLAMP_TO_BORDER" ).setInteger( GL_CLAMP_TO_BORDER );
      self->addClassProperty( c_sdlglext, "COMBINE" ).setInteger( GL_COMBINE );
      self->addClassProperty( c_sdlglext, "COMBINE_RGB" ).setInteger( GL_COMBINE_RGB );
      self->addClassProperty( c_sdlglext, "COMBINE_ALPHA" ).setInteger( GL_COMBINE_ALPHA );
      self->addClassProperty( c_sdlglext, "SOURCE0_RGB" ).setInteger( GL_SOURCE0_RGB );
      self->addClassProperty( c_sdlglext, "SOURCE1_RGB" ).setInteger( GL_SOURCE1_RGB );
      self->addClassProperty( c_sdlglext, "SOURCE2_RGB" ).setInteger( GL_SOURCE2_RGB );
      self->addClassProperty( c_sdlglext, "SOURCE0_ALPHA" ).setInteger( GL_SOURCE0_ALPHA );
      self->addClassProperty( c_sdlglext, "SOURCE1_ALPHA" ).setInteger( GL_SOURCE1_ALPHA );
      self->addClassProperty( c_sdlglext, "SOURCE2_ALPHA" ).setInteger( GL_SOURCE2_ALPHA );
      self->addClassProperty( c_sdlglext, "OPERAND0_RGB" ).setInteger( GL_OPERAND0_RGB );
      self->addClassProperty( c_sdlglext, "OPERAND1_RGB" ).setInteger( GL_OPERAND1_RGB );
      self->addClassProperty( c_sdlglext, "OPERAND2_RGB" ).setInteger( GL_OPERAND2_RGB );
      self->addClassProperty( c_sdlglext, "OPERAND0_ALPHA" ).setInteger( GL_OPERAND0_ALPHA );
      self->addClassProperty( c_sdlglext, "OPERAND1_ALPHA" ).setInteger( GL_OPERAND1_ALPHA );
      self->addClassProperty( c_sdlglext, "OPERAND2_ALPHA" ).setInteger( GL_OPERAND2_ALPHA );
      self->addClassProperty( c_sdlglext, "RGB_SCALE" ).setInteger( GL_RGB_SCALE );
      self->addClassProperty( c_sdlglext, "ADD_SIGNED" ).setInteger( GL_ADD_SIGNED );
      self->addClassProperty( c_sdlglext, "INTERPOLATE" ).setInteger( GL_INTERPOLATE );
      self->addClassProperty( c_sdlglext, "SUBTRACT" ).setInteger( GL_SUBTRACT );
      self->addClassProperty( c_sdlglext, "CONSTANT" ).setInteger( GL_CONSTANT );
      self->addClassProperty( c_sdlglext, "PRIMARY_COLOR" ).setInteger( GL_PRIMARY_COLOR );
      self->addClassProperty( c_sdlglext, "PREVIOUS" ).setInteger( GL_PREVIOUS );
      self->addClassProperty( c_sdlglext, "DOT3_RGB" ).setInteger( GL_DOT3_RGB );
      self->addClassProperty( c_sdlglext, "DOT3_RGBA" ).setInteger( GL_DOT3_RGBA );
      #endif

      #ifndef GL_VERSION_1_4
      self->addClassProperty( c_sdlglext, "BLEND_DST_RGB" ).setInteger( GL_BLEND_DST_RGB );
      self->addClassProperty( c_sdlglext, "BLEND_SRC_RGB" ).setInteger( GL_BLEND_SRC_RGB );
      self->addClassProperty( c_sdlglext, "BLEND_DST_ALPHA" ).setInteger( GL_BLEND_DST_ALPHA );
      self->addClassProperty( c_sdlglext, "BLEND_SRC_ALPHA" ).setInteger( GL_BLEND_SRC_ALPHA );
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MIN" ).setInteger( GL_POINT_SIZE_MIN );
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MAX" ).setInteger( GL_POINT_SIZE_MAX );
      self->addClassProperty( c_sdlglext, "POINT_FADE_THRESHOLD_SIZE" ).setInteger( GL_POINT_FADE_THRESHOLD_SIZE );
      self->addClassProperty( c_sdlglext, "POINT_DISTANCE_ATTENUATION" ).setInteger( GL_POINT_DISTANCE_ATTENUATION );
      self->addClassProperty( c_sdlglext, "GENERATE_MIPMAP" ).setInteger( GL_GENERATE_MIPMAP );
      self->addClassProperty( c_sdlglext, "GENERATE_MIPMAP_HINT" ).setInteger( GL_GENERATE_MIPMAP_HINT );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT16" ).setInteger( GL_DEPTH_COMPONENT16 );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT24" ).setInteger( GL_DEPTH_COMPONENT24 );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT32" ).setInteger( GL_DEPTH_COMPONENT32 );
      self->addClassProperty( c_sdlglext, "MIRRORED_REPEAT" ).setInteger( GL_MIRRORED_REPEAT );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_SOURCE" ).setInteger( GL_FOG_COORDINATE_SOURCE );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE" ).setInteger( GL_FOG_COORDINATE );
      self->addClassProperty( c_sdlglext, "FRAGMENT_DEPTH" ).setInteger( GL_FRAGMENT_DEPTH );
      self->addClassProperty( c_sdlglext, "CURRENT_FOG_COORDINATE" ).setInteger( GL_CURRENT_FOG_COORDINATE );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_TYPE" ).setInteger( GL_FOG_COORDINATE_ARRAY_TYPE );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_STRIDE" ).setInteger( GL_FOG_COORDINATE_ARRAY_STRIDE );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_POINTER" ).setInteger( GL_FOG_COORDINATE_ARRAY_POINTER );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY" ).setInteger( GL_FOG_COORDINATE_ARRAY );
      self->addClassProperty( c_sdlglext, "COLOR_SUM" ).setInteger( GL_COLOR_SUM );
      self->addClassProperty( c_sdlglext, "CURRENT_SECONDARY_COLOR" ).setInteger( GL_CURRENT_SECONDARY_COLOR );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_SIZE" ).setInteger( GL_SECONDARY_COLOR_ARRAY_SIZE );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_TYPE" ).setInteger( GL_SECONDARY_COLOR_ARRAY_TYPE );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_STRIDE" ).setInteger( GL_SECONDARY_COLOR_ARRAY_STRIDE );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_POINTER" ).setInteger( GL_SECONDARY_COLOR_ARRAY_POINTER );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY" ).setInteger( GL_SECONDARY_COLOR_ARRAY );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_LOD_BIAS" ).setInteger( GL_MAX_TEXTURE_LOD_BIAS );
      self->addClassProperty( c_sdlglext, "TEXTURE_FILTER_CONTROL" ).setInteger( GL_TEXTURE_FILTER_CONTROL );
      self->addClassProperty( c_sdlglext, "TEXTURE_LOD_BIAS" ).setInteger( GL_TEXTURE_LOD_BIAS );
      self->addClassProperty( c_sdlglext, "INCR_WRAP" ).setInteger( GL_INCR_WRAP );
      self->addClassProperty( c_sdlglext, "DECR_WRAP" ).setInteger( GL_DECR_WRAP );
      self->addClassProperty( c_sdlglext, "TEXTURE_DEPTH_SIZE" ).setInteger( GL_TEXTURE_DEPTH_SIZE );
      self->addClassProperty( c_sdlglext, "DEPTH_TEXTURE_MODE" ).setInteger( GL_DEPTH_TEXTURE_MODE );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_MODE" ).setInteger( GL_TEXTURE_COMPARE_MODE );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_FUNC" ).setInteger( GL_TEXTURE_COMPARE_FUNC );
      self->addClassProperty( c_sdlglext, "COMPARE_R_TO_TEXTURE" ).setInteger( GL_COMPARE_R_TO_TEXTURE );
      #endif

      #ifndef GL_VERSION_1_5
      self->addClassProperty( c_sdlglext, "BUFFER_SIZE" ).setInteger( GL_BUFFER_SIZE );
      self->addClassProperty( c_sdlglext, "BUFFER_USAGE" ).setInteger( GL_BUFFER_USAGE );
      self->addClassProperty( c_sdlglext, "QUERY_COUNTER_BITS" ).setInteger( GL_QUERY_COUNTER_BITS );
      self->addClassProperty( c_sdlglext, "CURRENT_QUERY" ).setInteger( GL_CURRENT_QUERY );
      self->addClassProperty( c_sdlglext, "QUERY_RESULT" ).setInteger( GL_QUERY_RESULT );
      self->addClassProperty( c_sdlglext, "QUERY_RESULT_AVAILABLE" ).setInteger( GL_QUERY_RESULT_AVAILABLE );
      self->addClassProperty( c_sdlglext, "ARRAY_BUFFER" ).setInteger( GL_ARRAY_BUFFER );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_BUFFER" ).setInteger( GL_ELEMENT_ARRAY_BUFFER );
      self->addClassProperty( c_sdlglext, "ARRAY_BUFFER_BINDING" ).setInteger( GL_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_BUFFER_BINDING" ).setInteger( GL_ELEMENT_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_BUFFER_BINDING" ).setInteger( GL_VERTEX_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_BUFFER_BINDING" ).setInteger( GL_NORMAL_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_BUFFER_BINDING" ).setInteger( GL_COLOR_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_BUFFER_BINDING" ).setInteger( GL_INDEX_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_BUFFER_BINDING" ).setInteger( GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "EDGE_FLAG_ARRAY_BUFFER_BINDING" ).setInteger( GL_EDGE_FLAG_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_BUFFER_BINDING" ).setInteger( GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_BUFFER_BINDING" ).setInteger( GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_BUFFER_BINDING" ).setInteger( GL_WEIGHT_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_BUFFER_BINDING" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING );
      self->addClassProperty( c_sdlglext, "READ_ONLY" ).setInteger( GL_READ_ONLY );
      self->addClassProperty( c_sdlglext, "WRITE_ONLY" ).setInteger( GL_WRITE_ONLY );
      self->addClassProperty( c_sdlglext, "READ_WRITE" ).setInteger( GL_READ_WRITE );
      self->addClassProperty( c_sdlglext, "BUFFER_ACCESS" ).setInteger( GL_BUFFER_ACCESS );
      self->addClassProperty( c_sdlglext, "BUFFER_MAPPED" ).setInteger( GL_BUFFER_MAPPED );
      self->addClassProperty( c_sdlglext, "BUFFER_MAP_POINTER" ).setInteger( GL_BUFFER_MAP_POINTER );
      self->addClassProperty( c_sdlglext, "STREAM_DRAW" ).setInteger( GL_STREAM_DRAW );
      self->addClassProperty( c_sdlglext, "STREAM_READ" ).setInteger( GL_STREAM_READ );
      self->addClassProperty( c_sdlglext, "STREAM_COPY" ).setInteger( GL_STREAM_COPY );
      self->addClassProperty( c_sdlglext, "STATIC_DRAW" ).setInteger( GL_STATIC_DRAW );
      self->addClassProperty( c_sdlglext, "STATIC_READ" ).setInteger( GL_STATIC_READ );
      self->addClassProperty( c_sdlglext, "STATIC_COPY" ).setInteger( GL_STATIC_COPY );
      self->addClassProperty( c_sdlglext, "DYNAMIC_DRAW" ).setInteger( GL_DYNAMIC_DRAW );
      self->addClassProperty( c_sdlglext, "DYNAMIC_READ" ).setInteger( GL_DYNAMIC_READ );
      self->addClassProperty( c_sdlglext, "DYNAMIC_COPY" ).setInteger( GL_DYNAMIC_COPY );
      self->addClassProperty( c_sdlglext, "SAMPLES_PASSED" ).setInteger( GL_SAMPLES_PASSED );
      #define GL_FOG_COORD_SRC                  GL_FOG_COORDINATE_SOURCE
      #define GL_FOG_COORD                      GL_FOG_COORDINATE
      #define GL_CURRENT_FOG_COORD              GL_CURRENT_FOG_COORDINATE
      #define GL_FOG_COORD_ARRAY_TYPE           GL_FOG_COORDINATE_ARRAY_TYPE
      #define GL_FOG_COORD_ARRAY_STRIDE         GL_FOG_COORDINATE_ARRAY_STRIDE
      #define GL_FOG_COORD_ARRAY_POINTER        GL_FOG_COORDINATE_ARRAY_POINTER
      #define GL_FOG_COORD_ARRAY                GL_FOG_COORDINATE_ARRAY
      #define GL_FOG_COORD_ARRAY_BUFFER_BINDING GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING
      #define GL_SRC0_RGB                       GL_SOURCE0_RGB
      #define GL_SRC1_RGB                       GL_SOURCE1_RGB
      #define GL_SRC2_RGB                       GL_SOURCE2_RGB
      #define GL_SRC0_ALPHA                     GL_SOURCE0_ALPHA
      #define GL_SRC1_ALPHA                     GL_SOURCE1_ALPHA
      #define GL_SRC2_ALPHA                     GL_SOURCE2_ALPHA
      #endif

      #ifndef GL_VERSION_2_0
      #define GL_BLEND_EQUATION_RGB             GL_BLEND_EQUATION
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_ENABLED" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_ENABLED );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_SIZE" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_SIZE );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_STRIDE" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_STRIDE );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_TYPE" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_TYPE );
      self->addClassProperty( c_sdlglext, "CURRENT_VERTEX_ATTRIB" ).setInteger( GL_CURRENT_VERTEX_ATTRIB );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_POINT_SIZE" ).setInteger( GL_VERTEX_PROGRAM_POINT_SIZE );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_TWO_SIDE" ).setInteger( GL_VERTEX_PROGRAM_TWO_SIDE );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_POINTER" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_POINTER );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_FUNC" ).setInteger( GL_STENCIL_BACK_FUNC );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_FAIL" ).setInteger( GL_STENCIL_BACK_FAIL );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_PASS_DEPTH_FAIL" ).setInteger( GL_STENCIL_BACK_PASS_DEPTH_FAIL );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_PASS_DEPTH_PASS" ).setInteger( GL_STENCIL_BACK_PASS_DEPTH_PASS );
      self->addClassProperty( c_sdlglext, "MAX_DRAW_BUFFERS" ).setInteger( GL_MAX_DRAW_BUFFERS );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER0" ).setInteger( GL_DRAW_BUFFER0 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER1" ).setInteger( GL_DRAW_BUFFER1 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER2" ).setInteger( GL_DRAW_BUFFER2 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER3" ).setInteger( GL_DRAW_BUFFER3 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER4" ).setInteger( GL_DRAW_BUFFER4 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER5" ).setInteger( GL_DRAW_BUFFER5 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER6" ).setInteger( GL_DRAW_BUFFER6 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER7" ).setInteger( GL_DRAW_BUFFER7 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER8" ).setInteger( GL_DRAW_BUFFER8 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER9" ).setInteger( GL_DRAW_BUFFER9 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER10" ).setInteger( GL_DRAW_BUFFER10 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER11" ).setInteger( GL_DRAW_BUFFER11 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER12" ).setInteger( GL_DRAW_BUFFER12 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER13" ).setInteger( GL_DRAW_BUFFER13 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER14" ).setInteger( GL_DRAW_BUFFER14 );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER15" ).setInteger( GL_DRAW_BUFFER15 );
      self->addClassProperty( c_sdlglext, "BLEND_EQUATION_ALPHA" ).setInteger( GL_BLEND_EQUATION_ALPHA );
      self->addClassProperty( c_sdlglext, "POINT_SPRITE" ).setInteger( GL_POINT_SPRITE );
      self->addClassProperty( c_sdlglext, "COORD_REPLACE" ).setInteger( GL_COORD_REPLACE );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_ATTRIBS" ).setInteger( GL_MAX_VERTEX_ATTRIBS );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_NORMALIZED" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_NORMALIZED );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_COORDS" ).setInteger( GL_MAX_TEXTURE_COORDS );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_IMAGE_UNITS" ).setInteger( GL_MAX_TEXTURE_IMAGE_UNITS );
      self->addClassProperty( c_sdlglext, "FRAGMENT_SHADER" ).setInteger( GL_FRAGMENT_SHADER );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER" ).setInteger( GL_VERTEX_SHADER );
      self->addClassProperty( c_sdlglext, "MAX_FRAGMENT_UNIFORM_COMPONENTS" ).setInteger( GL_MAX_FRAGMENT_UNIFORM_COMPONENTS );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_UNIFORM_COMPONENTS" ).setInteger( GL_MAX_VERTEX_UNIFORM_COMPONENTS );
      self->addClassProperty( c_sdlglext, "MAX_VARYING_FLOATS" ).setInteger( GL_MAX_VARYING_FLOATS );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_TEXTURE_IMAGE_UNITS" ).setInteger( GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS );
      self->addClassProperty( c_sdlglext, "MAX_COMBINED_TEXTURE_IMAGE_UNITS" ).setInteger( GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS );
      self->addClassProperty( c_sdlglext, "SHADER_TYPE" ).setInteger( GL_SHADER_TYPE );
      self->addClassProperty( c_sdlglext, "FLOAT_VEC2" ).setInteger( GL_FLOAT_VEC2 );
      self->addClassProperty( c_sdlglext, "FLOAT_VEC3" ).setInteger( GL_FLOAT_VEC3 );
      self->addClassProperty( c_sdlglext, "FLOAT_VEC4" ).setInteger( GL_FLOAT_VEC4 );
      self->addClassProperty( c_sdlglext, "INT_VEC2" ).setInteger( GL_INT_VEC2 );
      self->addClassProperty( c_sdlglext, "INT_VEC3" ).setInteger( GL_INT_VEC3 );
      self->addClassProperty( c_sdlglext, "INT_VEC4" ).setInteger( GL_INT_VEC4 );
      self->addClassProperty( c_sdlglext, "BOOL" ).setInteger( GL_BOOL );
      self->addClassProperty( c_sdlglext, "BOOL_VEC2" ).setInteger( GL_BOOL_VEC2 );
      self->addClassProperty( c_sdlglext, "BOOL_VEC3" ).setInteger( GL_BOOL_VEC3 );
      self->addClassProperty( c_sdlglext, "BOOL_VEC4" ).setInteger( GL_BOOL_VEC4 );
      self->addClassProperty( c_sdlglext, "FLOAT_MAT2" ).setInteger( GL_FLOAT_MAT2 );
      self->addClassProperty( c_sdlglext, "FLOAT_MAT3" ).setInteger( GL_FLOAT_MAT3 );
      self->addClassProperty( c_sdlglext, "FLOAT_MAT4" ).setInteger( GL_FLOAT_MAT4 );
      self->addClassProperty( c_sdlglext, "SAMPLER_1D" ).setInteger( GL_SAMPLER_1D );
      self->addClassProperty( c_sdlglext, "SAMPLER_2D" ).setInteger( GL_SAMPLER_2D );
      self->addClassProperty( c_sdlglext, "SAMPLER_3D" ).setInteger( GL_SAMPLER_3D );
      self->addClassProperty( c_sdlglext, "SAMPLER_CUBE" ).setInteger( GL_SAMPLER_CUBE );
      self->addClassProperty( c_sdlglext, "SAMPLER_1D_SHADOW" ).setInteger( GL_SAMPLER_1D_SHADOW );
      self->addClassProperty( c_sdlglext, "SAMPLER_2D_SHADOW" ).setInteger( GL_SAMPLER_2D_SHADOW );
      self->addClassProperty( c_sdlglext, "DELETE_STATUS" ).setInteger( GL_DELETE_STATUS );
      self->addClassProperty( c_sdlglext, "COMPILE_STATUS" ).setInteger( GL_COMPILE_STATUS );
      self->addClassProperty( c_sdlglext, "LINK_STATUS" ).setInteger( GL_LINK_STATUS );
      self->addClassProperty( c_sdlglext, "VALIDATE_STATUS" ).setInteger( GL_VALIDATE_STATUS );
      self->addClassProperty( c_sdlglext, "INFO_LOG_LENGTH" ).setInteger( GL_INFO_LOG_LENGTH );
      self->addClassProperty( c_sdlglext, "ATTACHED_SHADERS" ).setInteger( GL_ATTACHED_SHADERS );
      self->addClassProperty( c_sdlglext, "ACTIVE_UNIFORMS" ).setInteger( GL_ACTIVE_UNIFORMS );
      self->addClassProperty( c_sdlglext, "ACTIVE_UNIFORM_MAX_LENGTH" ).setInteger( GL_ACTIVE_UNIFORM_MAX_LENGTH );
      self->addClassProperty( c_sdlglext, "SHADER_SOURCE_LENGTH" ).setInteger( GL_SHADER_SOURCE_LENGTH );
      self->addClassProperty( c_sdlglext, "ACTIVE_ATTRIBUTES" ).setInteger( GL_ACTIVE_ATTRIBUTES );
      self->addClassProperty( c_sdlglext, "ACTIVE_ATTRIBUTE_MAX_LENGTH" ).setInteger( GL_ACTIVE_ATTRIBUTE_MAX_LENGTH );
      self->addClassProperty( c_sdlglext, "FRAGMENT_SHADER_DERIVATIVE_HINT" ).setInteger( GL_FRAGMENT_SHADER_DERIVATIVE_HINT );
      self->addClassProperty( c_sdlglext, "SHADING_LANGUAGE_VERSION" ).setInteger( GL_SHADING_LANGUAGE_VERSION );
      self->addClassProperty( c_sdlglext, "CURRENT_PROGRAM" ).setInteger( GL_CURRENT_PROGRAM );
      self->addClassProperty( c_sdlglext, "POINT_SPRITE_COORD_ORIGIN" ).setInteger( GL_POINT_SPRITE_COORD_ORIGIN );
      self->addClassProperty( c_sdlglext, "LOWER_LEFT" ).setInteger( GL_LOWER_LEFT );
      self->addClassProperty( c_sdlglext, "UPPER_LEFT" ).setInteger( GL_UPPER_LEFT );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_REF" ).setInteger( GL_STENCIL_BACK_REF );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_VALUE_MASK" ).setInteger( GL_STENCIL_BACK_VALUE_MASK );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_WRITEMASK" ).setInteger( GL_STENCIL_BACK_WRITEMASK );
      #endif

      #ifndef GL_ARB_multitexture
      self->addClassProperty( c_sdlglext, "TEXTURE0_ARB" ).setInteger( GL_TEXTURE0_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE1_ARB" ).setInteger( GL_TEXTURE1_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE2_ARB" ).setInteger( GL_TEXTURE2_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE3_ARB" ).setInteger( GL_TEXTURE3_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE4_ARB" ).setInteger( GL_TEXTURE4_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE5_ARB" ).setInteger( GL_TEXTURE5_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE6_ARB" ).setInteger( GL_TEXTURE6_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE7_ARB" ).setInteger( GL_TEXTURE7_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE8_ARB" ).setInteger( GL_TEXTURE8_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE9_ARB" ).setInteger( GL_TEXTURE9_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE10_ARB" ).setInteger( GL_TEXTURE10_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE11_ARB" ).setInteger( GL_TEXTURE11_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE12_ARB" ).setInteger( GL_TEXTURE12_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE13_ARB" ).setInteger( GL_TEXTURE13_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE14_ARB" ).setInteger( GL_TEXTURE14_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE15_ARB" ).setInteger( GL_TEXTURE15_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE16_ARB" ).setInteger( GL_TEXTURE16_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE17_ARB" ).setInteger( GL_TEXTURE17_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE18_ARB" ).setInteger( GL_TEXTURE18_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE19_ARB" ).setInteger( GL_TEXTURE19_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE20_ARB" ).setInteger( GL_TEXTURE20_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE21_ARB" ).setInteger( GL_TEXTURE21_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE22_ARB" ).setInteger( GL_TEXTURE22_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE23_ARB" ).setInteger( GL_TEXTURE23_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE24_ARB" ).setInteger( GL_TEXTURE24_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE25_ARB" ).setInteger( GL_TEXTURE25_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE26_ARB" ).setInteger( GL_TEXTURE26_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE27_ARB" ).setInteger( GL_TEXTURE27_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE28_ARB" ).setInteger( GL_TEXTURE28_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE29_ARB" ).setInteger( GL_TEXTURE29_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE30_ARB" ).setInteger( GL_TEXTURE30_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE31_ARB" ).setInteger( GL_TEXTURE31_ARB );
      self->addClassProperty( c_sdlglext, "ACTIVE_TEXTURE_ARB" ).setInteger( GL_ACTIVE_TEXTURE_ARB );
      self->addClassProperty( c_sdlglext, "CLIENT_ACTIVE_TEXTURE_ARB" ).setInteger( GL_CLIENT_ACTIVE_TEXTURE_ARB );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_UNITS_ARB" ).setInteger( GL_MAX_TEXTURE_UNITS_ARB );
      #endif

      #ifndef GL_ARB_transpose_matrix
      self->addClassProperty( c_sdlglext, "TRANSPOSE_MODELVIEW_MATRIX_ARB" ).setInteger( GL_TRANSPOSE_MODELVIEW_MATRIX_ARB );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_PROJECTION_MATRIX_ARB" ).setInteger( GL_TRANSPOSE_PROJECTION_MATRIX_ARB );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_TEXTURE_MATRIX_ARB" ).setInteger( GL_TRANSPOSE_TEXTURE_MATRIX_ARB );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_COLOR_MATRIX_ARB" ).setInteger( GL_TRANSPOSE_COLOR_MATRIX_ARB );
      #endif

      #ifndef GL_ARB_multisample
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_ARB" ).setInteger( GL_MULTISAMPLE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_COVERAGE_ARB" ).setInteger( GL_SAMPLE_ALPHA_TO_COVERAGE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_ONE_ARB" ).setInteger( GL_SAMPLE_ALPHA_TO_ONE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLE_COVERAGE_ARB" ).setInteger( GL_SAMPLE_COVERAGE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLE_BUFFERS_ARB" ).setInteger( GL_SAMPLE_BUFFERS_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLES_ARB" ).setInteger( GL_SAMPLES_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLE_COVERAGE_VALUE_ARB" ).setInteger( GL_SAMPLE_COVERAGE_VALUE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLE_COVERAGE_INVERT_ARB" ).setInteger( GL_SAMPLE_COVERAGE_INVERT_ARB );
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_BIT_ARB" ).setInteger( GL_MULTISAMPLE_BIT_ARB );
      #endif

      #ifndef GL_ARB_texture_cube_map
      self->addClassProperty( c_sdlglext, "NORMAL_MAP_ARB" ).setInteger( GL_NORMAL_MAP_ARB );
      self->addClassProperty( c_sdlglext, "REFLECTION_MAP_ARB" ).setInteger( GL_REFLECTION_MAP_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_BINDING_CUBE_MAP_ARB" ).setInteger( GL_TEXTURE_BINDING_CUBE_MAP_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_X_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_X_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_Y_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_Z_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_CUBE_MAP_ARB" ).setInteger( GL_PROXY_TEXTURE_CUBE_MAP_ARB );
      self->addClassProperty( c_sdlglext, "MAX_CUBE_MAP_TEXTURE_SIZE_ARB" ).setInteger( GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB );
      #endif

      #ifndef GL_ARB_texture_compression
      self->addClassProperty( c_sdlglext, "COMPRESSED_ALPHA_ARB" ).setInteger( GL_COMPRESSED_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_LUMINANCE_ARB" ).setInteger( GL_COMPRESSED_LUMINANCE_ARB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_LUMINANCE_ALPHA_ARB" ).setInteger( GL_COMPRESSED_LUMINANCE_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_INTENSITY_ARB" ).setInteger( GL_COMPRESSED_INTENSITY_ARB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGB_ARB" ).setInteger( GL_COMPRESSED_RGB_ARB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGBA_ARB" ).setInteger( GL_COMPRESSED_RGBA_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPRESSION_HINT_ARB" ).setInteger( GL_TEXTURE_COMPRESSION_HINT_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPRESSED_IMAGE_SIZE_ARB" ).setInteger( GL_TEXTURE_COMPRESSED_IMAGE_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPRESSED_ARB" ).setInteger( GL_TEXTURE_COMPRESSED_ARB );
      self->addClassProperty( c_sdlglext, "NUM_COMPRESSED_TEXTURE_FORMATS_ARB" ).setInteger( GL_NUM_COMPRESSED_TEXTURE_FORMATS_ARB );
      self->addClassProperty( c_sdlglext, "COMPRESSED_TEXTURE_FORMATS_ARB" ).setInteger( GL_COMPRESSED_TEXTURE_FORMATS_ARB );
      #endif

      #ifndef GL_ARB_texture_border_clamp
      self->addClassProperty( c_sdlglext, "CLAMP_TO_BORDER_ARB" ).setInteger( GL_CLAMP_TO_BORDER_ARB );
      #endif

      #ifndef GL_ARB_point_parameters
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MIN_ARB" ).setInteger( GL_POINT_SIZE_MIN_ARB );
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MAX_ARB" ).setInteger( GL_POINT_SIZE_MAX_ARB );
      self->addClassProperty( c_sdlglext, "POINT_FADE_THRESHOLD_SIZE_ARB" ).setInteger( GL_POINT_FADE_THRESHOLD_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "POINT_DISTANCE_ATTENUATION_ARB" ).setInteger( GL_POINT_DISTANCE_ATTENUATION_ARB );
      #endif

      #ifndef GL_ARB_vertex_blend
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_UNITS_ARB" ).setInteger( GL_MAX_VERTEX_UNITS_ARB );
      self->addClassProperty( c_sdlglext, "ACTIVE_VERTEX_UNITS_ARB" ).setInteger( GL_ACTIVE_VERTEX_UNITS_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_SUM_UNITY_ARB" ).setInteger( GL_WEIGHT_SUM_UNITY_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_BLEND_ARB" ).setInteger( GL_VERTEX_BLEND_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_WEIGHT_ARB" ).setInteger( GL_CURRENT_WEIGHT_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_TYPE_ARB" ).setInteger( GL_WEIGHT_ARRAY_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_STRIDE_ARB" ).setInteger( GL_WEIGHT_ARRAY_STRIDE_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_SIZE_ARB" ).setInteger( GL_WEIGHT_ARRAY_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_POINTER_ARB" ).setInteger( GL_WEIGHT_ARRAY_POINTER_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_ARB" ).setInteger( GL_WEIGHT_ARRAY_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW0_ARB" ).setInteger( GL_MODELVIEW0_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW1_ARB" ).setInteger( GL_MODELVIEW1_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW2_ARB" ).setInteger( GL_MODELVIEW2_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW3_ARB" ).setInteger( GL_MODELVIEW3_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW4_ARB" ).setInteger( GL_MODELVIEW4_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW5_ARB" ).setInteger( GL_MODELVIEW5_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW6_ARB" ).setInteger( GL_MODELVIEW6_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW7_ARB" ).setInteger( GL_MODELVIEW7_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW8_ARB" ).setInteger( GL_MODELVIEW8_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW9_ARB" ).setInteger( GL_MODELVIEW9_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW10_ARB" ).setInteger( GL_MODELVIEW10_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW11_ARB" ).setInteger( GL_MODELVIEW11_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW12_ARB" ).setInteger( GL_MODELVIEW12_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW13_ARB" ).setInteger( GL_MODELVIEW13_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW14_ARB" ).setInteger( GL_MODELVIEW14_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW15_ARB" ).setInteger( GL_MODELVIEW15_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW16_ARB" ).setInteger( GL_MODELVIEW16_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW17_ARB" ).setInteger( GL_MODELVIEW17_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW18_ARB" ).setInteger( GL_MODELVIEW18_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW19_ARB" ).setInteger( GL_MODELVIEW19_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW20_ARB" ).setInteger( GL_MODELVIEW20_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW21_ARB" ).setInteger( GL_MODELVIEW21_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW22_ARB" ).setInteger( GL_MODELVIEW22_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW23_ARB" ).setInteger( GL_MODELVIEW23_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW24_ARB" ).setInteger( GL_MODELVIEW24_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW25_ARB" ).setInteger( GL_MODELVIEW25_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW26_ARB" ).setInteger( GL_MODELVIEW26_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW27_ARB" ).setInteger( GL_MODELVIEW27_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW28_ARB" ).setInteger( GL_MODELVIEW28_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW29_ARB" ).setInteger( GL_MODELVIEW29_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW30_ARB" ).setInteger( GL_MODELVIEW30_ARB );
      self->addClassProperty( c_sdlglext, "MODELVIEW31_ARB" ).setInteger( GL_MODELVIEW31_ARB );
      #endif

      #ifndef GL_ARB_matrix_palette
      self->addClassProperty( c_sdlglext, "MATRIX_PALETTE_ARB" ).setInteger( GL_MATRIX_PALETTE_ARB );
      self->addClassProperty( c_sdlglext, "MAX_MATRIX_PALETTE_STACK_DEPTH_ARB" ).setInteger( GL_MAX_MATRIX_PALETTE_STACK_DEPTH_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PALETTE_MATRICES_ARB" ).setInteger( GL_MAX_PALETTE_MATRICES_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_PALETTE_MATRIX_ARB" ).setInteger( GL_CURRENT_PALETTE_MATRIX_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX_INDEX_ARRAY_ARB" ).setInteger( GL_MATRIX_INDEX_ARRAY_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_MATRIX_INDEX_ARB" ).setInteger( GL_CURRENT_MATRIX_INDEX_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX_INDEX_ARRAY_SIZE_ARB" ).setInteger( GL_MATRIX_INDEX_ARRAY_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX_INDEX_ARRAY_TYPE_ARB" ).setInteger( GL_MATRIX_INDEX_ARRAY_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX_INDEX_ARRAY_STRIDE_ARB" ).setInteger( GL_MATRIX_INDEX_ARRAY_STRIDE_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX_INDEX_ARRAY_POINTER_ARB" ).setInteger( GL_MATRIX_INDEX_ARRAY_POINTER_ARB );
      #endif

      #ifndef GL_ARB_texture_env_combine
      self->addClassProperty( c_sdlglext, "COMBINE_ARB" ).setInteger( GL_COMBINE_ARB );
      self->addClassProperty( c_sdlglext, "COMBINE_RGB_ARB" ).setInteger( GL_COMBINE_RGB_ARB );
      self->addClassProperty( c_sdlglext, "COMBINE_ALPHA_ARB" ).setInteger( GL_COMBINE_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "SOURCE0_RGB_ARB" ).setInteger( GL_SOURCE0_RGB_ARB );
      self->addClassProperty( c_sdlglext, "SOURCE1_RGB_ARB" ).setInteger( GL_SOURCE1_RGB_ARB );
      self->addClassProperty( c_sdlglext, "SOURCE2_RGB_ARB" ).setInteger( GL_SOURCE2_RGB_ARB );
      self->addClassProperty( c_sdlglext, "SOURCE0_ALPHA_ARB" ).setInteger( GL_SOURCE0_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "SOURCE1_ALPHA_ARB" ).setInteger( GL_SOURCE1_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "SOURCE2_ALPHA_ARB" ).setInteger( GL_SOURCE2_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "OPERAND0_RGB_ARB" ).setInteger( GL_OPERAND0_RGB_ARB );
      self->addClassProperty( c_sdlglext, "OPERAND1_RGB_ARB" ).setInteger( GL_OPERAND1_RGB_ARB );
      self->addClassProperty( c_sdlglext, "OPERAND2_RGB_ARB" ).setInteger( GL_OPERAND2_RGB_ARB );
      self->addClassProperty( c_sdlglext, "OPERAND0_ALPHA_ARB" ).setInteger( GL_OPERAND0_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "OPERAND1_ALPHA_ARB" ).setInteger( GL_OPERAND1_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "OPERAND2_ALPHA_ARB" ).setInteger( GL_OPERAND2_ALPHA_ARB );
      self->addClassProperty( c_sdlglext, "RGB_SCALE_ARB" ).setInteger( GL_RGB_SCALE_ARB );
      self->addClassProperty( c_sdlglext, "ADD_SIGNED_ARB" ).setInteger( GL_ADD_SIGNED_ARB );
      self->addClassProperty( c_sdlglext, "INTERPOLATE_ARB" ).setInteger( GL_INTERPOLATE_ARB );
      self->addClassProperty( c_sdlglext, "SUBTRACT_ARB" ).setInteger( GL_SUBTRACT_ARB );
      self->addClassProperty( c_sdlglext, "CONSTANT_ARB" ).setInteger( GL_CONSTANT_ARB );
      self->addClassProperty( c_sdlglext, "PRIMARY_COLOR_ARB" ).setInteger( GL_PRIMARY_COLOR_ARB );
      self->addClassProperty( c_sdlglext, "PREVIOUS_ARB" ).setInteger( GL_PREVIOUS_ARB );
      #endif

      #ifndef GL_ARB_texture_env_dot3
      self->addClassProperty( c_sdlglext, "DOT3_RGB_ARB" ).setInteger( GL_DOT3_RGB_ARB );
      self->addClassProperty( c_sdlglext, "DOT3_RGBA_ARB" ).setInteger( GL_DOT3_RGBA_ARB );
      #endif

      #ifndef GL_ARB_texture_mirrored_repeat
      self->addClassProperty( c_sdlglext, "MIRRORED_REPEAT_ARB" ).setInteger( GL_MIRRORED_REPEAT_ARB );
      #endif

      #ifndef GL_ARB_depth_texture
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT16_ARB" ).setInteger( GL_DEPTH_COMPONENT16_ARB );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT24_ARB" ).setInteger( GL_DEPTH_COMPONENT24_ARB );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT32_ARB" ).setInteger( GL_DEPTH_COMPONENT32_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_DEPTH_SIZE_ARB" ).setInteger( GL_TEXTURE_DEPTH_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "DEPTH_TEXTURE_MODE_ARB" ).setInteger( GL_DEPTH_TEXTURE_MODE_ARB );
      #endif

      #ifndef GL_ARB_shadow
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_MODE_ARB" ).setInteger( GL_TEXTURE_COMPARE_MODE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_FUNC_ARB" ).setInteger( GL_TEXTURE_COMPARE_FUNC_ARB );
      self->addClassProperty( c_sdlglext, "COMPARE_R_TO_TEXTURE_ARB" ).setInteger( GL_COMPARE_R_TO_TEXTURE_ARB );
      #endif

      #ifndef GL_ARB_shadow_ambient
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_FAIL_VALUE_ARB" ).setInteger( GL_TEXTURE_COMPARE_FAIL_VALUE_ARB );
      #endif

      #ifndef GL_ARB_vertex_program
      self->addClassProperty( c_sdlglext, "COLOR_SUM_ARB" ).setInteger( GL_COLOR_SUM_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_ARB" ).setInteger( GL_VERTEX_PROGRAM_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_ENABLED_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_ENABLED_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_SIZE_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_STRIDE_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_STRIDE_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_TYPE_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_VERTEX_ATTRIB_ARB" ).setInteger( GL_CURRENT_VERTEX_ATTRIB_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_LENGTH_ARB" ).setInteger( GL_PROGRAM_LENGTH_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_STRING_ARB" ).setInteger( GL_PROGRAM_STRING_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB" ).setInteger( GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_MATRICES_ARB" ).setInteger( GL_MAX_PROGRAM_MATRICES_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_MATRIX_STACK_DEPTH_ARB" ).setInteger( GL_CURRENT_MATRIX_STACK_DEPTH_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_MATRIX_ARB" ).setInteger( GL_CURRENT_MATRIX_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_POINT_SIZE_ARB" ).setInteger( GL_VERTEX_PROGRAM_POINT_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_TWO_SIDE_ARB" ).setInteger( GL_VERTEX_PROGRAM_TWO_SIDE_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_POINTER_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_POINTER_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_ERROR_POSITION_ARB" ).setInteger( GL_PROGRAM_ERROR_POSITION_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_BINDING_ARB" ).setInteger( GL_PROGRAM_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_ATTRIBS_ARB" ).setInteger( GL_MAX_VERTEX_ATTRIBS_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_NORMALIZED_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_NORMALIZED_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_ERROR_STRING_ARB" ).setInteger( GL_PROGRAM_ERROR_STRING_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_FORMAT_ASCII_ARB" ).setInteger( GL_PROGRAM_FORMAT_ASCII_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_FORMAT_ARB" ).setInteger( GL_PROGRAM_FORMAT_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_INSTRUCTIONS_ARB" ).setInteger( GL_PROGRAM_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_INSTRUCTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_INSTRUCTIONS_ARB" ).setInteger( GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_TEMPORARIES_ARB" ).setInteger( GL_PROGRAM_TEMPORARIES_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_TEMPORARIES_ARB" ).setInteger( GL_MAX_PROGRAM_TEMPORARIES_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_TEMPORARIES_ARB" ).setInteger( GL_PROGRAM_NATIVE_TEMPORARIES_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_TEMPORARIES_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_PARAMETERS_ARB" ).setInteger( GL_PROGRAM_PARAMETERS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_PARAMETERS_ARB" ).setInteger( GL_MAX_PROGRAM_PARAMETERS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_PARAMETERS_ARB" ).setInteger( GL_PROGRAM_NATIVE_PARAMETERS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_PARAMETERS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_ATTRIBS_ARB" ).setInteger( GL_PROGRAM_ATTRIBS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_ATTRIBS_ARB" ).setInteger( GL_MAX_PROGRAM_ATTRIBS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_ATTRIBS_ARB" ).setInteger( GL_PROGRAM_NATIVE_ATTRIBS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_ATTRIBS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_ADDRESS_REGISTERS_ARB" ).setInteger( GL_PROGRAM_ADDRESS_REGISTERS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_ADDRESS_REGISTERS_ARB" ).setInteger( GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB" ).setInteger( GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_LOCAL_PARAMETERS_ARB" ).setInteger( GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_ENV_PARAMETERS_ARB" ).setInteger( GL_MAX_PROGRAM_ENV_PARAMETERS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_UNDER_NATIVE_LIMITS_ARB" ).setInteger( GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_CURRENT_MATRIX_ARB" ).setInteger( GL_TRANSPOSE_CURRENT_MATRIX_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX0_ARB" ).setInteger( GL_MATRIX0_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX1_ARB" ).setInteger( GL_MATRIX1_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX2_ARB" ).setInteger( GL_MATRIX2_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX3_ARB" ).setInteger( GL_MATRIX3_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX4_ARB" ).setInteger( GL_MATRIX4_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX5_ARB" ).setInteger( GL_MATRIX5_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX6_ARB" ).setInteger( GL_MATRIX6_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX7_ARB" ).setInteger( GL_MATRIX7_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX8_ARB" ).setInteger( GL_MATRIX8_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX9_ARB" ).setInteger( GL_MATRIX9_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX10_ARB" ).setInteger( GL_MATRIX10_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX11_ARB" ).setInteger( GL_MATRIX11_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX12_ARB" ).setInteger( GL_MATRIX12_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX13_ARB" ).setInteger( GL_MATRIX13_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX14_ARB" ).setInteger( GL_MATRIX14_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX15_ARB" ).setInteger( GL_MATRIX15_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX16_ARB" ).setInteger( GL_MATRIX16_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX17_ARB" ).setInteger( GL_MATRIX17_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX18_ARB" ).setInteger( GL_MATRIX18_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX19_ARB" ).setInteger( GL_MATRIX19_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX20_ARB" ).setInteger( GL_MATRIX20_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX21_ARB" ).setInteger( GL_MATRIX21_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX22_ARB" ).setInteger( GL_MATRIX22_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX23_ARB" ).setInteger( GL_MATRIX23_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX24_ARB" ).setInteger( GL_MATRIX24_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX25_ARB" ).setInteger( GL_MATRIX25_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX26_ARB" ).setInteger( GL_MATRIX26_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX27_ARB" ).setInteger( GL_MATRIX27_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX28_ARB" ).setInteger( GL_MATRIX28_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX29_ARB" ).setInteger( GL_MATRIX29_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX30_ARB" ).setInteger( GL_MATRIX30_ARB );
      self->addClassProperty( c_sdlglext, "MATRIX31_ARB" ).setInteger( GL_MATRIX31_ARB );
      #endif

      #ifndef GL_ARB_fragment_program
      self->addClassProperty( c_sdlglext, "FRAGMENT_PROGRAM_ARB" ).setInteger( GL_FRAGMENT_PROGRAM_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_ALU_INSTRUCTIONS_ARB" ).setInteger( GL_PROGRAM_ALU_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_TEX_INSTRUCTIONS_ARB" ).setInteger( GL_PROGRAM_TEX_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_TEX_INDIRECTIONS_ARB" ).setInteger( GL_PROGRAM_TEX_INDIRECTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB" ).setInteger( GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB" ).setInteger( GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB" ).setInteger( GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_ALU_INSTRUCTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_TEX_INSTRUCTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_TEX_INDIRECTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB" ).setInteger( GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_COORDS_ARB" ).setInteger( GL_MAX_TEXTURE_COORDS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_IMAGE_UNITS_ARB" ).setInteger( GL_MAX_TEXTURE_IMAGE_UNITS_ARB );
      #endif

      #ifndef GL_ARB_vertex_buffer_object
      self->addClassProperty( c_sdlglext, "BUFFER_SIZE_ARB" ).setInteger( GL_BUFFER_SIZE_ARB );
      self->addClassProperty( c_sdlglext, "BUFFER_USAGE_ARB" ).setInteger( GL_BUFFER_USAGE_ARB );
      self->addClassProperty( c_sdlglext, "ARRAY_BUFFER_ARB" ).setInteger( GL_ARRAY_BUFFER_ARB );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_BUFFER_ARB" ).setInteger( GL_ELEMENT_ARRAY_BUFFER_ARB );
      self->addClassProperty( c_sdlglext, "ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_ELEMENT_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_VERTEX_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_NORMAL_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_COLOR_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_INDEX_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "EDGE_FLAG_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_EDGE_FLAG_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "WEIGHT_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_WEIGHT_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY_BUFFER_BINDING_ARB" ).setInteger( GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "READ_ONLY_ARB" ).setInteger( GL_READ_ONLY_ARB );
      self->addClassProperty( c_sdlglext, "WRITE_ONLY_ARB" ).setInteger( GL_WRITE_ONLY_ARB );
      self->addClassProperty( c_sdlglext, "READ_WRITE_ARB" ).setInteger( GL_READ_WRITE_ARB );
      self->addClassProperty( c_sdlglext, "BUFFER_ACCESS_ARB" ).setInteger( GL_BUFFER_ACCESS_ARB );
      self->addClassProperty( c_sdlglext, "BUFFER_MAPPED_ARB" ).setInteger( GL_BUFFER_MAPPED_ARB );
      self->addClassProperty( c_sdlglext, "BUFFER_MAP_POINTER_ARB" ).setInteger( GL_BUFFER_MAP_POINTER_ARB );
      self->addClassProperty( c_sdlglext, "STREAM_DRAW_ARB" ).setInteger( GL_STREAM_DRAW_ARB );
      self->addClassProperty( c_sdlglext, "STREAM_READ_ARB" ).setInteger( GL_STREAM_READ_ARB );
      self->addClassProperty( c_sdlglext, "STREAM_COPY_ARB" ).setInteger( GL_STREAM_COPY_ARB );
      self->addClassProperty( c_sdlglext, "STATIC_DRAW_ARB" ).setInteger( GL_STATIC_DRAW_ARB );
      self->addClassProperty( c_sdlglext, "STATIC_READ_ARB" ).setInteger( GL_STATIC_READ_ARB );
      self->addClassProperty( c_sdlglext, "STATIC_COPY_ARB" ).setInteger( GL_STATIC_COPY_ARB );
      self->addClassProperty( c_sdlglext, "DYNAMIC_DRAW_ARB" ).setInteger( GL_DYNAMIC_DRAW_ARB );
      self->addClassProperty( c_sdlglext, "DYNAMIC_READ_ARB" ).setInteger( GL_DYNAMIC_READ_ARB );
      self->addClassProperty( c_sdlglext, "DYNAMIC_COPY_ARB" ).setInteger( GL_DYNAMIC_COPY_ARB );
      #endif

      #ifndef GL_ARB_occlusion_query
      self->addClassProperty( c_sdlglext, "QUERY_COUNTER_BITS_ARB" ).setInteger( GL_QUERY_COUNTER_BITS_ARB );
      self->addClassProperty( c_sdlglext, "CURRENT_QUERY_ARB" ).setInteger( GL_CURRENT_QUERY_ARB );
      self->addClassProperty( c_sdlglext, "QUERY_RESULT_ARB" ).setInteger( GL_QUERY_RESULT_ARB );
      self->addClassProperty( c_sdlglext, "QUERY_RESULT_AVAILABLE_ARB" ).setInteger( GL_QUERY_RESULT_AVAILABLE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLES_PASSED_ARB" ).setInteger( GL_SAMPLES_PASSED_ARB );
      #endif

      #ifndef GL_ARB_shader_objects
      self->addClassProperty( c_sdlglext, "PROGRAM_OBJECT_ARB" ).setInteger( GL_PROGRAM_OBJECT_ARB );
      self->addClassProperty( c_sdlglext, "SHADER_OBJECT_ARB" ).setInteger( GL_SHADER_OBJECT_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_TYPE_ARB" ).setInteger( GL_OBJECT_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_SUBTYPE_ARB" ).setInteger( GL_OBJECT_SUBTYPE_ARB );
      self->addClassProperty( c_sdlglext, "FLOAT_VEC2_ARB" ).setInteger( GL_FLOAT_VEC2_ARB );
      self->addClassProperty( c_sdlglext, "FLOAT_VEC3_ARB" ).setInteger( GL_FLOAT_VEC3_ARB );
      self->addClassProperty( c_sdlglext, "FLOAT_VEC4_ARB" ).setInteger( GL_FLOAT_VEC4_ARB );
      self->addClassProperty( c_sdlglext, "INT_VEC2_ARB" ).setInteger( GL_INT_VEC2_ARB );
      self->addClassProperty( c_sdlglext, "INT_VEC3_ARB" ).setInteger( GL_INT_VEC3_ARB );
      self->addClassProperty( c_sdlglext, "INT_VEC4_ARB" ).setInteger( GL_INT_VEC4_ARB );
      self->addClassProperty( c_sdlglext, "BOOL_ARB" ).setInteger( GL_BOOL_ARB );
      self->addClassProperty( c_sdlglext, "BOOL_VEC2_ARB" ).setInteger( GL_BOOL_VEC2_ARB );
      self->addClassProperty( c_sdlglext, "BOOL_VEC3_ARB" ).setInteger( GL_BOOL_VEC3_ARB );
      self->addClassProperty( c_sdlglext, "BOOL_VEC4_ARB" ).setInteger( GL_BOOL_VEC4_ARB );
      self->addClassProperty( c_sdlglext, "FLOAT_MAT2_ARB" ).setInteger( GL_FLOAT_MAT2_ARB );
      self->addClassProperty( c_sdlglext, "FLOAT_MAT3_ARB" ).setInteger( GL_FLOAT_MAT3_ARB );
      self->addClassProperty( c_sdlglext, "FLOAT_MAT4_ARB" ).setInteger( GL_FLOAT_MAT4_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_1D_ARB" ).setInteger( GL_SAMPLER_1D_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_2D_ARB" ).setInteger( GL_SAMPLER_2D_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_3D_ARB" ).setInteger( GL_SAMPLER_3D_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_CUBE_ARB" ).setInteger( GL_SAMPLER_CUBE_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_1D_SHADOW_ARB" ).setInteger( GL_SAMPLER_1D_SHADOW_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_2D_SHADOW_ARB" ).setInteger( GL_SAMPLER_2D_SHADOW_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_2D_RECT_ARB" ).setInteger( GL_SAMPLER_2D_RECT_ARB );
      self->addClassProperty( c_sdlglext, "SAMPLER_2D_RECT_SHADOW_ARB" ).setInteger( GL_SAMPLER_2D_RECT_SHADOW_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_DELETE_STATUS_ARB" ).setInteger( GL_OBJECT_DELETE_STATUS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_COMPILE_STATUS_ARB" ).setInteger( GL_OBJECT_COMPILE_STATUS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_LINK_STATUS_ARB" ).setInteger( GL_OBJECT_LINK_STATUS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_VALIDATE_STATUS_ARB" ).setInteger( GL_OBJECT_VALIDATE_STATUS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_INFO_LOG_LENGTH_ARB" ).setInteger( GL_OBJECT_INFO_LOG_LENGTH_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_ATTACHED_OBJECTS_ARB" ).setInteger( GL_OBJECT_ATTACHED_OBJECTS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_ACTIVE_UNIFORMS_ARB" ).setInteger( GL_OBJECT_ACTIVE_UNIFORMS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB" ).setInteger( GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_SHADER_SOURCE_LENGTH_ARB" ).setInteger( GL_OBJECT_SHADER_SOURCE_LENGTH_ARB );
      #endif

      #ifndef GL_ARB_vertex_shader
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_ARB" ).setInteger( GL_VERTEX_SHADER_ARB );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_UNIFORM_COMPONENTS_ARB" ).setInteger( GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_VARYING_FLOATS_ARB" ).setInteger( GL_MAX_VARYING_FLOATS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB" ).setInteger( GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB );
      self->addClassProperty( c_sdlglext, "MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB" ).setInteger( GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_ACTIVE_ATTRIBUTES_ARB" ).setInteger( GL_OBJECT_ACTIVE_ATTRIBUTES_ARB );
      self->addClassProperty( c_sdlglext, "OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH_ARB" ).setInteger( GL_OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH_ARB );
      #endif

      #ifndef GL_ARB_fragment_shader
      self->addClassProperty( c_sdlglext, "FRAGMENT_SHADER_ARB" ).setInteger( GL_FRAGMENT_SHADER_ARB );
      self->addClassProperty( c_sdlglext, "MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB" ).setInteger( GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB );
      self->addClassProperty( c_sdlglext, "FRAGMENT_SHADER_DERIVATIVE_HINT_ARB" ).setInteger( GL_FRAGMENT_SHADER_DERIVATIVE_HINT_ARB );
      #endif

      #ifndef GL_ARB_shading_language_100
      self->addClassProperty( c_sdlglext, "SHADING_LANGUAGE_VERSION_ARB" ).setInteger( GL_SHADING_LANGUAGE_VERSION_ARB );
      #endif

      #ifndef GL_ARB_point_sprite
      self->addClassProperty( c_sdlglext, "POINT_SPRITE_ARB" ).setInteger( GL_POINT_SPRITE_ARB );
      self->addClassProperty( c_sdlglext, "COORD_REPLACE_ARB" ).setInteger( GL_COORD_REPLACE_ARB );
      #endif

      #ifndef GL_ARB_draw_buffers
      self->addClassProperty( c_sdlglext, "MAX_DRAW_BUFFERS_ARB" ).setInteger( GL_MAX_DRAW_BUFFERS_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER0_ARB" ).setInteger( GL_DRAW_BUFFER0_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER1_ARB" ).setInteger( GL_DRAW_BUFFER1_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER2_ARB" ).setInteger( GL_DRAW_BUFFER2_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER3_ARB" ).setInteger( GL_DRAW_BUFFER3_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER4_ARB" ).setInteger( GL_DRAW_BUFFER4_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER5_ARB" ).setInteger( GL_DRAW_BUFFER5_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER6_ARB" ).setInteger( GL_DRAW_BUFFER6_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER7_ARB" ).setInteger( GL_DRAW_BUFFER7_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER8_ARB" ).setInteger( GL_DRAW_BUFFER8_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER9_ARB" ).setInteger( GL_DRAW_BUFFER9_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER10_ARB" ).setInteger( GL_DRAW_BUFFER10_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER11_ARB" ).setInteger( GL_DRAW_BUFFER11_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER12_ARB" ).setInteger( GL_DRAW_BUFFER12_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER13_ARB" ).setInteger( GL_DRAW_BUFFER13_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER14_ARB" ).setInteger( GL_DRAW_BUFFER14_ARB );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER15_ARB" ).setInteger( GL_DRAW_BUFFER15_ARB );
      #endif

      #ifndef GL_ARB_texture_rectangle
      self->addClassProperty( c_sdlglext, "TEXTURE_RECTANGLE_ARB" ).setInteger( GL_TEXTURE_RECTANGLE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_BINDING_RECTANGLE_ARB" ).setInteger( GL_TEXTURE_BINDING_RECTANGLE_ARB );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_RECTANGLE_ARB" ).setInteger( GL_PROXY_TEXTURE_RECTANGLE_ARB );
      self->addClassProperty( c_sdlglext, "MAX_RECTANGLE_TEXTURE_SIZE_ARB" ).setInteger( GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB );
      #endif

      #ifndef GL_ARB_color_buffer_float
      self->addClassProperty( c_sdlglext, "RGBA_FLOAT_MODE_ARB" ).setInteger( GL_RGBA_FLOAT_MODE_ARB );
      self->addClassProperty( c_sdlglext, "CLAMP_VERTEX_COLOR_ARB" ).setInteger( GL_CLAMP_VERTEX_COLOR_ARB );
      self->addClassProperty( c_sdlglext, "CLAMP_FRAGMENT_COLOR_ARB" ).setInteger( GL_CLAMP_FRAGMENT_COLOR_ARB );
      self->addClassProperty( c_sdlglext, "CLAMP_READ_COLOR_ARB" ).setInteger( GL_CLAMP_READ_COLOR_ARB );
      self->addClassProperty( c_sdlglext, "FIXED_ONLY_ARB" ).setInteger( GL_FIXED_ONLY_ARB );
      #endif

      #ifndef GL_ARB_half_float_pixel
      self->addClassProperty( c_sdlglext, "HALF_FLOAT_ARB" ).setInteger( GL_HALF_FLOAT_ARB );
      #endif

      #ifndef GL_ARB_texture_float
      self->addClassProperty( c_sdlglext, "TEXTURE_RED_TYPE_ARB" ).setInteger( GL_TEXTURE_RED_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_GREEN_TYPE_ARB" ).setInteger( GL_TEXTURE_GREEN_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_BLUE_TYPE_ARB" ).setInteger( GL_TEXTURE_BLUE_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_ALPHA_TYPE_ARB" ).setInteger( GL_TEXTURE_ALPHA_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_LUMINANCE_TYPE_ARB" ).setInteger( GL_TEXTURE_LUMINANCE_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_INTENSITY_TYPE_ARB" ).setInteger( GL_TEXTURE_INTENSITY_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "TEXTURE_DEPTH_TYPE_ARB" ).setInteger( GL_TEXTURE_DEPTH_TYPE_ARB );
      self->addClassProperty( c_sdlglext, "UNSIGNED_NORMALIZED_ARB" ).setInteger( GL_UNSIGNED_NORMALIZED_ARB );
      self->addClassProperty( c_sdlglext, "RGBA32F_ARB" ).setInteger( GL_RGBA32F_ARB );
      self->addClassProperty( c_sdlglext, "RGB32F_ARB" ).setInteger( GL_RGB32F_ARB );
      self->addClassProperty( c_sdlglext, "ALPHA32F_ARB" ).setInteger( GL_ALPHA32F_ARB );
      self->addClassProperty( c_sdlglext, "INTENSITY32F_ARB" ).setInteger( GL_INTENSITY32F_ARB );
      self->addClassProperty( c_sdlglext, "LUMINANCE32F_ARB" ).setInteger( GL_LUMINANCE32F_ARB );
      self->addClassProperty( c_sdlglext, "LUMINANCE_ALPHA32F_ARB" ).setInteger( GL_LUMINANCE_ALPHA32F_ARB );
      self->addClassProperty( c_sdlglext, "RGBA16F_ARB" ).setInteger( GL_RGBA16F_ARB );
      self->addClassProperty( c_sdlglext, "RGB16F_ARB" ).setInteger( GL_RGB16F_ARB );
      self->addClassProperty( c_sdlglext, "ALPHA16F_ARB" ).setInteger( GL_ALPHA16F_ARB );
      self->addClassProperty( c_sdlglext, "INTENSITY16F_ARB" ).setInteger( GL_INTENSITY16F_ARB );
      self->addClassProperty( c_sdlglext, "LUMINANCE16F_ARB" ).setInteger( GL_LUMINANCE16F_ARB );
      self->addClassProperty( c_sdlglext, "LUMINANCE_ALPHA16F_ARB" ).setInteger( GL_LUMINANCE_ALPHA16F_ARB );
      #endif

      #ifndef GL_ARB_pixel_buffer_object
      self->addClassProperty( c_sdlglext, "PIXEL_PACK_BUFFER_ARB" ).setInteger( GL_PIXEL_PACK_BUFFER_ARB );
      self->addClassProperty( c_sdlglext, "PIXEL_UNPACK_BUFFER_ARB" ).setInteger( GL_PIXEL_UNPACK_BUFFER_ARB );
      self->addClassProperty( c_sdlglext, "PIXEL_PACK_BUFFER_BINDING_ARB" ).setInteger( GL_PIXEL_PACK_BUFFER_BINDING_ARB );
      self->addClassProperty( c_sdlglext, "PIXEL_UNPACK_BUFFER_BINDING_ARB" ).setInteger( GL_PIXEL_UNPACK_BUFFER_BINDING_ARB );
      #endif

      #ifndef GL_EXT_abgr
      self->addClassProperty( c_sdlglext, "ABGR_EXT" ).setInteger( GL_ABGR_EXT );
      #endif

      #ifndef GL_EXT_blend_color
      self->addClassProperty( c_sdlglext, "CONSTANT_COLOR_EXT" ).setInteger( GL_CONSTANT_COLOR_EXT );
      self->addClassProperty( c_sdlglext, "ONE_MINUS_CONSTANT_COLOR_EXT" ).setInteger( GL_ONE_MINUS_CONSTANT_COLOR_EXT );
      self->addClassProperty( c_sdlglext, "CONSTANT_ALPHA_EXT" ).setInteger( GL_CONSTANT_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "ONE_MINUS_CONSTANT_ALPHA_EXT" ).setInteger( GL_ONE_MINUS_CONSTANT_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "BLEND_COLOR_EXT" ).setInteger( GL_BLEND_COLOR_EXT );
      #endif

      #ifndef GL_EXT_polygon_offset
      self->addClassProperty( c_sdlglext, "POLYGON_OFFSET_EXT" ).setInteger( GL_POLYGON_OFFSET_EXT );
      self->addClassProperty( c_sdlglext, "POLYGON_OFFSET_FACTOR_EXT" ).setInteger( GL_POLYGON_OFFSET_FACTOR_EXT );
      self->addClassProperty( c_sdlglext, "POLYGON_OFFSET_BIAS_EXT" ).setInteger( GL_POLYGON_OFFSET_BIAS_EXT );
      #endif

      #ifndef GL_EXT_texture
      self->addClassProperty( c_sdlglext, "ALPHA4_EXT" ).setInteger( GL_ALPHA4_EXT );
      self->addClassProperty( c_sdlglext, "ALPHA8_EXT" ).setInteger( GL_ALPHA8_EXT );
      self->addClassProperty( c_sdlglext, "ALPHA12_EXT" ).setInteger( GL_ALPHA12_EXT );
      self->addClassProperty( c_sdlglext, "ALPHA16_EXT" ).setInteger( GL_ALPHA16_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE4_EXT" ).setInteger( GL_LUMINANCE4_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE8_EXT" ).setInteger( GL_LUMINANCE8_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE12_EXT" ).setInteger( GL_LUMINANCE12_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE16_EXT" ).setInteger( GL_LUMINANCE16_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE4_ALPHA4_EXT" ).setInteger( GL_LUMINANCE4_ALPHA4_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE6_ALPHA2_EXT" ).setInteger( GL_LUMINANCE6_ALPHA2_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE8_ALPHA8_EXT" ).setInteger( GL_LUMINANCE8_ALPHA8_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE12_ALPHA4_EXT" ).setInteger( GL_LUMINANCE12_ALPHA4_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE12_ALPHA12_EXT" ).setInteger( GL_LUMINANCE12_ALPHA12_EXT );
      self->addClassProperty( c_sdlglext, "LUMINANCE16_ALPHA16_EXT" ).setInteger( GL_LUMINANCE16_ALPHA16_EXT );
      self->addClassProperty( c_sdlglext, "INTENSITY_EXT" ).setInteger( GL_INTENSITY_EXT );
      self->addClassProperty( c_sdlglext, "INTENSITY4_EXT" ).setInteger( GL_INTENSITY4_EXT );
      self->addClassProperty( c_sdlglext, "INTENSITY8_EXT" ).setInteger( GL_INTENSITY8_EXT );
      self->addClassProperty( c_sdlglext, "INTENSITY12_EXT" ).setInteger( GL_INTENSITY12_EXT );
      self->addClassProperty( c_sdlglext, "INTENSITY16_EXT" ).setInteger( GL_INTENSITY16_EXT );
      self->addClassProperty( c_sdlglext, "RGB2_EXT" ).setInteger( GL_RGB2_EXT );
      self->addClassProperty( c_sdlglext, "RGB4_EXT" ).setInteger( GL_RGB4_EXT );
      self->addClassProperty( c_sdlglext, "RGB5_EXT" ).setInteger( GL_RGB5_EXT );
      self->addClassProperty( c_sdlglext, "RGB8_EXT" ).setInteger( GL_RGB8_EXT );
      self->addClassProperty( c_sdlglext, "RGB10_EXT" ).setInteger( GL_RGB10_EXT );
      self->addClassProperty( c_sdlglext, "RGB12_EXT" ).setInteger( GL_RGB12_EXT );
      self->addClassProperty( c_sdlglext, "RGB16_EXT" ).setInteger( GL_RGB16_EXT );
      self->addClassProperty( c_sdlglext, "RGBA2_EXT" ).setInteger( GL_RGBA2_EXT );
      self->addClassProperty( c_sdlglext, "RGBA4_EXT" ).setInteger( GL_RGBA4_EXT );
      self->addClassProperty( c_sdlglext, "RGB5_A1_EXT" ).setInteger( GL_RGB5_A1_EXT );
      self->addClassProperty( c_sdlglext, "RGBA8_EXT" ).setInteger( GL_RGBA8_EXT );
      self->addClassProperty( c_sdlglext, "RGB10_A2_EXT" ).setInteger( GL_RGB10_A2_EXT );
      self->addClassProperty( c_sdlglext, "RGBA12_EXT" ).setInteger( GL_RGBA12_EXT );
      self->addClassProperty( c_sdlglext, "RGBA16_EXT" ).setInteger( GL_RGBA16_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_RED_SIZE_EXT" ).setInteger( GL_TEXTURE_RED_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_GREEN_SIZE_EXT" ).setInteger( GL_TEXTURE_GREEN_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_BLUE_SIZE_EXT" ).setInteger( GL_TEXTURE_BLUE_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_ALPHA_SIZE_EXT" ).setInteger( GL_TEXTURE_ALPHA_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_LUMINANCE_SIZE_EXT" ).setInteger( GL_TEXTURE_LUMINANCE_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_INTENSITY_SIZE_EXT" ).setInteger( GL_TEXTURE_INTENSITY_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "REPLACE_EXT" ).setInteger( GL_REPLACE_EXT );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_1D_EXT" ).setInteger( GL_PROXY_TEXTURE_1D_EXT );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_2D_EXT" ).setInteger( GL_PROXY_TEXTURE_2D_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_TOO_LARGE_EXT" ).setInteger( GL_TEXTURE_TOO_LARGE_EXT );
      #endif

      #ifndef GL_EXT_texture3D
      self->addClassProperty( c_sdlglext, "PACK_SKIP_IMAGES_EXT" ).setInteger( GL_PACK_SKIP_IMAGES_EXT );
      self->addClassProperty( c_sdlglext, "PACK_IMAGE_HEIGHT_EXT" ).setInteger( GL_PACK_IMAGE_HEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "UNPACK_SKIP_IMAGES_EXT" ).setInteger( GL_UNPACK_SKIP_IMAGES_EXT );
      self->addClassProperty( c_sdlglext, "UNPACK_IMAGE_HEIGHT_EXT" ).setInteger( GL_UNPACK_IMAGE_HEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_3D_EXT" ).setInteger( GL_TEXTURE_3D_EXT );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_3D_EXT" ).setInteger( GL_PROXY_TEXTURE_3D_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_DEPTH_EXT" ).setInteger( GL_TEXTURE_DEPTH_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_WRAP_R_EXT" ).setInteger( GL_TEXTURE_WRAP_R_EXT );
      self->addClassProperty( c_sdlglext, "MAX_3D_TEXTURE_SIZE_EXT" ).setInteger( GL_MAX_3D_TEXTURE_SIZE_EXT );
      #endif

      #ifndef GL_SGIS_texture_filter4
      self->addClassProperty( c_sdlglext, "FILTER4_SGIS" ).setInteger( GL_FILTER4_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_FILTER4_SIZE_SGIS" ).setInteger( GL_TEXTURE_FILTER4_SIZE_SGIS );
      #endif

      #ifndef GL_EXT_histogram
      self->addClassProperty( c_sdlglext, "HISTOGRAM_EXT" ).setInteger( GL_HISTOGRAM_EXT );
      self->addClassProperty( c_sdlglext, "PROXY_HISTOGRAM_EXT" ).setInteger( GL_PROXY_HISTOGRAM_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_WIDTH_EXT" ).setInteger( GL_HISTOGRAM_WIDTH_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_FORMAT_EXT" ).setInteger( GL_HISTOGRAM_FORMAT_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_RED_SIZE_EXT" ).setInteger( GL_HISTOGRAM_RED_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_GREEN_SIZE_EXT" ).setInteger( GL_HISTOGRAM_GREEN_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_BLUE_SIZE_EXT" ).setInteger( GL_HISTOGRAM_BLUE_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_ALPHA_SIZE_EXT" ).setInteger( GL_HISTOGRAM_ALPHA_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_LUMINANCE_SIZE_EXT" ).setInteger( GL_HISTOGRAM_LUMINANCE_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "HISTOGRAM_SINK_EXT" ).setInteger( GL_HISTOGRAM_SINK_EXT );
      self->addClassProperty( c_sdlglext, "MINMAX_EXT" ).setInteger( GL_MINMAX_EXT );
      self->addClassProperty( c_sdlglext, "MINMAX_FORMAT_EXT" ).setInteger( GL_MINMAX_FORMAT_EXT );
      self->addClassProperty( c_sdlglext, "MINMAX_SINK_EXT" ).setInteger( GL_MINMAX_SINK_EXT );
      self->addClassProperty( c_sdlglext, "TABLE_TOO_LARGE_EXT" ).setInteger( GL_TABLE_TOO_LARGE_EXT );
      #endif

      #ifndef GL_EXT_convolution
      self->addClassProperty( c_sdlglext, "CONVOLUTION_1D_EXT" ).setInteger( GL_CONVOLUTION_1D_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_2D_EXT" ).setInteger( GL_CONVOLUTION_2D_EXT );
      self->addClassProperty( c_sdlglext, "SEPARABLE_2D_EXT" ).setInteger( GL_SEPARABLE_2D_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_BORDER_MODE_EXT" ).setInteger( GL_CONVOLUTION_BORDER_MODE_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_FILTER_SCALE_EXT" ).setInteger( GL_CONVOLUTION_FILTER_SCALE_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_FILTER_BIAS_EXT" ).setInteger( GL_CONVOLUTION_FILTER_BIAS_EXT );
      self->addClassProperty( c_sdlglext, "REDUCE_EXT" ).setInteger( GL_REDUCE_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_FORMAT_EXT" ).setInteger( GL_CONVOLUTION_FORMAT_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_WIDTH_EXT" ).setInteger( GL_CONVOLUTION_WIDTH_EXT );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_HEIGHT_EXT" ).setInteger( GL_CONVOLUTION_HEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "MAX_CONVOLUTION_WIDTH_EXT" ).setInteger( GL_MAX_CONVOLUTION_WIDTH_EXT );
      self->addClassProperty( c_sdlglext, "MAX_CONVOLUTION_HEIGHT_EXT" ).setInteger( GL_MAX_CONVOLUTION_HEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_RED_SCALE_EXT" ).setInteger( GL_POST_CONVOLUTION_RED_SCALE_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_GREEN_SCALE_EXT" ).setInteger( GL_POST_CONVOLUTION_GREEN_SCALE_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_BLUE_SCALE_EXT" ).setInteger( GL_POST_CONVOLUTION_BLUE_SCALE_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_ALPHA_SCALE_EXT" ).setInteger( GL_POST_CONVOLUTION_ALPHA_SCALE_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_RED_BIAS_EXT" ).setInteger( GL_POST_CONVOLUTION_RED_BIAS_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_GREEN_BIAS_EXT" ).setInteger( GL_POST_CONVOLUTION_GREEN_BIAS_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_BLUE_BIAS_EXT" ).setInteger( GL_POST_CONVOLUTION_BLUE_BIAS_EXT );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_ALPHA_BIAS_EXT" ).setInteger( GL_POST_CONVOLUTION_ALPHA_BIAS_EXT );
      #endif

      #ifndef GL_SGI_color_matrix
      self->addClassProperty( c_sdlglext, "COLOR_MATRIX_SGI" ).setInteger( GL_COLOR_MATRIX_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_MATRIX_STACK_DEPTH_SGI" ).setInteger( GL_COLOR_MATRIX_STACK_DEPTH_SGI );
      self->addClassProperty( c_sdlglext, "MAX_COLOR_MATRIX_STACK_DEPTH_SGI" ).setInteger( GL_MAX_COLOR_MATRIX_STACK_DEPTH_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_RED_SCALE_SGI" ).setInteger( GL_POST_COLOR_MATRIX_RED_SCALE_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_GREEN_SCALE_SGI" ).setInteger( GL_POST_COLOR_MATRIX_GREEN_SCALE_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_BLUE_SCALE_SGI" ).setInteger( GL_POST_COLOR_MATRIX_BLUE_SCALE_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_ALPHA_SCALE_SGI" ).setInteger( GL_POST_COLOR_MATRIX_ALPHA_SCALE_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_RED_BIAS_SGI" ).setInteger( GL_POST_COLOR_MATRIX_RED_BIAS_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_GREEN_BIAS_SGI" ).setInteger( GL_POST_COLOR_MATRIX_GREEN_BIAS_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_BLUE_BIAS_SGI" ).setInteger( GL_POST_COLOR_MATRIX_BLUE_BIAS_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_ALPHA_BIAS_SGI" ).setInteger( GL_POST_COLOR_MATRIX_ALPHA_BIAS_SGI );
      #endif

      #ifndef GL_SGI_color_table
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_SGI" ).setInteger( GL_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "POST_CONVOLUTION_COLOR_TABLE_SGI" ).setInteger( GL_POST_CONVOLUTION_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "POST_COLOR_MATRIX_COLOR_TABLE_SGI" ).setInteger( GL_POST_COLOR_MATRIX_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "PROXY_COLOR_TABLE_SGI" ).setInteger( GL_PROXY_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "PROXY_POST_CONVOLUTION_COLOR_TABLE_SGI" ).setInteger( GL_PROXY_POST_CONVOLUTION_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "PROXY_POST_COLOR_MATRIX_COLOR_TABLE_SGI" ).setInteger( GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_SCALE_SGI" ).setInteger( GL_COLOR_TABLE_SCALE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_BIAS_SGI" ).setInteger( GL_COLOR_TABLE_BIAS_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_FORMAT_SGI" ).setInteger( GL_COLOR_TABLE_FORMAT_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_WIDTH_SGI" ).setInteger( GL_COLOR_TABLE_WIDTH_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_RED_SIZE_SGI" ).setInteger( GL_COLOR_TABLE_RED_SIZE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_GREEN_SIZE_SGI" ).setInteger( GL_COLOR_TABLE_GREEN_SIZE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_BLUE_SIZE_SGI" ).setInteger( GL_COLOR_TABLE_BLUE_SIZE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_ALPHA_SIZE_SGI" ).setInteger( GL_COLOR_TABLE_ALPHA_SIZE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_LUMINANCE_SIZE_SGI" ).setInteger( GL_COLOR_TABLE_LUMINANCE_SIZE_SGI );
      self->addClassProperty( c_sdlglext, "COLOR_TABLE_INTENSITY_SIZE_SGI" ).setInteger( GL_COLOR_TABLE_INTENSITY_SIZE_SGI );
      #endif

      #ifndef GL_SGIS_pixel_texture
      self->addClassProperty( c_sdlglext, "PIXEL_TEXTURE_SGIS" ).setInteger( GL_PIXEL_TEXTURE_SGIS );
      self->addClassProperty( c_sdlglext, "PIXEL_FRAGMENT_RGB_SOURCE_SGIS" ).setInteger( GL_PIXEL_FRAGMENT_RGB_SOURCE_SGIS );
      self->addClassProperty( c_sdlglext, "PIXEL_FRAGMENT_ALPHA_SOURCE_SGIS" ).setInteger( GL_PIXEL_FRAGMENT_ALPHA_SOURCE_SGIS );
      self->addClassProperty( c_sdlglext, "PIXEL_GROUP_COLOR_SGIS" ).setInteger( GL_PIXEL_GROUP_COLOR_SGIS );
      #endif

      #ifndef GL_SGIX_pixel_texture
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_MODE_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_MODE_SGIX );
      #endif

      #ifndef GL_SGIS_texture4D
      self->addClassProperty( c_sdlglext, "PACK_SKIP_VOLUMES_SGIS" ).setInteger( GL_PACK_SKIP_VOLUMES_SGIS );
      self->addClassProperty( c_sdlglext, "PACK_IMAGE_DEPTH_SGIS" ).setInteger( GL_PACK_IMAGE_DEPTH_SGIS );
      self->addClassProperty( c_sdlglext, "UNPACK_SKIP_VOLUMES_SGIS" ).setInteger( GL_UNPACK_SKIP_VOLUMES_SGIS );
      self->addClassProperty( c_sdlglext, "UNPACK_IMAGE_DEPTH_SGIS" ).setInteger( GL_UNPACK_IMAGE_DEPTH_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_4D_SGIS" ).setInteger( GL_TEXTURE_4D_SGIS );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_4D_SGIS" ).setInteger( GL_PROXY_TEXTURE_4D_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_4DSIZE_SGIS" ).setInteger( GL_TEXTURE_4DSIZE_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_WRAP_Q_SGIS" ).setInteger( GL_TEXTURE_WRAP_Q_SGIS );
      self->addClassProperty( c_sdlglext, "MAX_4D_TEXTURE_SIZE_SGIS" ).setInteger( GL_MAX_4D_TEXTURE_SIZE_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_4D_BINDING_SGIS" ).setInteger( GL_TEXTURE_4D_BINDING_SGIS );
      #endif

      #ifndef GL_SGI_texture_color_table
      self->addClassProperty( c_sdlglext, "TEXTURE_COLOR_TABLE_SGI" ).setInteger( GL_TEXTURE_COLOR_TABLE_SGI );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_COLOR_TABLE_SGI" ).setInteger( GL_PROXY_TEXTURE_COLOR_TABLE_SGI );
      #endif

      #ifndef GL_EXT_cmyka
      self->addClassProperty( c_sdlglext, "CMYK_EXT" ).setInteger( GL_CMYK_EXT );
      self->addClassProperty( c_sdlglext, "CMYKA_EXT" ).setInteger( GL_CMYKA_EXT );
      self->addClassProperty( c_sdlglext, "PACK_CMYK_HINT_EXT" ).setInteger( GL_PACK_CMYK_HINT_EXT );
      self->addClassProperty( c_sdlglext, "UNPACK_CMYK_HINT_EXT" ).setInteger( GL_UNPACK_CMYK_HINT_EXT );
      #endif

      #ifndef GL_EXT_texture_object
      self->addClassProperty( c_sdlglext, "TEXTURE_PRIORITY_EXT" ).setInteger( GL_TEXTURE_PRIORITY_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_RESIDENT_EXT" ).setInteger( GL_TEXTURE_RESIDENT_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_1D_BINDING_EXT" ).setInteger( GL_TEXTURE_1D_BINDING_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_2D_BINDING_EXT" ).setInteger( GL_TEXTURE_2D_BINDING_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_3D_BINDING_EXT" ).setInteger( GL_TEXTURE_3D_BINDING_EXT );
      #endif

      #ifndef GL_SGIS_detail_texture
      self->addClassProperty( c_sdlglext, "DETAIL_TEXTURE_2D_SGIS" ).setInteger( GL_DETAIL_TEXTURE_2D_SGIS );
      self->addClassProperty( c_sdlglext, "DETAIL_TEXTURE_2D_BINDING_SGIS" ).setInteger( GL_DETAIL_TEXTURE_2D_BINDING_SGIS );
      self->addClassProperty( c_sdlglext, "LINEAR_DETAIL_SGIS" ).setInteger( GL_LINEAR_DETAIL_SGIS );
      self->addClassProperty( c_sdlglext, "LINEAR_DETAIL_ALPHA_SGIS" ).setInteger( GL_LINEAR_DETAIL_ALPHA_SGIS );
      self->addClassProperty( c_sdlglext, "LINEAR_DETAIL_COLOR_SGIS" ).setInteger( GL_LINEAR_DETAIL_COLOR_SGIS );
      self->addClassProperty( c_sdlglext, "DETAIL_TEXTURE_LEVEL_SGIS" ).setInteger( GL_DETAIL_TEXTURE_LEVEL_SGIS );
      self->addClassProperty( c_sdlglext, "DETAIL_TEXTURE_MODE_SGIS" ).setInteger( GL_DETAIL_TEXTURE_MODE_SGIS );
      self->addClassProperty( c_sdlglext, "DETAIL_TEXTURE_FUNC_POINTS_SGIS" ).setInteger( GL_DETAIL_TEXTURE_FUNC_POINTS_SGIS );
      #endif

      #ifndef GL_SGIS_sharpen_texture
      self->addClassProperty( c_sdlglext, "LINEAR_SHARPEN_SGIS" ).setInteger( GL_LINEAR_SHARPEN_SGIS );
      self->addClassProperty( c_sdlglext, "LINEAR_SHARPEN_ALPHA_SGIS" ).setInteger( GL_LINEAR_SHARPEN_ALPHA_SGIS );
      self->addClassProperty( c_sdlglext, "LINEAR_SHARPEN_COLOR_SGIS" ).setInteger( GL_LINEAR_SHARPEN_COLOR_SGIS );
      self->addClassProperty( c_sdlglext, "SHARPEN_TEXTURE_FUNC_POINTS_SGIS" ).setInteger( GL_SHARPEN_TEXTURE_FUNC_POINTS_SGIS );
      #endif

      #ifndef GL_EXT_packed_pixels
      self->addClassProperty( c_sdlglext, "UNSIGNED_BYTE_3_3_2_EXT" ).setInteger( GL_UNSIGNED_BYTE_3_3_2_EXT );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_4_4_4_4_EXT" ).setInteger( GL_UNSIGNED_SHORT_4_4_4_4_EXT );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_5_5_5_1_EXT" ).setInteger( GL_UNSIGNED_SHORT_5_5_5_1_EXT );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_8_8_8_8_EXT" ).setInteger( GL_UNSIGNED_INT_8_8_8_8_EXT );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_10_10_10_2_EXT" ).setInteger( GL_UNSIGNED_INT_10_10_10_2_EXT );
      #endif

      #ifndef GL_SGIS_texture_lod
      self->addClassProperty( c_sdlglext, "TEXTURE_MIN_LOD_SGIS" ).setInteger( GL_TEXTURE_MIN_LOD_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_LOD_SGIS" ).setInteger( GL_TEXTURE_MAX_LOD_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_BASE_LEVEL_SGIS" ).setInteger( GL_TEXTURE_BASE_LEVEL_SGIS );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_LEVEL_SGIS" ).setInteger( GL_TEXTURE_MAX_LEVEL_SGIS );
      #endif

      #ifndef GL_SGIS_multisample
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_SGIS" ).setInteger( GL_MULTISAMPLE_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_MASK_SGIS" ).setInteger( GL_SAMPLE_ALPHA_TO_MASK_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_ONE_SGIS" ).setInteger( GL_SAMPLE_ALPHA_TO_ONE_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_MASK_SGIS" ).setInteger( GL_SAMPLE_MASK_SGIS );
      self->addClassProperty( c_sdlglext, "1PASS_SGIS" ).setInteger( GL_1PASS_SGIS );
      self->addClassProperty( c_sdlglext, "2PASS_0_SGIS" ).setInteger( GL_2PASS_0_SGIS );
      self->addClassProperty( c_sdlglext, "2PASS_1_SGIS" ).setInteger( GL_2PASS_1_SGIS );
      self->addClassProperty( c_sdlglext, "4PASS_0_SGIS" ).setInteger( GL_4PASS_0_SGIS );
      self->addClassProperty( c_sdlglext, "4PASS_1_SGIS" ).setInteger( GL_4PASS_1_SGIS );
      self->addClassProperty( c_sdlglext, "4PASS_2_SGIS" ).setInteger( GL_4PASS_2_SGIS );
      self->addClassProperty( c_sdlglext, "4PASS_3_SGIS" ).setInteger( GL_4PASS_3_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_BUFFERS_SGIS" ).setInteger( GL_SAMPLE_BUFFERS_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLES_SGIS" ).setInteger( GL_SAMPLES_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_MASK_VALUE_SGIS" ).setInteger( GL_SAMPLE_MASK_VALUE_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_MASK_INVERT_SGIS" ).setInteger( GL_SAMPLE_MASK_INVERT_SGIS );
      self->addClassProperty( c_sdlglext, "SAMPLE_PATTERN_SGIS" ).setInteger( GL_SAMPLE_PATTERN_SGIS );
      #endif

      #ifndef GL_EXT_rescale_normal
      self->addClassProperty( c_sdlglext, "RESCALE_NORMAL_EXT" ).setInteger( GL_RESCALE_NORMAL_EXT );
      #endif

      #ifndef GL_EXT_vertex_array
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_EXT" ).setInteger( GL_VERTEX_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_EXT" ).setInteger( GL_NORMAL_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_EXT" ).setInteger( GL_COLOR_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_EXT" ).setInteger( GL_INDEX_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "EDGE_FLAG_ARRAY_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_SIZE_EXT" ).setInteger( GL_VERTEX_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_TYPE_EXT" ).setInteger( GL_VERTEX_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_STRIDE_EXT" ).setInteger( GL_VERTEX_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_COUNT_EXT" ).setInteger( GL_VERTEX_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_TYPE_EXT" ).setInteger( GL_NORMAL_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_STRIDE_EXT" ).setInteger( GL_NORMAL_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_COUNT_EXT" ).setInteger( GL_NORMAL_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_SIZE_EXT" ).setInteger( GL_COLOR_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_TYPE_EXT" ).setInteger( GL_COLOR_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_STRIDE_EXT" ).setInteger( GL_COLOR_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_COUNT_EXT" ).setInteger( GL_COLOR_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_TYPE_EXT" ).setInteger( GL_INDEX_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_STRIDE_EXT" ).setInteger( GL_INDEX_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_COUNT_EXT" ).setInteger( GL_INDEX_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_SIZE_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_TYPE_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_STRIDE_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_COUNT_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlglext, "EDGE_FLAG_ARRAY_STRIDE_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "EDGE_FLAG_ARRAY_COUNT_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_POINTER_EXT" ).setInteger( GL_VERTEX_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_POINTER_EXT" ).setInteger( GL_NORMAL_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_POINTER_EXT" ).setInteger( GL_COLOR_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_ARRAY_POINTER_EXT" ).setInteger( GL_INDEX_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_POINTER_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "EDGE_FLAG_ARRAY_POINTER_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_POINTER_EXT );
      #endif

      #ifndef GL_SGIS_generate_mipmap
      self->addClassProperty( c_sdlglext, "GENERATE_MIPMAP_SGIS" ).setInteger( GL_GENERATE_MIPMAP_SGIS );
      self->addClassProperty( c_sdlglext, "GENERATE_MIPMAP_HINT_SGIS" ).setInteger( GL_GENERATE_MIPMAP_HINT_SGIS );
      #endif

      #ifndef GL_SGIX_clipmap
      self->addClassProperty( c_sdlglext, "LINEAR_CLIPMAP_LINEAR_SGIX" ).setInteger( GL_LINEAR_CLIPMAP_LINEAR_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CLIPMAP_CENTER_SGIX" ).setInteger( GL_TEXTURE_CLIPMAP_CENTER_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CLIPMAP_FRAME_SGIX" ).setInteger( GL_TEXTURE_CLIPMAP_FRAME_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CLIPMAP_OFFSET_SGIX" ).setInteger( GL_TEXTURE_CLIPMAP_OFFSET_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CLIPMAP_VIRTUAL_DEPTH_SGIX" ).setInteger( GL_TEXTURE_CLIPMAP_VIRTUAL_DEPTH_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CLIPMAP_LOD_OFFSET_SGIX" ).setInteger( GL_TEXTURE_CLIPMAP_LOD_OFFSET_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CLIPMAP_DEPTH_SGIX" ).setInteger( GL_TEXTURE_CLIPMAP_DEPTH_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_CLIPMAP_DEPTH_SGIX" ).setInteger( GL_MAX_CLIPMAP_DEPTH_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_CLIPMAP_VIRTUAL_DEPTH_SGIX" ).setInteger( GL_MAX_CLIPMAP_VIRTUAL_DEPTH_SGIX );
      self->addClassProperty( c_sdlglext, "NEAREST_CLIPMAP_NEAREST_SGIX" ).setInteger( GL_NEAREST_CLIPMAP_NEAREST_SGIX );
      self->addClassProperty( c_sdlglext, "NEAREST_CLIPMAP_LINEAR_SGIX" ).setInteger( GL_NEAREST_CLIPMAP_LINEAR_SGIX );
      self->addClassProperty( c_sdlglext, "LINEAR_CLIPMAP_NEAREST_SGIX" ).setInteger( GL_LINEAR_CLIPMAP_NEAREST_SGIX );
      #endif

      #ifndef GL_SGIX_shadow
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_SGIX" ).setInteger( GL_TEXTURE_COMPARE_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_COMPARE_OPERATOR_SGIX" ).setInteger( GL_TEXTURE_COMPARE_OPERATOR_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_LEQUAL_R_SGIX" ).setInteger( GL_TEXTURE_LEQUAL_R_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_GEQUAL_R_SGIX" ).setInteger( GL_TEXTURE_GEQUAL_R_SGIX );
      #endif

      #ifndef GL_SGIS_texture_edge_clamp
      self->addClassProperty( c_sdlglext, "CLAMP_TO_EDGE_SGIS" ).setInteger( GL_CLAMP_TO_EDGE_SGIS );
      #endif

      #ifndef GL_SGIS_texture_border_clamp
      self->addClassProperty( c_sdlglext, "CLAMP_TO_BORDER_SGIS" ).setInteger( GL_CLAMP_TO_BORDER_SGIS );
      #endif

      #ifndef GL_EXT_blend_minmax
      self->addClassProperty( c_sdlglext, "FUNC_ADD_EXT" ).setInteger( GL_FUNC_ADD_EXT );
      self->addClassProperty( c_sdlglext, "MIN_EXT" ).setInteger( GL_MIN_EXT );
      self->addClassProperty( c_sdlglext, "MAX_EXT" ).setInteger( GL_MAX_EXT );
      self->addClassProperty( c_sdlglext, "BLEND_EQUATION_EXT" ).setInteger( GL_BLEND_EQUATION_EXT );
      #endif

      #ifndef GL_EXT_blend_subtract
      self->addClassProperty( c_sdlglext, "FUNC_SUBTRACT_EXT" ).setInteger( GL_FUNC_SUBTRACT_EXT );
      self->addClassProperty( c_sdlglext, "FUNC_REVERSE_SUBTRACT_EXT" ).setInteger( GL_FUNC_REVERSE_SUBTRACT_EXT );
      #endif

      #ifndef GL_SGIX_interlace
      self->addClassProperty( c_sdlglext, "INTERLACE_SGIX" ).setInteger( GL_INTERLACE_SGIX );
      #endif

      #ifndef GL_SGIX_pixel_tiles
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_BEST_ALIGNMENT_SGIX" ).setInteger( GL_PIXEL_TILE_BEST_ALIGNMENT_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_CACHE_INCREMENT_SGIX" ).setInteger( GL_PIXEL_TILE_CACHE_INCREMENT_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_WIDTH_SGIX" ).setInteger( GL_PIXEL_TILE_WIDTH_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_HEIGHT_SGIX" ).setInteger( GL_PIXEL_TILE_HEIGHT_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_GRID_WIDTH_SGIX" ).setInteger( GL_PIXEL_TILE_GRID_WIDTH_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_GRID_HEIGHT_SGIX" ).setInteger( GL_PIXEL_TILE_GRID_HEIGHT_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_GRID_DEPTH_SGIX" ).setInteger( GL_PIXEL_TILE_GRID_DEPTH_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TILE_CACHE_SIZE_SGIX" ).setInteger( GL_PIXEL_TILE_CACHE_SIZE_SGIX );
      #endif

      #ifndef GL_SGIS_texture_select
      self->addClassProperty( c_sdlglext, "DUAL_ALPHA4_SGIS" ).setInteger( GL_DUAL_ALPHA4_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_ALPHA8_SGIS" ).setInteger( GL_DUAL_ALPHA8_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_ALPHA12_SGIS" ).setInteger( GL_DUAL_ALPHA12_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_ALPHA16_SGIS" ).setInteger( GL_DUAL_ALPHA16_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_LUMINANCE4_SGIS" ).setInteger( GL_DUAL_LUMINANCE4_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_LUMINANCE8_SGIS" ).setInteger( GL_DUAL_LUMINANCE8_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_LUMINANCE12_SGIS" ).setInteger( GL_DUAL_LUMINANCE12_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_LUMINANCE16_SGIS" ).setInteger( GL_DUAL_LUMINANCE16_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_INTENSITY4_SGIS" ).setInteger( GL_DUAL_INTENSITY4_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_INTENSITY8_SGIS" ).setInteger( GL_DUAL_INTENSITY8_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_INTENSITY12_SGIS" ).setInteger( GL_DUAL_INTENSITY12_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_INTENSITY16_SGIS" ).setInteger( GL_DUAL_INTENSITY16_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_LUMINANCE_ALPHA4_SGIS" ).setInteger( GL_DUAL_LUMINANCE_ALPHA4_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_LUMINANCE_ALPHA8_SGIS" ).setInteger( GL_DUAL_LUMINANCE_ALPHA8_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_ALPHA4_SGIS" ).setInteger( GL_QUAD_ALPHA4_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_ALPHA8_SGIS" ).setInteger( GL_QUAD_ALPHA8_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_LUMINANCE4_SGIS" ).setInteger( GL_QUAD_LUMINANCE4_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_LUMINANCE8_SGIS" ).setInteger( GL_QUAD_LUMINANCE8_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_INTENSITY4_SGIS" ).setInteger( GL_QUAD_INTENSITY4_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_INTENSITY8_SGIS" ).setInteger( GL_QUAD_INTENSITY8_SGIS );
      self->addClassProperty( c_sdlglext, "DUAL_TEXTURE_SELECT_SGIS" ).setInteger( GL_DUAL_TEXTURE_SELECT_SGIS );
      self->addClassProperty( c_sdlglext, "QUAD_TEXTURE_SELECT_SGIS" ).setInteger( GL_QUAD_TEXTURE_SELECT_SGIS );
      #endif

      #ifndef GL_SGIX_sprite
      self->addClassProperty( c_sdlglext, "SPRITE_SGIX" ).setInteger( GL_SPRITE_SGIX );
      self->addClassProperty( c_sdlglext, "SPRITE_MODE_SGIX" ).setInteger( GL_SPRITE_MODE_SGIX );
      self->addClassProperty( c_sdlglext, "SPRITE_AXIS_SGIX" ).setInteger( GL_SPRITE_AXIS_SGIX );
      self->addClassProperty( c_sdlglext, "SPRITE_TRANSLATION_SGIX" ).setInteger( GL_SPRITE_TRANSLATION_SGIX );
      self->addClassProperty( c_sdlglext, "SPRITE_AXIAL_SGIX" ).setInteger( GL_SPRITE_AXIAL_SGIX );
      self->addClassProperty( c_sdlglext, "SPRITE_OBJECT_ALIGNED_SGIX" ).setInteger( GL_SPRITE_OBJECT_ALIGNED_SGIX );
      self->addClassProperty( c_sdlglext, "SPRITE_EYE_ALIGNED_SGIX" ).setInteger( GL_SPRITE_EYE_ALIGNED_SGIX );
      #endif

      #ifndef GL_SGIX_texture_multi_buffer
      self->addClassProperty( c_sdlglext, "TEXTURE_MULTI_BUFFER_HINT_SGIX" ).setInteger( GL_TEXTURE_MULTI_BUFFER_HINT_SGIX );
      #endif

      #ifndef GL_EXT_point_parameters
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MIN_EXT" ).setInteger( GL_POINT_SIZE_MIN_EXT );
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MAX_EXT" ).setInteger( GL_POINT_SIZE_MAX_EXT );
      self->addClassProperty( c_sdlglext, "POINT_FADE_THRESHOLD_SIZE_EXT" ).setInteger( GL_POINT_FADE_THRESHOLD_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "DISTANCE_ATTENUATION_EXT" ).setInteger( GL_DISTANCE_ATTENUATION_EXT );
      #endif

      #ifndef GL_SGIS_point_parameters
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MIN_SGIS" ).setInteger( GL_POINT_SIZE_MIN_SGIS );
      self->addClassProperty( c_sdlglext, "POINT_SIZE_MAX_SGIS" ).setInteger( GL_POINT_SIZE_MAX_SGIS );
      self->addClassProperty( c_sdlglext, "POINT_FADE_THRESHOLD_SIZE_SGIS" ).setInteger( GL_POINT_FADE_THRESHOLD_SIZE_SGIS );
      self->addClassProperty( c_sdlglext, "DISTANCE_ATTENUATION_SGIS" ).setInteger( GL_DISTANCE_ATTENUATION_SGIS );
      #endif

      #ifndef GL_SGIX_instruments
      self->addClassProperty( c_sdlglext, "INSTRUMENT_BUFFER_POINTER_SGIX" ).setInteger( GL_INSTRUMENT_BUFFER_POINTER_SGIX );
      self->addClassProperty( c_sdlglext, "INSTRUMENT_MEASUREMENTS_SGIX" ).setInteger( GL_INSTRUMENT_MEASUREMENTS_SGIX );
      #endif

      #ifndef GL_SGIX_texture_scale_bias
      self->addClassProperty( c_sdlglext, "POST_TEXTURE_FILTER_BIAS_SGIX" ).setInteger( GL_POST_TEXTURE_FILTER_BIAS_SGIX );
      self->addClassProperty( c_sdlglext, "POST_TEXTURE_FILTER_SCALE_SGIX" ).setInteger( GL_POST_TEXTURE_FILTER_SCALE_SGIX );
      self->addClassProperty( c_sdlglext, "POST_TEXTURE_FILTER_BIAS_RANGE_SGIX" ).setInteger( GL_POST_TEXTURE_FILTER_BIAS_RANGE_SGIX );
      self->addClassProperty( c_sdlglext, "POST_TEXTURE_FILTER_SCALE_RANGE_SGIX" ).setInteger( GL_POST_TEXTURE_FILTER_SCALE_RANGE_SGIX );
      #endif

      #ifndef GL_SGIX_framezoom
      self->addClassProperty( c_sdlglext, "FRAMEZOOM_SGIX" ).setInteger( GL_FRAMEZOOM_SGIX );
      self->addClassProperty( c_sdlglext, "FRAMEZOOM_FACTOR_SGIX" ).setInteger( GL_FRAMEZOOM_FACTOR_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_FRAMEZOOM_FACTOR_SGIX" ).setInteger( GL_MAX_FRAMEZOOM_FACTOR_SGIX );
      #endif

      #ifndef GL_FfdMaskSGIX
      self->addClassProperty( c_sdlglext, "TEXTURE_DEFORMATION_BIT_SGIX" ).setInteger( GL_TEXTURE_DEFORMATION_BIT_SGIX );
      self->addClassProperty( c_sdlglext, "GEOMETRY_DEFORMATION_BIT_SGIX" ).setInteger( GL_GEOMETRY_DEFORMATION_BIT_SGIX );
      #endif

      #ifndef GL_SGIX_polynomial_ffd
      self->addClassProperty( c_sdlglext, "GEOMETRY_DEFORMATION_SGIX" ).setInteger( GL_GEOMETRY_DEFORMATION_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_DEFORMATION_SGIX" ).setInteger( GL_TEXTURE_DEFORMATION_SGIX );
      self->addClassProperty( c_sdlglext, "DEFORMATIONS_MASK_SGIX" ).setInteger( GL_DEFORMATIONS_MASK_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_DEFORMATION_ORDER_SGIX" ).setInteger( GL_MAX_DEFORMATION_ORDER_SGIX );
      #endif

      #ifndef GL_SGIX_reference_plane
      self->addClassProperty( c_sdlglext, "REFERENCE_PLANE_SGIX" ).setInteger( GL_REFERENCE_PLANE_SGIX );
      self->addClassProperty( c_sdlglext, "REFERENCE_PLANE_EQUATION_SGIX" ).setInteger( GL_REFERENCE_PLANE_EQUATION_SGIX );
      #endif

      #ifndef GL_SGIX_depth_texture
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT16_SGIX" ).setInteger( GL_DEPTH_COMPONENT16_SGIX );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT24_SGIX" ).setInteger( GL_DEPTH_COMPONENT24_SGIX );
      self->addClassProperty( c_sdlglext, "DEPTH_COMPONENT32_SGIX" ).setInteger( GL_DEPTH_COMPONENT32_SGIX );
      #endif

      #ifndef GL_SGIS_fog_function
      self->addClassProperty( c_sdlglext, "FOG_FUNC_SGIS" ).setInteger( GL_FOG_FUNC_SGIS );
      self->addClassProperty( c_sdlglext, "FOG_FUNC_POINTS_SGIS" ).setInteger( GL_FOG_FUNC_POINTS_SGIS );
      self->addClassProperty( c_sdlglext, "MAX_FOG_FUNC_POINTS_SGIS" ).setInteger( GL_MAX_FOG_FUNC_POINTS_SGIS );
      #endif

      #ifndef GL_SGIX_fog_offset
      self->addClassProperty( c_sdlglext, "FOG_OFFSET_SGIX" ).setInteger( GL_FOG_OFFSET_SGIX );
      self->addClassProperty( c_sdlglext, "FOG_OFFSET_VALUE_SGIX" ).setInteger( GL_FOG_OFFSET_VALUE_SGIX );
      #endif

      #ifndef GL_HP_image_transform
      self->addClassProperty( c_sdlglext, "IMAGE_SCALE_X_HP" ).setInteger( GL_IMAGE_SCALE_X_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_SCALE_Y_HP" ).setInteger( GL_IMAGE_SCALE_Y_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_TRANSLATE_X_HP" ).setInteger( GL_IMAGE_TRANSLATE_X_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_TRANSLATE_Y_HP" ).setInteger( GL_IMAGE_TRANSLATE_Y_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_ROTATE_ANGLE_HP" ).setInteger( GL_IMAGE_ROTATE_ANGLE_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_ROTATE_ORIGIN_X_HP" ).setInteger( GL_IMAGE_ROTATE_ORIGIN_X_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_ROTATE_ORIGIN_Y_HP" ).setInteger( GL_IMAGE_ROTATE_ORIGIN_Y_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_MAG_FILTER_HP" ).setInteger( GL_IMAGE_MAG_FILTER_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_MIN_FILTER_HP" ).setInteger( GL_IMAGE_MIN_FILTER_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_CUBIC_WEIGHT_HP" ).setInteger( GL_IMAGE_CUBIC_WEIGHT_HP );
      self->addClassProperty( c_sdlglext, "CUBIC_HP" ).setInteger( GL_CUBIC_HP );
      self->addClassProperty( c_sdlglext, "AVERAGE_HP" ).setInteger( GL_AVERAGE_HP );
      self->addClassProperty( c_sdlglext, "IMAGE_TRANSFORM_2D_HP" ).setInteger( GL_IMAGE_TRANSFORM_2D_HP );
      self->addClassProperty( c_sdlglext, "POST_IMAGE_TRANSFORM_COLOR_TABLE_HP" ).setInteger( GL_POST_IMAGE_TRANSFORM_COLOR_TABLE_HP );
      self->addClassProperty( c_sdlglext, "PROXY_POST_IMAGE_TRANSFORM_COLOR_TABLE_HP" ).setInteger( GL_PROXY_POST_IMAGE_TRANSFORM_COLOR_TABLE_HP );
      #endif

      #ifndef GL_HP_convolution_border_modes
      self->addClassProperty( c_sdlglext, "IGNORE_BORDER_HP" ).setInteger( GL_IGNORE_BORDER_HP );
      self->addClassProperty( c_sdlglext, "CONSTANT_BORDER_HP" ).setInteger( GL_CONSTANT_BORDER_HP );
      self->addClassProperty( c_sdlglext, "REPLICATE_BORDER_HP" ).setInteger( GL_REPLICATE_BORDER_HP );
      self->addClassProperty( c_sdlglext, "CONVOLUTION_BORDER_COLOR_HP" ).setInteger( GL_CONVOLUTION_BORDER_COLOR_HP );
      #endif

      #ifndef GL_SGIX_texture_add_env
      self->addClassProperty( c_sdlglext, "TEXTURE_ENV_BIAS_SGIX" ).setInteger( GL_TEXTURE_ENV_BIAS_SGIX );
      #endif

      #ifndef GL_PGI_vertex_hints
      self->addClassProperty( c_sdlglext, "VERTEX_DATA_HINT_PGI" ).setInteger( GL_VERTEX_DATA_HINT_PGI );
      self->addClassProperty( c_sdlglext, "VERTEX_CONSISTENT_HINT_PGI" ).setInteger( GL_VERTEX_CONSISTENT_HINT_PGI );
      self->addClassProperty( c_sdlglext, "MATERIAL_SIDE_HINT_PGI" ).setInteger( GL_MATERIAL_SIDE_HINT_PGI );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_HINT_PGI" ).setInteger( GL_MAX_VERTEX_HINT_PGI );
      self->addClassProperty( c_sdlglext, "COLOR3_BIT_PGI" ).setInteger( GL_COLOR3_BIT_PGI );
      self->addClassProperty( c_sdlglext, "COLOR4_BIT_PGI" ).setInteger( GL_COLOR4_BIT_PGI );
      self->addClassProperty( c_sdlglext, "EDGEFLAG_BIT_PGI" ).setInteger( GL_EDGEFLAG_BIT_PGI );
      self->addClassProperty( c_sdlglext, "INDEX_BIT_PGI" ).setInteger( GL_INDEX_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_AMBIENT_BIT_PGI" ).setInteger( GL_MAT_AMBIENT_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_AMBIENT_AND_DIFFUSE_BIT_PGI" ).setInteger( GL_MAT_AMBIENT_AND_DIFFUSE_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_DIFFUSE_BIT_PGI" ).setInteger( GL_MAT_DIFFUSE_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_EMISSION_BIT_PGI" ).setInteger( GL_MAT_EMISSION_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_COLOR_INDEXES_BIT_PGI" ).setInteger( GL_MAT_COLOR_INDEXES_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_SHININESS_BIT_PGI" ).setInteger( GL_MAT_SHININESS_BIT_PGI );
      self->addClassProperty( c_sdlglext, "MAT_SPECULAR_BIT_PGI" ).setInteger( GL_MAT_SPECULAR_BIT_PGI );
      self->addClassProperty( c_sdlglext, "NORMAL_BIT_PGI" ).setInteger( GL_NORMAL_BIT_PGI );
      self->addClassProperty( c_sdlglext, "TEXCOORD1_BIT_PGI" ).setInteger( GL_TEXCOORD1_BIT_PGI );
      self->addClassProperty( c_sdlglext, "TEXCOORD2_BIT_PGI" ).setInteger( GL_TEXCOORD2_BIT_PGI );
      self->addClassProperty( c_sdlglext, "TEXCOORD3_BIT_PGI" ).setInteger( GL_TEXCOORD3_BIT_PGI );
      self->addClassProperty( c_sdlglext, "TEXCOORD4_BIT_PGI" ).setInteger( GL_TEXCOORD4_BIT_PGI );
      self->addClassProperty( c_sdlglext, "VERTEX23_BIT_PGI" ).setInteger( GL_VERTEX23_BIT_PGI );
      self->addClassProperty( c_sdlglext, "VERTEX4_BIT_PGI" ).setInteger( GL_VERTEX4_BIT_PGI );
      #endif

      #ifndef GL_PGI_misc_hints
      self->addClassProperty( c_sdlglext, "PREFER_DOUBLEBUFFER_HINT_PGI" ).setInteger( GL_PREFER_DOUBLEBUFFER_HINT_PGI );
      self->addClassProperty( c_sdlglext, "CONSERVE_MEMORY_HINT_PGI" ).setInteger( GL_CONSERVE_MEMORY_HINT_PGI );
      self->addClassProperty( c_sdlglext, "RECLAIM_MEMORY_HINT_PGI" ).setInteger( GL_RECLAIM_MEMORY_HINT_PGI );
      self->addClassProperty( c_sdlglext, "NATIVE_GRAPHICS_HANDLE_PGI" ).setInteger( GL_NATIVE_GRAPHICS_HANDLE_PGI );
      self->addClassProperty( c_sdlglext, "NATIVE_GRAPHICS_BEGIN_HINT_PGI" ).setInteger( GL_NATIVE_GRAPHICS_BEGIN_HINT_PGI );
      self->addClassProperty( c_sdlglext, "NATIVE_GRAPHICS_END_HINT_PGI" ).setInteger( GL_NATIVE_GRAPHICS_END_HINT_PGI );
      self->addClassProperty( c_sdlglext, "ALWAYS_FAST_HINT_PGI" ).setInteger( GL_ALWAYS_FAST_HINT_PGI );
      self->addClassProperty( c_sdlglext, "ALWAYS_SOFT_HINT_PGI" ).setInteger( GL_ALWAYS_SOFT_HINT_PGI );
      self->addClassProperty( c_sdlglext, "ALLOW_DRAW_OBJ_HINT_PGI" ).setInteger( GL_ALLOW_DRAW_OBJ_HINT_PGI );
      self->addClassProperty( c_sdlglext, "ALLOW_DRAW_WIN_HINT_PGI" ).setInteger( GL_ALLOW_DRAW_WIN_HINT_PGI );
      self->addClassProperty( c_sdlglext, "ALLOW_DRAW_FRG_HINT_PGI" ).setInteger( GL_ALLOW_DRAW_FRG_HINT_PGI );
      self->addClassProperty( c_sdlglext, "ALLOW_DRAW_MEM_HINT_PGI" ).setInteger( GL_ALLOW_DRAW_MEM_HINT_PGI );
      self->addClassProperty( c_sdlglext, "STRICT_DEPTHFUNC_HINT_PGI" ).setInteger( GL_STRICT_DEPTHFUNC_HINT_PGI );
      self->addClassProperty( c_sdlglext, "STRICT_LIGHTING_HINT_PGI" ).setInteger( GL_STRICT_LIGHTING_HINT_PGI );
      self->addClassProperty( c_sdlglext, "STRICT_SCISSOR_HINT_PGI" ).setInteger( GL_STRICT_SCISSOR_HINT_PGI );
      self->addClassProperty( c_sdlglext, "FULL_STIPPLE_HINT_PGI" ).setInteger( GL_FULL_STIPPLE_HINT_PGI );
      self->addClassProperty( c_sdlglext, "CLIP_NEAR_HINT_PGI" ).setInteger( GL_CLIP_NEAR_HINT_PGI );
      self->addClassProperty( c_sdlglext, "CLIP_FAR_HINT_PGI" ).setInteger( GL_CLIP_FAR_HINT_PGI );
      self->addClassProperty( c_sdlglext, "WIDE_LINE_HINT_PGI" ).setInteger( GL_WIDE_LINE_HINT_PGI );
      self->addClassProperty( c_sdlglext, "BACK_NORMALS_HINT_PGI" ).setInteger( GL_BACK_NORMALS_HINT_PGI );
      #endif

      #ifndef GL_EXT_paletted_texture
      self->addClassProperty( c_sdlglext, "COLOR_INDEX1_EXT" ).setInteger( GL_COLOR_INDEX1_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_INDEX2_EXT" ).setInteger( GL_COLOR_INDEX2_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_INDEX4_EXT" ).setInteger( GL_COLOR_INDEX4_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_INDEX8_EXT" ).setInteger( GL_COLOR_INDEX8_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_INDEX12_EXT" ).setInteger( GL_COLOR_INDEX12_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_INDEX16_EXT" ).setInteger( GL_COLOR_INDEX16_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_INDEX_SIZE_EXT" ).setInteger( GL_TEXTURE_INDEX_SIZE_EXT );
      #endif

      #ifndef GL_EXT_clip_volume_hint
      self->addClassProperty( c_sdlglext, "CLIP_VOLUME_CLIPPING_HINT_EXT" ).setInteger( GL_CLIP_VOLUME_CLIPPING_HINT_EXT );
      #endif

      #ifndef GL_SGIX_list_priority
      self->addClassProperty( c_sdlglext, "LIST_PRIORITY_SGIX" ).setInteger( GL_LIST_PRIORITY_SGIX );
      #endif

      #ifndef GL_SGIX_ir_instrument1
      self->addClassProperty( c_sdlglext, "IR_INSTRUMENT1_SGIX" ).setInteger( GL_IR_INSTRUMENT1_SGIX );
      #endif

      #ifndef GL_SGIX_calligraphic_fragment
      self->addClassProperty( c_sdlglext, "CALLIGRAPHIC_FRAGMENT_SGIX" ).setInteger( GL_CALLIGRAPHIC_FRAGMENT_SGIX );
      #endif

      #ifndef GL_SGIX_texture_lod_bias
      self->addClassProperty( c_sdlglext, "TEXTURE_LOD_BIAS_S_SGIX" ).setInteger( GL_TEXTURE_LOD_BIAS_S_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_LOD_BIAS_T_SGIX" ).setInteger( GL_TEXTURE_LOD_BIAS_T_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_LOD_BIAS_R_SGIX" ).setInteger( GL_TEXTURE_LOD_BIAS_R_SGIX );
      #endif

      #ifndef GL_SGIX_shadow_ambient
      self->addClassProperty( c_sdlglext, "SHADOW_AMBIENT_SGIX" ).setInteger( GL_SHADOW_AMBIENT_SGIX );
      #endif

      #ifndef GL_EXT_index_material
      self->addClassProperty( c_sdlglext, "INDEX_MATERIAL_EXT" ).setInteger( GL_INDEX_MATERIAL_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_MATERIAL_PARAMETER_EXT" ).setInteger( GL_INDEX_MATERIAL_PARAMETER_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_MATERIAL_FACE_EXT" ).setInteger( GL_INDEX_MATERIAL_FACE_EXT );
      #endif

      #ifndef GL_EXT_index_func
      self->addClassProperty( c_sdlglext, "INDEX_TEST_EXT" ).setInteger( GL_INDEX_TEST_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_TEST_FUNC_EXT" ).setInteger( GL_INDEX_TEST_FUNC_EXT );
      self->addClassProperty( c_sdlglext, "INDEX_TEST_REF_EXT" ).setInteger( GL_INDEX_TEST_REF_EXT );
      #endif

      #ifndef GL_EXT_index_array_formats
      self->addClassProperty( c_sdlglext, "IUI_V2F_EXT" ).setInteger( GL_IUI_V2F_EXT );
      self->addClassProperty( c_sdlglext, "IUI_V3F_EXT" ).setInteger( GL_IUI_V3F_EXT );
      self->addClassProperty( c_sdlglext, "IUI_N3F_V2F_EXT" ).setInteger( GL_IUI_N3F_V2F_EXT );
      self->addClassProperty( c_sdlglext, "IUI_N3F_V3F_EXT" ).setInteger( GL_IUI_N3F_V3F_EXT );
      self->addClassProperty( c_sdlglext, "T2F_IUI_V2F_EXT" ).setInteger( GL_T2F_IUI_V2F_EXT );
      self->addClassProperty( c_sdlglext, "T2F_IUI_V3F_EXT" ).setInteger( GL_T2F_IUI_V3F_EXT );
      self->addClassProperty( c_sdlglext, "T2F_IUI_N3F_V2F_EXT" ).setInteger( GL_T2F_IUI_N3F_V2F_EXT );
      self->addClassProperty( c_sdlglext, "T2F_IUI_N3F_V3F_EXT" ).setInteger( GL_T2F_IUI_N3F_V3F_EXT );
      #endif

      #ifndef GL_EXT_compiled_vertex_array
      self->addClassProperty( c_sdlglext, "ARRAY_ELEMENT_LOCK_FIRST_EXT" ).setInteger( GL_ARRAY_ELEMENT_LOCK_FIRST_EXT );
      self->addClassProperty( c_sdlglext, "ARRAY_ELEMENT_LOCK_COUNT_EXT" ).setInteger( GL_ARRAY_ELEMENT_LOCK_COUNT_EXT );
      #endif

      #ifndef GL_EXT_cull_vertex
      self->addClassProperty( c_sdlglext, "CULL_VERTEX_EXT" ).setInteger( GL_CULL_VERTEX_EXT );
      self->addClassProperty( c_sdlglext, "CULL_VERTEX_EYE_POSITION_EXT" ).setInteger( GL_CULL_VERTEX_EYE_POSITION_EXT );
      self->addClassProperty( c_sdlglext, "CULL_VERTEX_OBJECT_POSITION_EXT" ).setInteger( GL_CULL_VERTEX_OBJECT_POSITION_EXT );
      #endif

      #ifndef GL_SGIX_ycrcb
      self->addClassProperty( c_sdlglext, "YCRCB_422_SGIX" ).setInteger( GL_YCRCB_422_SGIX );
      self->addClassProperty( c_sdlglext, "YCRCB_444_SGIX" ).setInteger( GL_YCRCB_444_SGIX );
      #endif

      #ifndef GL_SGIX_fragment_lighting
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHTING_SGIX" ).setInteger( GL_FRAGMENT_LIGHTING_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_COLOR_MATERIAL_SGIX" ).setInteger( GL_FRAGMENT_COLOR_MATERIAL_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_COLOR_MATERIAL_FACE_SGIX" ).setInteger( GL_FRAGMENT_COLOR_MATERIAL_FACE_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_COLOR_MATERIAL_PARAMETER_SGIX" ).setInteger( GL_FRAGMENT_COLOR_MATERIAL_PARAMETER_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_FRAGMENT_LIGHTS_SGIX" ).setInteger( GL_MAX_FRAGMENT_LIGHTS_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_ACTIVE_LIGHTS_SGIX" ).setInteger( GL_MAX_ACTIVE_LIGHTS_SGIX );
      self->addClassProperty( c_sdlglext, "CURRENT_RASTER_NORMAL_SGIX" ).setInteger( GL_CURRENT_RASTER_NORMAL_SGIX );
      self->addClassProperty( c_sdlglext, "LIGHT_ENV_MODE_SGIX" ).setInteger( GL_LIGHT_ENV_MODE_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT_MODEL_LOCAL_VIEWER_SGIX" ).setInteger( GL_FRAGMENT_LIGHT_MODEL_LOCAL_VIEWER_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT_MODEL_TWO_SIDE_SGIX" ).setInteger( GL_FRAGMENT_LIGHT_MODEL_TWO_SIDE_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT_MODEL_AMBIENT_SGIX" ).setInteger( GL_FRAGMENT_LIGHT_MODEL_AMBIENT_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT_MODEL_NORMAL_INTERPOLATION_SGIX" ).setInteger( GL_FRAGMENT_LIGHT_MODEL_NORMAL_INTERPOLATION_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT0_SGIX" ).setInteger( GL_FRAGMENT_LIGHT0_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT1_SGIX" ).setInteger( GL_FRAGMENT_LIGHT1_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT2_SGIX" ).setInteger( GL_FRAGMENT_LIGHT2_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT3_SGIX" ).setInteger( GL_FRAGMENT_LIGHT3_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT4_SGIX" ).setInteger( GL_FRAGMENT_LIGHT4_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT5_SGIX" ).setInteger( GL_FRAGMENT_LIGHT5_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT6_SGIX" ).setInteger( GL_FRAGMENT_LIGHT6_SGIX );
      self->addClassProperty( c_sdlglext, "FRAGMENT_LIGHT7_SGIX" ).setInteger( GL_FRAGMENT_LIGHT7_SGIX );
      #endif

      #ifndef GL_IBM_rasterpos_clip
      self->addClassProperty( c_sdlglext, "RASTER_POSITION_UNCLIPPED_IBM" ).setInteger( GL_RASTER_POSITION_UNCLIPPED_IBM );
      #endif

      #ifndef GL_HP_texture_lighting
      self->addClassProperty( c_sdlglext, "TEXTURE_LIGHTING_MODE_HP" ).setInteger( GL_TEXTURE_LIGHTING_MODE_HP );
      self->addClassProperty( c_sdlglext, "TEXTURE_POST_SPECULAR_HP" ).setInteger( GL_TEXTURE_POST_SPECULAR_HP );
      self->addClassProperty( c_sdlglext, "TEXTURE_PRE_SPECULAR_HP" ).setInteger( GL_TEXTURE_PRE_SPECULAR_HP );
      #endif

      #ifndef GL_EXT_draw_range_elements
      self->addClassProperty( c_sdlglext, "MAX_ELEMENTS_VERTICES_EXT" ).setInteger( GL_MAX_ELEMENTS_VERTICES_EXT );
      self->addClassProperty( c_sdlglext, "MAX_ELEMENTS_INDICES_EXT" ).setInteger( GL_MAX_ELEMENTS_INDICES_EXT );
      #endif

      #ifndef GL_WIN_phong_shading
      self->addClassProperty( c_sdlglext, "PHONG_WIN" ).setInteger( GL_PHONG_WIN );
      self->addClassProperty( c_sdlglext, "PHONG_HINT_WIN" ).setInteger( GL_PHONG_HINT_WIN );
      #endif

      #ifndef GL_WIN_specular_fog
      self->addClassProperty( c_sdlglext, "FOG_SPECULAR_TEXTURE_WIN" ).setInteger( GL_FOG_SPECULAR_TEXTURE_WIN );
      #endif

      #ifndef GL_EXT_light_texture
      self->addClassProperty( c_sdlglext, "FRAGMENT_MATERIAL_EXT" ).setInteger( GL_FRAGMENT_MATERIAL_EXT );
      self->addClassProperty( c_sdlglext, "FRAGMENT_NORMAL_EXT" ).setInteger( GL_FRAGMENT_NORMAL_EXT );
      self->addClassProperty( c_sdlglext, "FRAGMENT_COLOR_EXT" ).setInteger( GL_FRAGMENT_COLOR_EXT );
      self->addClassProperty( c_sdlglext, "ATTENUATION_EXT" ).setInteger( GL_ATTENUATION_EXT );
      self->addClassProperty( c_sdlglext, "SHADOW_ATTENUATION_EXT" ).setInteger( GL_SHADOW_ATTENUATION_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_APPLICATION_MODE_EXT" ).setInteger( GL_TEXTURE_APPLICATION_MODE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_LIGHT_EXT" ).setInteger( GL_TEXTURE_LIGHT_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_MATERIAL_FACE_EXT" ).setInteger( GL_TEXTURE_MATERIAL_FACE_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_MATERIAL_PARAMETER_EXT" ).setInteger( GL_TEXTURE_MATERIAL_PARAMETER_EXT );
      /* reuse GL_FRAGMENT_DEPTH_EXT */
      #endif

      #ifndef GL_SGIX_blend_alpha_minmax
      self->addClassProperty( c_sdlglext, "ALPHA_MIN_SGIX" ).setInteger( GL_ALPHA_MIN_SGIX );
      self->addClassProperty( c_sdlglext, "ALPHA_MAX_SGIX" ).setInteger( GL_ALPHA_MAX_SGIX );
      #endif

      #ifndef GL_SGIX_impact_pixel_texture
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_Q_CEILING_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_Q_CEILING_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_Q_ROUND_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_Q_ROUND_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_Q_FLOOR_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_Q_FLOOR_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_ALPHA_REPLACE_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_ALPHA_REPLACE_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_ALPHA_NO_REPLACE_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_ALPHA_NO_REPLACE_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_ALPHA_LS_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_ALPHA_LS_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_TEX_GEN_ALPHA_MS_SGIX" ).setInteger( GL_PIXEL_TEX_GEN_ALPHA_MS_SGIX );
      #endif

      #ifndef GL_EXT_bgra
      self->addClassProperty( c_sdlglext, "BGR_EXT" ).setInteger( GL_BGR_EXT );
      self->addClassProperty( c_sdlglext, "BGRA_EXT" ).setInteger( GL_BGRA_EXT );
      #endif

      #ifndef GL_SGIX_async
      self->addClassProperty( c_sdlglext, "ASYNC_MARKER_SGIX" ).setInteger( GL_ASYNC_MARKER_SGIX );
      #endif

      #ifndef GL_SGIX_async_pixel
      self->addClassProperty( c_sdlglext, "ASYNC_TEX_IMAGE_SGIX" ).setInteger( GL_ASYNC_TEX_IMAGE_SGIX );
      self->addClassProperty( c_sdlglext, "ASYNC_DRAW_PIXELS_SGIX" ).setInteger( GL_ASYNC_DRAW_PIXELS_SGIX );
      self->addClassProperty( c_sdlglext, "ASYNC_READ_PIXELS_SGIX" ).setInteger( GL_ASYNC_READ_PIXELS_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_ASYNC_TEX_IMAGE_SGIX" ).setInteger( GL_MAX_ASYNC_TEX_IMAGE_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_ASYNC_DRAW_PIXELS_SGIX" ).setInteger( GL_MAX_ASYNC_DRAW_PIXELS_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_ASYNC_READ_PIXELS_SGIX" ).setInteger( GL_MAX_ASYNC_READ_PIXELS_SGIX );
      #endif

      #ifndef GL_SGIX_async_histogram
      self->addClassProperty( c_sdlglext, "ASYNC_HISTOGRAM_SGIX" ).setInteger( GL_ASYNC_HISTOGRAM_SGIX );
      self->addClassProperty( c_sdlglext, "MAX_ASYNC_HISTOGRAM_SGIX" ).setInteger( GL_MAX_ASYNC_HISTOGRAM_SGIX );
      #endif

      #ifndef GL_INTEL_parallel_arrays
      self->addClassProperty( c_sdlglext, "PARALLEL_ARRAYS_INTEL" ).setInteger( GL_PARALLEL_ARRAYS_INTEL );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_PARALLEL_POINTERS_INTEL" ).setInteger( GL_VERTEX_ARRAY_PARALLEL_POINTERS_INTEL );
      self->addClassProperty( c_sdlglext, "NORMAL_ARRAY_PARALLEL_POINTERS_INTEL" ).setInteger( GL_NORMAL_ARRAY_PARALLEL_POINTERS_INTEL );
      self->addClassProperty( c_sdlglext, "COLOR_ARRAY_PARALLEL_POINTERS_INTEL" ).setInteger( GL_COLOR_ARRAY_PARALLEL_POINTERS_INTEL );
      self->addClassProperty( c_sdlglext, "TEXTURE_COORD_ARRAY_PARALLEL_POINTERS_INTEL" ).setInteger( GL_TEXTURE_COORD_ARRAY_PARALLEL_POINTERS_INTEL );
      #endif

      #ifndef GL_HP_occlusion_test
      self->addClassProperty( c_sdlglext, "OCCLUSION_TEST_HP" ).setInteger( GL_OCCLUSION_TEST_HP );
      self->addClassProperty( c_sdlglext, "OCCLUSION_TEST_RESULT_HP" ).setInteger( GL_OCCLUSION_TEST_RESULT_HP );
      #endif

      #ifndef GL_EXT_pixel_transform
      self->addClassProperty( c_sdlglext, "PIXEL_TRANSFORM_2D_EXT" ).setInteger( GL_PIXEL_TRANSFORM_2D_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_MAG_FILTER_EXT" ).setInteger( GL_PIXEL_MAG_FILTER_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_MIN_FILTER_EXT" ).setInteger( GL_PIXEL_MIN_FILTER_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_CUBIC_WEIGHT_EXT" ).setInteger( GL_PIXEL_CUBIC_WEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "CUBIC_EXT" ).setInteger( GL_CUBIC_EXT );
      self->addClassProperty( c_sdlglext, "AVERAGE_EXT" ).setInteger( GL_AVERAGE_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT" ).setInteger( GL_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT );
      self->addClassProperty( c_sdlglext, "MAX_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT" ).setInteger( GL_MAX_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_TRANSFORM_2D_MATRIX_EXT" ).setInteger( GL_PIXEL_TRANSFORM_2D_MATRIX_EXT );
      #endif

      #ifndef GL_EXT_shared_texture_palette
      self->addClassProperty( c_sdlglext, "SHARED_TEXTURE_PALETTE_EXT" ).setInteger( GL_SHARED_TEXTURE_PALETTE_EXT );
      #endif

      #ifndef GL_EXT_separate_specular_color
      self->addClassProperty( c_sdlglext, "LIGHT_MODEL_COLOR_CONTROL_EXT" ).setInteger( GL_LIGHT_MODEL_COLOR_CONTROL_EXT );
      self->addClassProperty( c_sdlglext, "SINGLE_COLOR_EXT" ).setInteger( GL_SINGLE_COLOR_EXT );
      self->addClassProperty( c_sdlglext, "SEPARATE_SPECULAR_COLOR_EXT" ).setInteger( GL_SEPARATE_SPECULAR_COLOR_EXT );
      #endif

      #ifndef GL_EXT_secondary_color
      self->addClassProperty( c_sdlglext, "COLOR_SUM_EXT" ).setInteger( GL_COLOR_SUM_EXT );
      self->addClassProperty( c_sdlglext, "CURRENT_SECONDARY_COLOR_EXT" ).setInteger( GL_CURRENT_SECONDARY_COLOR_EXT );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_SIZE_EXT" ).setInteger( GL_SECONDARY_COLOR_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_TYPE_EXT" ).setInteger( GL_SECONDARY_COLOR_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_STRIDE_EXT" ).setInteger( GL_SECONDARY_COLOR_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_POINTER_EXT" ).setInteger( GL_SECONDARY_COLOR_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_ARRAY_EXT" ).setInteger( GL_SECONDARY_COLOR_ARRAY_EXT );
      #endif

      #ifndef GL_EXT_texture_perturb_normal
      self->addClassProperty( c_sdlglext, "PERTURB_EXT" ).setInteger( GL_PERTURB_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_NORMAL_EXT" ).setInteger( GL_TEXTURE_NORMAL_EXT );
      #endif

      #ifndef GL_EXT_fog_coord
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_SOURCE_EXT" ).setInteger( GL_FOG_COORDINATE_SOURCE_EXT );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_EXT" ).setInteger( GL_FOG_COORDINATE_EXT );
      self->addClassProperty( c_sdlglext, "FRAGMENT_DEPTH_EXT" ).setInteger( GL_FRAGMENT_DEPTH_EXT );
      self->addClassProperty( c_sdlglext, "CURRENT_FOG_COORDINATE_EXT" ).setInteger( GL_CURRENT_FOG_COORDINATE_EXT );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_TYPE_EXT" ).setInteger( GL_FOG_COORDINATE_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_STRIDE_EXT" ).setInteger( GL_FOG_COORDINATE_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_POINTER_EXT" ).setInteger( GL_FOG_COORDINATE_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "FOG_COORDINATE_ARRAY_EXT" ).setInteger( GL_FOG_COORDINATE_ARRAY_EXT );
      #endif

      #ifndef GL_REND_screen_coordinates
      self->addClassProperty( c_sdlglext, "SCREEN_COORDINATES_REND" ).setInteger( GL_SCREEN_COORDINATES_REND );
      self->addClassProperty( c_sdlglext, "INVERTED_SCREEN_W_REND" ).setInteger( GL_INVERTED_SCREEN_W_REND );
      #endif

      #ifndef GL_EXT_coordinate_frame
      self->addClassProperty( c_sdlglext, "TANGENT_ARRAY_EXT" ).setInteger( GL_TANGENT_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "BINORMAL_ARRAY_EXT" ).setInteger( GL_BINORMAL_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "CURRENT_TANGENT_EXT" ).setInteger( GL_CURRENT_TANGENT_EXT );
      self->addClassProperty( c_sdlglext, "CURRENT_BINORMAL_EXT" ).setInteger( GL_CURRENT_BINORMAL_EXT );
      self->addClassProperty( c_sdlglext, "TANGENT_ARRAY_TYPE_EXT" ).setInteger( GL_TANGENT_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "TANGENT_ARRAY_STRIDE_EXT" ).setInteger( GL_TANGENT_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "BINORMAL_ARRAY_TYPE_EXT" ).setInteger( GL_BINORMAL_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "BINORMAL_ARRAY_STRIDE_EXT" ).setInteger( GL_BINORMAL_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "TANGENT_ARRAY_POINTER_EXT" ).setInteger( GL_TANGENT_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "BINORMAL_ARRAY_POINTER_EXT" ).setInteger( GL_BINORMAL_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "MAP1_TANGENT_EXT" ).setInteger( GL_MAP1_TANGENT_EXT );
      self->addClassProperty( c_sdlglext, "MAP2_TANGENT_EXT" ).setInteger( GL_MAP2_TANGENT_EXT );
      self->addClassProperty( c_sdlglext, "MAP1_BINORMAL_EXT" ).setInteger( GL_MAP1_BINORMAL_EXT );
      self->addClassProperty( c_sdlglext, "MAP2_BINORMAL_EXT" ).setInteger( GL_MAP2_BINORMAL_EXT );
      #endif

      #ifndef GL_EXT_texture_env_combine
      self->addClassProperty( c_sdlglext, "COMBINE_EXT" ).setInteger( GL_COMBINE_EXT );
      self->addClassProperty( c_sdlglext, "COMBINE_RGB_EXT" ).setInteger( GL_COMBINE_RGB_EXT );
      self->addClassProperty( c_sdlglext, "COMBINE_ALPHA_EXT" ).setInteger( GL_COMBINE_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "RGB_SCALE_EXT" ).setInteger( GL_RGB_SCALE_EXT );
      self->addClassProperty( c_sdlglext, "ADD_SIGNED_EXT" ).setInteger( GL_ADD_SIGNED_EXT );
      self->addClassProperty( c_sdlglext, "INTERPOLATE_EXT" ).setInteger( GL_INTERPOLATE_EXT );
      self->addClassProperty( c_sdlglext, "CONSTANT_EXT" ).setInteger( GL_CONSTANT_EXT );
      self->addClassProperty( c_sdlglext, "PRIMARY_COLOR_EXT" ).setInteger( GL_PRIMARY_COLOR_EXT );
      self->addClassProperty( c_sdlglext, "PREVIOUS_EXT" ).setInteger( GL_PREVIOUS_EXT );
      self->addClassProperty( c_sdlglext, "SOURCE0_RGB_EXT" ).setInteger( GL_SOURCE0_RGB_EXT );
      self->addClassProperty( c_sdlglext, "SOURCE1_RGB_EXT" ).setInteger( GL_SOURCE1_RGB_EXT );
      self->addClassProperty( c_sdlglext, "SOURCE2_RGB_EXT" ).setInteger( GL_SOURCE2_RGB_EXT );
      self->addClassProperty( c_sdlglext, "SOURCE0_ALPHA_EXT" ).setInteger( GL_SOURCE0_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "SOURCE1_ALPHA_EXT" ).setInteger( GL_SOURCE1_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "SOURCE2_ALPHA_EXT" ).setInteger( GL_SOURCE2_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "OPERAND0_RGB_EXT" ).setInteger( GL_OPERAND0_RGB_EXT );
      self->addClassProperty( c_sdlglext, "OPERAND1_RGB_EXT" ).setInteger( GL_OPERAND1_RGB_EXT );
      self->addClassProperty( c_sdlglext, "OPERAND2_RGB_EXT" ).setInteger( GL_OPERAND2_RGB_EXT );
      self->addClassProperty( c_sdlglext, "OPERAND0_ALPHA_EXT" ).setInteger( GL_OPERAND0_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "OPERAND1_ALPHA_EXT" ).setInteger( GL_OPERAND1_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "OPERAND2_ALPHA_EXT" ).setInteger( GL_OPERAND2_ALPHA_EXT );
      #endif

      #ifndef GL_APPLE_specular_vector
      self->addClassProperty( c_sdlglext, "LIGHT_MODEL_SPECULAR_VECTOR_APPLE" ).setInteger( GL_LIGHT_MODEL_SPECULAR_VECTOR_APPLE );
      #endif

      #ifndef GL_APPLE_transform_hint
      self->addClassProperty( c_sdlglext, "TRANSFORM_HINT_APPLE" ).setInteger( GL_TRANSFORM_HINT_APPLE );
      #endif

      #ifndef GL_SGIX_fog_scale
      self->addClassProperty( c_sdlglext, "FOG_SCALE_SGIX" ).setInteger( GL_FOG_SCALE_SGIX );
      self->addClassProperty( c_sdlglext, "FOG_SCALE_VALUE_SGIX" ).setInteger( GL_FOG_SCALE_VALUE_SGIX );
      #endif

      #ifndef GL_SUNX_constant_data
      self->addClassProperty( c_sdlglext, "UNPACK_CONSTANT_DATA_SUNX" ).setInteger( GL_UNPACK_CONSTANT_DATA_SUNX );
      self->addClassProperty( c_sdlglext, "TEXTURE_CONSTANT_DATA_SUNX" ).setInteger( GL_TEXTURE_CONSTANT_DATA_SUNX );
      #endif

      #ifndef GL_SUN_global_alpha
      self->addClassProperty( c_sdlglext, "GLOBAL_ALPHA_SUN" ).setInteger( GL_GLOBAL_ALPHA_SUN );
      self->addClassProperty( c_sdlglext, "GLOBAL_ALPHA_FACTOR_SUN" ).setInteger( GL_GLOBAL_ALPHA_FACTOR_SUN );
      #endif

      #ifndef GL_SUN_triangle_list
      self->addClassProperty( c_sdlglext, "RESTART_SUN" ).setInteger( GL_RESTART_SUN );
      self->addClassProperty( c_sdlglext, "REPLACE_MIDDLE_SUN" ).setInteger( GL_REPLACE_MIDDLE_SUN );
      self->addClassProperty( c_sdlglext, "REPLACE_OLDEST_SUN" ).setInteger( GL_REPLACE_OLDEST_SUN );
      self->addClassProperty( c_sdlglext, "TRIANGLE_LIST_SUN" ).setInteger( GL_TRIANGLE_LIST_SUN );
      self->addClassProperty( c_sdlglext, "REPLACEMENT_CODE_SUN" ).setInteger( GL_REPLACEMENT_CODE_SUN );
      self->addClassProperty( c_sdlglext, "REPLACEMENT_CODE_ARRAY_SUN" ).setInteger( GL_REPLACEMENT_CODE_ARRAY_SUN );
      self->addClassProperty( c_sdlglext, "REPLACEMENT_CODE_ARRAY_TYPE_SUN" ).setInteger( GL_REPLACEMENT_CODE_ARRAY_TYPE_SUN );
      self->addClassProperty( c_sdlglext, "REPLACEMENT_CODE_ARRAY_STRIDE_SUN" ).setInteger( GL_REPLACEMENT_CODE_ARRAY_STRIDE_SUN );
      self->addClassProperty( c_sdlglext, "REPLACEMENT_CODE_ARRAY_POINTER_SUN" ).setInteger( GL_REPLACEMENT_CODE_ARRAY_POINTER_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_V3F_SUN" ).setInteger( GL_R1UI_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_C4UB_V3F_SUN" ).setInteger( GL_R1UI_C4UB_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_C3F_V3F_SUN" ).setInteger( GL_R1UI_C3F_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_N3F_V3F_SUN" ).setInteger( GL_R1UI_N3F_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_C4F_N3F_V3F_SUN" ).setInteger( GL_R1UI_C4F_N3F_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_T2F_V3F_SUN" ).setInteger( GL_R1UI_T2F_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_T2F_N3F_V3F_SUN" ).setInteger( GL_R1UI_T2F_N3F_V3F_SUN );
      self->addClassProperty( c_sdlglext, "R1UI_T2F_C4F_N3F_V3F_SUN" ).setInteger( GL_R1UI_T2F_C4F_N3F_V3F_SUN );
      #endif

      #ifndef GL_EXT_blend_func_separate
      self->addClassProperty( c_sdlglext, "BLEND_DST_RGB_EXT" ).setInteger( GL_BLEND_DST_RGB_EXT );
      self->addClassProperty( c_sdlglext, "BLEND_SRC_RGB_EXT" ).setInteger( GL_BLEND_SRC_RGB_EXT );
      self->addClassProperty( c_sdlglext, "BLEND_DST_ALPHA_EXT" ).setInteger( GL_BLEND_DST_ALPHA_EXT );
      self->addClassProperty( c_sdlglext, "BLEND_SRC_ALPHA_EXT" ).setInteger( GL_BLEND_SRC_ALPHA_EXT );
      #endif

      #ifndef GL_INGR_color_clamp
      self->addClassProperty( c_sdlglext, "RED_MIN_CLAMP_INGR" ).setInteger( GL_RED_MIN_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "GREEN_MIN_CLAMP_INGR" ).setInteger( GL_GREEN_MIN_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "BLUE_MIN_CLAMP_INGR" ).setInteger( GL_BLUE_MIN_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "ALPHA_MIN_CLAMP_INGR" ).setInteger( GL_ALPHA_MIN_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "RED_MAX_CLAMP_INGR" ).setInteger( GL_RED_MAX_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "GREEN_MAX_CLAMP_INGR" ).setInteger( GL_GREEN_MAX_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "BLUE_MAX_CLAMP_INGR" ).setInteger( GL_BLUE_MAX_CLAMP_INGR );
      self->addClassProperty( c_sdlglext, "ALPHA_MAX_CLAMP_INGR" ).setInteger( GL_ALPHA_MAX_CLAMP_INGR );
      #endif

      #ifndef GL_INGR_interlace_read
      self->addClassProperty( c_sdlglext, "INTERLACE_READ_INGR" ).setInteger( GL_INTERLACE_READ_INGR );
      #endif

      #ifndef GL_EXT_stencil_wrap
      self->addClassProperty( c_sdlglext, "INCR_WRAP_EXT" ).setInteger( GL_INCR_WRAP_EXT );
      self->addClassProperty( c_sdlglext, "DECR_WRAP_EXT" ).setInteger( GL_DECR_WRAP_EXT );
      #endif

      #ifndef GL_EXT_422_pixels
      self->addClassProperty( c_sdlglext, "422_EXT" ).setInteger( GL_422_EXT );
      self->addClassProperty( c_sdlglext, "422_REV_EXT" ).setInteger( GL_422_REV_EXT );
      self->addClassProperty( c_sdlglext, "422_AVERAGE_EXT" ).setInteger( GL_422_AVERAGE_EXT );
      self->addClassProperty( c_sdlglext, "422_REV_AVERAGE_EXT" ).setInteger( GL_422_REV_AVERAGE_EXT );
      #endif

      #ifndef GL_NV_texgen_reflection
      self->addClassProperty( c_sdlglext, "NORMAL_MAP_NV" ).setInteger( GL_NORMAL_MAP_NV );
      self->addClassProperty( c_sdlglext, "REFLECTION_MAP_NV" ).setInteger( GL_REFLECTION_MAP_NV );
      #endif

      #ifndef GL_EXT_texture_cube_map
      self->addClassProperty( c_sdlglext, "NORMAL_MAP_EXT" ).setInteger( GL_NORMAL_MAP_EXT );
      self->addClassProperty( c_sdlglext, "REFLECTION_MAP_EXT" ).setInteger( GL_REFLECTION_MAP_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_BINDING_CUBE_MAP_EXT" ).setInteger( GL_TEXTURE_BINDING_CUBE_MAP_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_X_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_X_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_Y_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_POSITIVE_Z_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT" ).setInteger( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_CUBE_MAP_EXT" ).setInteger( GL_PROXY_TEXTURE_CUBE_MAP_EXT );
      self->addClassProperty( c_sdlglext, "MAX_CUBE_MAP_TEXTURE_SIZE_EXT" ).setInteger( GL_MAX_CUBE_MAP_TEXTURE_SIZE_EXT );
      #endif

      #ifndef GL_SUN_convolution_border_modes
      self->addClassProperty( c_sdlglext, "WRAP_BORDER_SUN" ).setInteger( GL_WRAP_BORDER_SUN );
      #endif

      #ifndef GL_EXT_texture_lod_bias
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_LOD_BIAS_EXT" ).setInteger( GL_MAX_TEXTURE_LOD_BIAS_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_FILTER_CONTROL_EXT" ).setInteger( GL_TEXTURE_FILTER_CONTROL_EXT );
      self->addClassProperty( c_sdlglext, "TEXTURE_LOD_BIAS_EXT" ).setInteger( GL_TEXTURE_LOD_BIAS_EXT );
      #endif

      #ifndef GL_EXT_texture_filter_anisotropic
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_ANISOTROPY_EXT" ).setInteger( GL_TEXTURE_MAX_ANISOTROPY_EXT );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_MAX_ANISOTROPY_EXT" ).setInteger( GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT );
      #endif

      #ifndef GL_EXT_vertex_weighting
      #define GL_MODELVIEW0_STACK_DEPTH_EXT     GL_MODELVIEW_STACK_DEPTH
      self->addClassProperty( c_sdlglext, "MODELVIEW1_STACK_DEPTH_EXT" ).setInteger( GL_MODELVIEW1_STACK_DEPTH_EXT );
      #define GL_MODELVIEW0_MATRIX_EXT          GL_MODELVIEW_MATRIX
      self->addClassProperty( c_sdlglext, "MODELVIEW1_MATRIX_EXT" ).setInteger( GL_MODELVIEW1_MATRIX_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_WEIGHTING_EXT" ).setInteger( GL_VERTEX_WEIGHTING_EXT );
      #define GL_MODELVIEW0_EXT                 GL_MODELVIEW
      self->addClassProperty( c_sdlglext, "MODELVIEW1_EXT" ).setInteger( GL_MODELVIEW1_EXT );
      self->addClassProperty( c_sdlglext, "CURRENT_VERTEX_WEIGHT_EXT" ).setInteger( GL_CURRENT_VERTEX_WEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_WEIGHT_ARRAY_EXT" ).setInteger( GL_VERTEX_WEIGHT_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_WEIGHT_ARRAY_SIZE_EXT" ).setInteger( GL_VERTEX_WEIGHT_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_WEIGHT_ARRAY_TYPE_EXT" ).setInteger( GL_VERTEX_WEIGHT_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_WEIGHT_ARRAY_STRIDE_EXT" ).setInteger( GL_VERTEX_WEIGHT_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_WEIGHT_ARRAY_POINTER_EXT" ).setInteger( GL_VERTEX_WEIGHT_ARRAY_POINTER_EXT );
      #endif

      #ifndef GL_NV_light_max_exponent
      self->addClassProperty( c_sdlglext, "MAX_SHININESS_NV" ).setInteger( GL_MAX_SHININESS_NV );
      self->addClassProperty( c_sdlglext, "MAX_SPOT_EXPONENT_NV" ).setInteger( GL_MAX_SPOT_EXPONENT_NV );
      #endif

      #ifndef GL_NV_vertex_array_range
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_NV" ).setInteger( GL_VERTEX_ARRAY_RANGE_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_LENGTH_NV" ).setInteger( GL_VERTEX_ARRAY_RANGE_LENGTH_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_VALID_NV" ).setInteger( GL_VERTEX_ARRAY_RANGE_VALID_NV );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_ARRAY_RANGE_ELEMENT_NV" ).setInteger( GL_MAX_VERTEX_ARRAY_RANGE_ELEMENT_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_POINTER_NV" ).setInteger( GL_VERTEX_ARRAY_RANGE_POINTER_NV );
      #endif

      #ifndef GL_NV_register_combiners
      self->addClassProperty( c_sdlglext, "REGISTER_COMBINERS_NV" ).setInteger( GL_REGISTER_COMBINERS_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_A_NV" ).setInteger( GL_VARIABLE_A_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_B_NV" ).setInteger( GL_VARIABLE_B_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_C_NV" ).setInteger( GL_VARIABLE_C_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_D_NV" ).setInteger( GL_VARIABLE_D_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_E_NV" ).setInteger( GL_VARIABLE_E_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_F_NV" ).setInteger( GL_VARIABLE_F_NV );
      self->addClassProperty( c_sdlglext, "VARIABLE_G_NV" ).setInteger( GL_VARIABLE_G_NV );
      self->addClassProperty( c_sdlglext, "CONSTANT_COLOR0_NV" ).setInteger( GL_CONSTANT_COLOR0_NV );
      self->addClassProperty( c_sdlglext, "CONSTANT_COLOR1_NV" ).setInteger( GL_CONSTANT_COLOR1_NV );
      self->addClassProperty( c_sdlglext, "PRIMARY_COLOR_NV" ).setInteger( GL_PRIMARY_COLOR_NV );
      self->addClassProperty( c_sdlglext, "SECONDARY_COLOR_NV" ).setInteger( GL_SECONDARY_COLOR_NV );
      self->addClassProperty( c_sdlglext, "SPARE0_NV" ).setInteger( GL_SPARE0_NV );
      self->addClassProperty( c_sdlglext, "SPARE1_NV" ).setInteger( GL_SPARE1_NV );
      self->addClassProperty( c_sdlglext, "DISCARD_NV" ).setInteger( GL_DISCARD_NV );
      self->addClassProperty( c_sdlglext, "E_TIMES_F_NV" ).setInteger( GL_E_TIMES_F_NV );
      self->addClassProperty( c_sdlglext, "SPARE0_PLUS_SECONDARY_COLOR_NV" ).setInteger( GL_SPARE0_PLUS_SECONDARY_COLOR_NV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_IDENTITY_NV" ).setInteger( GL_UNSIGNED_IDENTITY_NV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INVERT_NV" ).setInteger( GL_UNSIGNED_INVERT_NV );
      self->addClassProperty( c_sdlglext, "EXPAND_NORMAL_NV" ).setInteger( GL_EXPAND_NORMAL_NV );
      self->addClassProperty( c_sdlglext, "EXPAND_NEGATE_NV" ).setInteger( GL_EXPAND_NEGATE_NV );
      self->addClassProperty( c_sdlglext, "HALF_BIAS_NORMAL_NV" ).setInteger( GL_HALF_BIAS_NORMAL_NV );
      self->addClassProperty( c_sdlglext, "HALF_BIAS_NEGATE_NV" ).setInteger( GL_HALF_BIAS_NEGATE_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_IDENTITY_NV" ).setInteger( GL_SIGNED_IDENTITY_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_NEGATE_NV" ).setInteger( GL_SIGNED_NEGATE_NV );
      self->addClassProperty( c_sdlglext, "SCALE_BY_TWO_NV" ).setInteger( GL_SCALE_BY_TWO_NV );
      self->addClassProperty( c_sdlglext, "SCALE_BY_FOUR_NV" ).setInteger( GL_SCALE_BY_FOUR_NV );
      self->addClassProperty( c_sdlglext, "SCALE_BY_ONE_HALF_NV" ).setInteger( GL_SCALE_BY_ONE_HALF_NV );
      self->addClassProperty( c_sdlglext, "BIAS_BY_NEGATIVE_ONE_HALF_NV" ).setInteger( GL_BIAS_BY_NEGATIVE_ONE_HALF_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_INPUT_NV" ).setInteger( GL_COMBINER_INPUT_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_MAPPING_NV" ).setInteger( GL_COMBINER_MAPPING_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_COMPONENT_USAGE_NV" ).setInteger( GL_COMBINER_COMPONENT_USAGE_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_AB_DOT_PRODUCT_NV" ).setInteger( GL_COMBINER_AB_DOT_PRODUCT_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_CD_DOT_PRODUCT_NV" ).setInteger( GL_COMBINER_CD_DOT_PRODUCT_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_MUX_SUM_NV" ).setInteger( GL_COMBINER_MUX_SUM_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_SCALE_NV" ).setInteger( GL_COMBINER_SCALE_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_BIAS_NV" ).setInteger( GL_COMBINER_BIAS_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_AB_OUTPUT_NV" ).setInteger( GL_COMBINER_AB_OUTPUT_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_CD_OUTPUT_NV" ).setInteger( GL_COMBINER_CD_OUTPUT_NV );
      self->addClassProperty( c_sdlglext, "COMBINER_SUM_OUTPUT_NV" ).setInteger( GL_COMBINER_SUM_OUTPUT_NV );
      self->addClassProperty( c_sdlglext, "MAX_GENERAL_COMBINERS_NV" ).setInteger( GL_MAX_GENERAL_COMBINERS_NV );
      self->addClassProperty( c_sdlglext, "NUM_GENERAL_COMBINERS_NV" ).setInteger( GL_NUM_GENERAL_COMBINERS_NV );
      self->addClassProperty( c_sdlglext, "COLOR_SUM_CLAMP_NV" ).setInteger( GL_COLOR_SUM_CLAMP_NV );
      self->addClassProperty( c_sdlglext, "COMBINER0_NV" ).setInteger( GL_COMBINER0_NV );
      self->addClassProperty( c_sdlglext, "COMBINER1_NV" ).setInteger( GL_COMBINER1_NV );
      self->addClassProperty( c_sdlglext, "COMBINER2_NV" ).setInteger( GL_COMBINER2_NV );
      self->addClassProperty( c_sdlglext, "COMBINER3_NV" ).setInteger( GL_COMBINER3_NV );
      self->addClassProperty( c_sdlglext, "COMBINER4_NV" ).setInteger( GL_COMBINER4_NV );
      self->addClassProperty( c_sdlglext, "COMBINER5_NV" ).setInteger( GL_COMBINER5_NV );
      self->addClassProperty( c_sdlglext, "COMBINER6_NV" ).setInteger( GL_COMBINER6_NV );
      self->addClassProperty( c_sdlglext, "COMBINER7_NV" ).setInteger( GL_COMBINER7_NV );
      /* reuse GL_TEXTURE0_ARB */
      /* reuse GL_TEXTURE1_ARB */
      /* reuse GL_ZERO */
      /* reuse GL_NONE */
      /* reuse GL_FOG */
      #endif

      #ifndef GL_NV_fog_distance
      self->addClassProperty( c_sdlglext, "FOG_DISTANCE_MODE_NV" ).setInteger( GL_FOG_DISTANCE_MODE_NV );
      self->addClassProperty( c_sdlglext, "EYE_RADIAL_NV" ).setInteger( GL_EYE_RADIAL_NV );
      self->addClassProperty( c_sdlglext, "EYE_PLANE_ABSOLUTE_NV" ).setInteger( GL_EYE_PLANE_ABSOLUTE_NV );
      /* reuse GL_EYE_PLANE */
      #endif

      #ifndef GL_NV_texgen_emboss
      self->addClassProperty( c_sdlglext, "EMBOSS_LIGHT_NV" ).setInteger( GL_EMBOSS_LIGHT_NV );
      self->addClassProperty( c_sdlglext, "EMBOSS_CONSTANT_NV" ).setInteger( GL_EMBOSS_CONSTANT_NV );
      self->addClassProperty( c_sdlglext, "EMBOSS_MAP_NV" ).setInteger( GL_EMBOSS_MAP_NV );
      #endif

      #ifndef GL_NV_texture_env_combine4
      self->addClassProperty( c_sdlglext, "COMBINE4_NV" ).setInteger( GL_COMBINE4_NV );
      self->addClassProperty( c_sdlglext, "SOURCE3_RGB_NV" ).setInteger( GL_SOURCE3_RGB_NV );
      self->addClassProperty( c_sdlglext, "SOURCE3_ALPHA_NV" ).setInteger( GL_SOURCE3_ALPHA_NV );
      self->addClassProperty( c_sdlglext, "OPERAND3_RGB_NV" ).setInteger( GL_OPERAND3_RGB_NV );
      self->addClassProperty( c_sdlglext, "OPERAND3_ALPHA_NV" ).setInteger( GL_OPERAND3_ALPHA_NV );
      #endif

      #ifndef GL_EXT_texture_compression_s3tc
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGB_S3TC_DXT1_EXT" ).setInteger( GL_COMPRESSED_RGB_S3TC_DXT1_EXT );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGBA_S3TC_DXT1_EXT" ).setInteger( GL_COMPRESSED_RGBA_S3TC_DXT1_EXT );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGBA_S3TC_DXT3_EXT" ).setInteger( GL_COMPRESSED_RGBA_S3TC_DXT3_EXT );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGBA_S3TC_DXT5_EXT" ).setInteger( GL_COMPRESSED_RGBA_S3TC_DXT5_EXT );
      #endif

      #ifndef GL_IBM_cull_vertex
      #define GL_CULL_VERTEX_IBM                103050
      #endif

      #ifndef GL_IBM_vertex_array_lists
      #define GL_VERTEX_ARRAY_LIST_IBM          103070
      #define GL_NORMAL_ARRAY_LIST_IBM          103071
      #define GL_COLOR_ARRAY_LIST_IBM           103072
      #define GL_INDEX_ARRAY_LIST_IBM           103073
      #define GL_TEXTURE_COORD_ARRAY_LIST_IBM   103074
      #define GL_EDGE_FLAG_ARRAY_LIST_IBM       103075
      #define GL_FOG_COORDINATE_ARRAY_LIST_IBM  103076
      #define GL_SECONDARY_COLOR_ARRAY_LIST_IBM 103077
      #define GL_VERTEX_ARRAY_LIST_STRIDE_IBM   103080
      #define GL_NORMAL_ARRAY_LIST_STRIDE_IBM   103081
      #define GL_COLOR_ARRAY_LIST_STRIDE_IBM    103082
      #define GL_INDEX_ARRAY_LIST_STRIDE_IBM    103083
      #define GL_TEXTURE_COORD_ARRAY_LIST_STRIDE_IBM 103084
      #define GL_EDGE_FLAG_ARRAY_LIST_STRIDE_IBM 103085
      #define GL_FOG_COORDINATE_ARRAY_LIST_STRIDE_IBM 103086
      #define GL_SECONDARY_COLOR_ARRAY_LIST_STRIDE_IBM 103087
      #endif

      #ifndef GL_SGIX_subsample
      self->addClassProperty( c_sdlglext, "PACK_SUBSAMPLE_RATE_SGIX" ).setInteger( GL_PACK_SUBSAMPLE_RATE_SGIX );
      self->addClassProperty( c_sdlglext, "UNPACK_SUBSAMPLE_RATE_SGIX" ).setInteger( GL_UNPACK_SUBSAMPLE_RATE_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_SUBSAMPLE_4444_SGIX" ).setInteger( GL_PIXEL_SUBSAMPLE_4444_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_SUBSAMPLE_2424_SGIX" ).setInteger( GL_PIXEL_SUBSAMPLE_2424_SGIX );
      self->addClassProperty( c_sdlglext, "PIXEL_SUBSAMPLE_4242_SGIX" ).setInteger( GL_PIXEL_SUBSAMPLE_4242_SGIX );
      #endif

      #ifndef GL_SGIX_ycrcba
      self->addClassProperty( c_sdlglext, "YCRCB_SGIX" ).setInteger( GL_YCRCB_SGIX );
      self->addClassProperty( c_sdlglext, "YCRCBA_SGIX" ).setInteger( GL_YCRCBA_SGIX );
      #endif

      #ifndef GL_SGI_depth_pass_instrument
      self->addClassProperty( c_sdlglext, "DEPTH_PASS_INSTRUMENT_SGIX" ).setInteger( GL_DEPTH_PASS_INSTRUMENT_SGIX );
      self->addClassProperty( c_sdlglext, "DEPTH_PASS_INSTRUMENT_COUNTERS_SGIX" ).setInteger( GL_DEPTH_PASS_INSTRUMENT_COUNTERS_SGIX );
      self->addClassProperty( c_sdlglext, "DEPTH_PASS_INSTRUMENT_MAX_SGIX" ).setInteger( GL_DEPTH_PASS_INSTRUMENT_MAX_SGIX );
      #endif

      #ifndef GL_3DFX_texture_compression_FXT1
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGB_FXT1_3DFX" ).setInteger( GL_COMPRESSED_RGB_FXT1_3DFX );
      self->addClassProperty( c_sdlglext, "COMPRESSED_RGBA_FXT1_3DFX" ).setInteger( GL_COMPRESSED_RGBA_FXT1_3DFX );
      #endif

      #ifndef GL_3DFX_multisample
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_3DFX" ).setInteger( GL_MULTISAMPLE_3DFX );
      self->addClassProperty( c_sdlglext, "SAMPLE_BUFFERS_3DFX" ).setInteger( GL_SAMPLE_BUFFERS_3DFX );
      self->addClassProperty( c_sdlglext, "SAMPLES_3DFX" ).setInteger( GL_SAMPLES_3DFX );
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_BIT_3DFX" ).setInteger( GL_MULTISAMPLE_BIT_3DFX );
      #endif


      #ifndef GL_EXT_multisample
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_EXT" ).setInteger( GL_MULTISAMPLE_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_MASK_EXT" ).setInteger( GL_SAMPLE_ALPHA_TO_MASK_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_ALPHA_TO_ONE_EXT" ).setInteger( GL_SAMPLE_ALPHA_TO_ONE_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_MASK_EXT" ).setInteger( GL_SAMPLE_MASK_EXT );
      self->addClassProperty( c_sdlglext, "1PASS_EXT" ).setInteger( GL_1PASS_EXT );
      self->addClassProperty( c_sdlglext, "2PASS_0_EXT" ).setInteger( GL_2PASS_0_EXT );
      self->addClassProperty( c_sdlglext, "2PASS_1_EXT" ).setInteger( GL_2PASS_1_EXT );
      self->addClassProperty( c_sdlglext, "4PASS_0_EXT" ).setInteger( GL_4PASS_0_EXT );
      self->addClassProperty( c_sdlglext, "4PASS_1_EXT" ).setInteger( GL_4PASS_1_EXT );
      self->addClassProperty( c_sdlglext, "4PASS_2_EXT" ).setInteger( GL_4PASS_2_EXT );
      self->addClassProperty( c_sdlglext, "4PASS_3_EXT" ).setInteger( GL_4PASS_3_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_BUFFERS_EXT" ).setInteger( GL_SAMPLE_BUFFERS_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLES_EXT" ).setInteger( GL_SAMPLES_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_MASK_VALUE_EXT" ).setInteger( GL_SAMPLE_MASK_VALUE_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_MASK_INVERT_EXT" ).setInteger( GL_SAMPLE_MASK_INVERT_EXT );
      self->addClassProperty( c_sdlglext, "SAMPLE_PATTERN_EXT" ).setInteger( GL_SAMPLE_PATTERN_EXT );
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_BIT_EXT" ).setInteger( GL_MULTISAMPLE_BIT_EXT );
      #endif

      #ifndef GL_SGIX_vertex_preclip
      self->addClassProperty( c_sdlglext, "VERTEX_PRECLIP_SGIX" ).setInteger( GL_VERTEX_PRECLIP_SGIX );
      self->addClassProperty( c_sdlglext, "VERTEX_PRECLIP_HINT_SGIX" ).setInteger( GL_VERTEX_PRECLIP_HINT_SGIX );
      #endif

      #ifndef GL_SGIX_convolution_accuracy
      self->addClassProperty( c_sdlglext, "CONVOLUTION_HINT_SGIX" ).setInteger( GL_CONVOLUTION_HINT_SGIX );
      #endif

      #ifndef GL_SGIX_resample
      self->addClassProperty( c_sdlglext, "PACK_RESAMPLE_SGIX" ).setInteger( GL_PACK_RESAMPLE_SGIX );
      self->addClassProperty( c_sdlglext, "UNPACK_RESAMPLE_SGIX" ).setInteger( GL_UNPACK_RESAMPLE_SGIX );
      self->addClassProperty( c_sdlglext, "RESAMPLE_REPLICATE_SGIX" ).setInteger( GL_RESAMPLE_REPLICATE_SGIX );
      self->addClassProperty( c_sdlglext, "RESAMPLE_ZERO_FILL_SGIX" ).setInteger( GL_RESAMPLE_ZERO_FILL_SGIX );
      self->addClassProperty( c_sdlglext, "RESAMPLE_DECIMATE_SGIX" ).setInteger( GL_RESAMPLE_DECIMATE_SGIX );
      #endif

      #ifndef GL_SGIS_point_line_texgen
      self->addClassProperty( c_sdlglext, "EYE_DISTANCE_TO_POINT_SGIS" ).setInteger( GL_EYE_DISTANCE_TO_POINT_SGIS );
      self->addClassProperty( c_sdlglext, "OBJECT_DISTANCE_TO_POINT_SGIS" ).setInteger( GL_OBJECT_DISTANCE_TO_POINT_SGIS );
      self->addClassProperty( c_sdlglext, "EYE_DISTANCE_TO_LINE_SGIS" ).setInteger( GL_EYE_DISTANCE_TO_LINE_SGIS );
      self->addClassProperty( c_sdlglext, "OBJECT_DISTANCE_TO_LINE_SGIS" ).setInteger( GL_OBJECT_DISTANCE_TO_LINE_SGIS );
      self->addClassProperty( c_sdlglext, "EYE_POINT_SGIS" ).setInteger( GL_EYE_POINT_SGIS );
      self->addClassProperty( c_sdlglext, "OBJECT_POINT_SGIS" ).setInteger( GL_OBJECT_POINT_SGIS );
      self->addClassProperty( c_sdlglext, "EYE_LINE_SGIS" ).setInteger( GL_EYE_LINE_SGIS );
      self->addClassProperty( c_sdlglext, "OBJECT_LINE_SGIS" ).setInteger( GL_OBJECT_LINE_SGIS );
      #endif

      #ifndef GL_SGIS_texture_color_mask
      self->addClassProperty( c_sdlglext, "TEXTURE_COLOR_WRITEMASK_SGIS" ).setInteger( GL_TEXTURE_COLOR_WRITEMASK_SGIS );
      #endif

      #ifndef GL_EXT_texture_env_dot3
      self->addClassProperty( c_sdlglext, "DOT3_RGB_EXT" ).setInteger( GL_DOT3_RGB_EXT );
      self->addClassProperty( c_sdlglext, "DOT3_RGBA_EXT" ).setInteger( GL_DOT3_RGBA_EXT );
      #endif

      #ifndef GL_ATI_texture_mirror_once
      self->addClassProperty( c_sdlglext, "MIRROR_CLAMP_ATI" ).setInteger( GL_MIRROR_CLAMP_ATI );
      self->addClassProperty( c_sdlglext, "MIRROR_CLAMP_TO_EDGE_ATI" ).setInteger( GL_MIRROR_CLAMP_TO_EDGE_ATI );
      #endif

      #ifndef GL_NV_fence
      self->addClassProperty( c_sdlglext, "ALL_COMPLETED_NV" ).setInteger( GL_ALL_COMPLETED_NV );
      self->addClassProperty( c_sdlglext, "FENCE_STATUS_NV" ).setInteger( GL_FENCE_STATUS_NV );
      self->addClassProperty( c_sdlglext, "FENCE_CONDITION_NV" ).setInteger( GL_FENCE_CONDITION_NV );
      #endif

      #ifndef GL_IBM_texture_mirrored_repeat
      self->addClassProperty( c_sdlglext, "MIRRORED_REPEAT_IBM" ).setInteger( GL_MIRRORED_REPEAT_IBM );
      #endif

      #ifndef GL_NV_evaluators
      self->addClassProperty( c_sdlglext, "EVAL_2D_NV" ).setInteger( GL_EVAL_2D_NV );
      self->addClassProperty( c_sdlglext, "EVAL_TRIANGULAR_2D_NV" ).setInteger( GL_EVAL_TRIANGULAR_2D_NV );
      self->addClassProperty( c_sdlglext, "MAP_TESSELLATION_NV" ).setInteger( GL_MAP_TESSELLATION_NV );
      self->addClassProperty( c_sdlglext, "MAP_ATTRIB_U_ORDER_NV" ).setInteger( GL_MAP_ATTRIB_U_ORDER_NV );
      self->addClassProperty( c_sdlglext, "MAP_ATTRIB_V_ORDER_NV" ).setInteger( GL_MAP_ATTRIB_V_ORDER_NV );
      self->addClassProperty( c_sdlglext, "EVAL_FRACTIONAL_TESSELLATION_NV" ).setInteger( GL_EVAL_FRACTIONAL_TESSELLATION_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB0_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB0_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB1_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB1_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB2_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB2_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB3_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB3_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB4_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB4_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB5_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB5_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB6_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB6_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB7_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB7_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB8_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB8_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB9_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB9_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB10_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB10_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB11_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB11_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB12_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB12_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB13_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB13_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB14_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB14_NV );
      self->addClassProperty( c_sdlglext, "EVAL_VERTEX_ATTRIB15_NV" ).setInteger( GL_EVAL_VERTEX_ATTRIB15_NV );
      self->addClassProperty( c_sdlglext, "MAX_MAP_TESSELLATION_NV" ).setInteger( GL_MAX_MAP_TESSELLATION_NV );
      self->addClassProperty( c_sdlglext, "MAX_RATIONAL_EVAL_ORDER_NV" ).setInteger( GL_MAX_RATIONAL_EVAL_ORDER_NV );
      #endif

      #ifndef GL_NV_packed_depth_stencil
      self->addClassProperty( c_sdlglext, "DEPTH_STENCIL_NV" ).setInteger( GL_DEPTH_STENCIL_NV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_24_8_NV" ).setInteger( GL_UNSIGNED_INT_24_8_NV );
      #endif

      #ifndef GL_NV_register_combiners2
      self->addClassProperty( c_sdlglext, "PER_STAGE_CONSTANTS_NV" ).setInteger( GL_PER_STAGE_CONSTANTS_NV );
      #endif

      #ifndef GL_NV_texture_rectangle
      self->addClassProperty( c_sdlglext, "TEXTURE_RECTANGLE_NV" ).setInteger( GL_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_BINDING_RECTANGLE_NV" ).setInteger( GL_TEXTURE_BINDING_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "PROXY_TEXTURE_RECTANGLE_NV" ).setInteger( GL_PROXY_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "MAX_RECTANGLE_TEXTURE_SIZE_NV" ).setInteger( GL_MAX_RECTANGLE_TEXTURE_SIZE_NV );
      #endif

      #ifndef GL_NV_texture_shader
      self->addClassProperty( c_sdlglext, "OFFSET_TEXTURE_RECTANGLE_NV" ).setInteger( GL_OFFSET_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_TEXTURE_RECTANGLE_SCALE_NV" ).setInteger( GL_OFFSET_TEXTURE_RECTANGLE_SCALE_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_TEXTURE_RECTANGLE_NV" ).setInteger( GL_DOT_PRODUCT_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV" ).setInteger( GL_RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_S8_S8_8_8_NV" ).setInteger( GL_UNSIGNED_INT_S8_S8_8_8_NV );
      self->addClassProperty( c_sdlglext, "UNSIGNED_INT_8_8_S8_S8_REV_NV" ).setInteger( GL_UNSIGNED_INT_8_8_S8_S8_REV_NV );
      self->addClassProperty( c_sdlglext, "DSDT_MAG_INTENSITY_NV" ).setInteger( GL_DSDT_MAG_INTENSITY_NV );
      self->addClassProperty( c_sdlglext, "SHADER_CONSISTENT_NV" ).setInteger( GL_SHADER_CONSISTENT_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_SHADER_NV" ).setInteger( GL_TEXTURE_SHADER_NV );
      self->addClassProperty( c_sdlglext, "SHADER_OPERATION_NV" ).setInteger( GL_SHADER_OPERATION_NV );
      self->addClassProperty( c_sdlglext, "CULL_MODES_NV" ).setInteger( GL_CULL_MODES_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_TEXTURE_MATRIX_NV" ).setInteger( GL_OFFSET_TEXTURE_MATRIX_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_TEXTURE_SCALE_NV" ).setInteger( GL_OFFSET_TEXTURE_SCALE_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_TEXTURE_BIAS_NV" ).setInteger( GL_OFFSET_TEXTURE_BIAS_NV );
      #define GL_OFFSET_TEXTURE_2D_MATRIX_NV    GL_OFFSET_TEXTURE_MATRIX_NV
      #define GL_OFFSET_TEXTURE_2D_SCALE_NV     GL_OFFSET_TEXTURE_SCALE_NV
      #define GL_OFFSET_TEXTURE_2D_BIAS_NV      GL_OFFSET_TEXTURE_BIAS_NV
      self->addClassProperty( c_sdlglext, "PREVIOUS_TEXTURE_INPUT_NV" ).setInteger( GL_PREVIOUS_TEXTURE_INPUT_NV );
      self->addClassProperty( c_sdlglext, "CONST_EYE_NV" ).setInteger( GL_CONST_EYE_NV );
      self->addClassProperty( c_sdlglext, "PASS_THROUGH_NV" ).setInteger( GL_PASS_THROUGH_NV );
      self->addClassProperty( c_sdlglext, "CULL_FRAGMENT_NV" ).setInteger( GL_CULL_FRAGMENT_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_TEXTURE_2D_NV" ).setInteger( GL_OFFSET_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "DEPENDENT_AR_TEXTURE_2D_NV" ).setInteger( GL_DEPENDENT_AR_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "DEPENDENT_GB_TEXTURE_2D_NV" ).setInteger( GL_DEPENDENT_GB_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_NV" ).setInteger( GL_DOT_PRODUCT_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_DEPTH_REPLACE_NV" ).setInteger( GL_DOT_PRODUCT_DEPTH_REPLACE_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_TEXTURE_2D_NV" ).setInteger( GL_DOT_PRODUCT_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_TEXTURE_CUBE_MAP_NV" ).setInteger( GL_DOT_PRODUCT_TEXTURE_CUBE_MAP_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_DIFFUSE_CUBE_MAP_NV" ).setInteger( GL_DOT_PRODUCT_DIFFUSE_CUBE_MAP_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_REFLECT_CUBE_MAP_NV" ).setInteger( GL_DOT_PRODUCT_REFLECT_CUBE_MAP_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_CONST_EYE_REFLECT_CUBE_MAP_NV" ).setInteger( GL_DOT_PRODUCT_CONST_EYE_REFLECT_CUBE_MAP_NV );
      self->addClassProperty( c_sdlglext, "HILO_NV" ).setInteger( GL_HILO_NV );
      self->addClassProperty( c_sdlglext, "DSDT_NV" ).setInteger( GL_DSDT_NV );
      self->addClassProperty( c_sdlglext, "DSDT_MAG_NV" ).setInteger( GL_DSDT_MAG_NV );
      self->addClassProperty( c_sdlglext, "DSDT_MAG_VIB_NV" ).setInteger( GL_DSDT_MAG_VIB_NV );
      self->addClassProperty( c_sdlglext, "HILO16_NV" ).setInteger( GL_HILO16_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_HILO_NV" ).setInteger( GL_SIGNED_HILO_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_HILO16_NV" ).setInteger( GL_SIGNED_HILO16_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_RGBA_NV" ).setInteger( GL_SIGNED_RGBA_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_RGBA8_NV" ).setInteger( GL_SIGNED_RGBA8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_RGB_NV" ).setInteger( GL_SIGNED_RGB_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_RGB8_NV" ).setInteger( GL_SIGNED_RGB8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_LUMINANCE_NV" ).setInteger( GL_SIGNED_LUMINANCE_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_LUMINANCE8_NV" ).setInteger( GL_SIGNED_LUMINANCE8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_LUMINANCE_ALPHA_NV" ).setInteger( GL_SIGNED_LUMINANCE_ALPHA_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_LUMINANCE8_ALPHA8_NV" ).setInteger( GL_SIGNED_LUMINANCE8_ALPHA8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_ALPHA_NV" ).setInteger( GL_SIGNED_ALPHA_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_ALPHA8_NV" ).setInteger( GL_SIGNED_ALPHA8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_INTENSITY_NV" ).setInteger( GL_SIGNED_INTENSITY_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_INTENSITY8_NV" ).setInteger( GL_SIGNED_INTENSITY8_NV );
      self->addClassProperty( c_sdlglext, "DSDT8_NV" ).setInteger( GL_DSDT8_NV );
      self->addClassProperty( c_sdlglext, "DSDT8_MAG8_NV" ).setInteger( GL_DSDT8_MAG8_NV );
      self->addClassProperty( c_sdlglext, "DSDT8_MAG8_INTENSITY8_NV" ).setInteger( GL_DSDT8_MAG8_INTENSITY8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_RGB_UNSIGNED_ALPHA_NV" ).setInteger( GL_SIGNED_RGB_UNSIGNED_ALPHA_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_RGB8_UNSIGNED_ALPHA8_NV" ).setInteger( GL_SIGNED_RGB8_UNSIGNED_ALPHA8_NV );
      self->addClassProperty( c_sdlglext, "HI_SCALE_NV" ).setInteger( GL_HI_SCALE_NV );
      self->addClassProperty( c_sdlglext, "LO_SCALE_NV" ).setInteger( GL_LO_SCALE_NV );
      self->addClassProperty( c_sdlglext, "DS_SCALE_NV" ).setInteger( GL_DS_SCALE_NV );
      self->addClassProperty( c_sdlglext, "DT_SCALE_NV" ).setInteger( GL_DT_SCALE_NV );
      self->addClassProperty( c_sdlglext, "MAGNITUDE_SCALE_NV" ).setInteger( GL_MAGNITUDE_SCALE_NV );
      self->addClassProperty( c_sdlglext, "VIBRANCE_SCALE_NV" ).setInteger( GL_VIBRANCE_SCALE_NV );
      self->addClassProperty( c_sdlglext, "HI_BIAS_NV" ).setInteger( GL_HI_BIAS_NV );
      self->addClassProperty( c_sdlglext, "LO_BIAS_NV" ).setInteger( GL_LO_BIAS_NV );
      self->addClassProperty( c_sdlglext, "DS_BIAS_NV" ).setInteger( GL_DS_BIAS_NV );
      self->addClassProperty( c_sdlglext, "DT_BIAS_NV" ).setInteger( GL_DT_BIAS_NV );
      self->addClassProperty( c_sdlglext, "MAGNITUDE_BIAS_NV" ).setInteger( GL_MAGNITUDE_BIAS_NV );
      self->addClassProperty( c_sdlglext, "VIBRANCE_BIAS_NV" ).setInteger( GL_VIBRANCE_BIAS_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_BORDER_VALUES_NV" ).setInteger( GL_TEXTURE_BORDER_VALUES_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_HI_SIZE_NV" ).setInteger( GL_TEXTURE_HI_SIZE_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_LO_SIZE_NV" ).setInteger( GL_TEXTURE_LO_SIZE_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_DS_SIZE_NV" ).setInteger( GL_TEXTURE_DS_SIZE_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_DT_SIZE_NV" ).setInteger( GL_TEXTURE_DT_SIZE_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAG_SIZE_NV" ).setInteger( GL_TEXTURE_MAG_SIZE_NV );
      #endif

      #ifndef GL_NV_texture_shader2
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_TEXTURE_3D_NV" ).setInteger( GL_DOT_PRODUCT_TEXTURE_3D_NV );
      #endif

      #ifndef GL_NV_vertex_array_range2
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_WITHOUT_FLUSH_NV" ).setInteger( GL_VERTEX_ARRAY_RANGE_WITHOUT_FLUSH_NV );
      #endif

      #ifndef GL_NV_vertex_program
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_NV" ).setInteger( GL_VERTEX_PROGRAM_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_STATE_PROGRAM_NV" ).setInteger( GL_VERTEX_STATE_PROGRAM_NV );
      self->addClassProperty( c_sdlglext, "ATTRIB_ARRAY_SIZE_NV" ).setInteger( GL_ATTRIB_ARRAY_SIZE_NV );
      self->addClassProperty( c_sdlglext, "ATTRIB_ARRAY_STRIDE_NV" ).setInteger( GL_ATTRIB_ARRAY_STRIDE_NV );
      self->addClassProperty( c_sdlglext, "ATTRIB_ARRAY_TYPE_NV" ).setInteger( GL_ATTRIB_ARRAY_TYPE_NV );
      self->addClassProperty( c_sdlglext, "CURRENT_ATTRIB_NV" ).setInteger( GL_CURRENT_ATTRIB_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_LENGTH_NV" ).setInteger( GL_PROGRAM_LENGTH_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_STRING_NV" ).setInteger( GL_PROGRAM_STRING_NV );
      self->addClassProperty( c_sdlglext, "MODELVIEW_PROJECTION_NV" ).setInteger( GL_MODELVIEW_PROJECTION_NV );
      self->addClassProperty( c_sdlglext, "IDENTITY_NV" ).setInteger( GL_IDENTITY_NV );
      self->addClassProperty( c_sdlglext, "INVERSE_NV" ).setInteger( GL_INVERSE_NV );
      self->addClassProperty( c_sdlglext, "TRANSPOSE_NV" ).setInteger( GL_TRANSPOSE_NV );
      self->addClassProperty( c_sdlglext, "INVERSE_TRANSPOSE_NV" ).setInteger( GL_INVERSE_TRANSPOSE_NV );
      self->addClassProperty( c_sdlglext, "MAX_TRACK_MATRIX_STACK_DEPTH_NV" ).setInteger( GL_MAX_TRACK_MATRIX_STACK_DEPTH_NV );
      self->addClassProperty( c_sdlglext, "MAX_TRACK_MATRICES_NV" ).setInteger( GL_MAX_TRACK_MATRICES_NV );
      self->addClassProperty( c_sdlglext, "MATRIX0_NV" ).setInteger( GL_MATRIX0_NV );
      self->addClassProperty( c_sdlglext, "MATRIX1_NV" ).setInteger( GL_MATRIX1_NV );
      self->addClassProperty( c_sdlglext, "MATRIX2_NV" ).setInteger( GL_MATRIX2_NV );
      self->addClassProperty( c_sdlglext, "MATRIX3_NV" ).setInteger( GL_MATRIX3_NV );
      self->addClassProperty( c_sdlglext, "MATRIX4_NV" ).setInteger( GL_MATRIX4_NV );
      self->addClassProperty( c_sdlglext, "MATRIX5_NV" ).setInteger( GL_MATRIX5_NV );
      self->addClassProperty( c_sdlglext, "MATRIX6_NV" ).setInteger( GL_MATRIX6_NV );
      self->addClassProperty( c_sdlglext, "MATRIX7_NV" ).setInteger( GL_MATRIX7_NV );
      self->addClassProperty( c_sdlglext, "CURRENT_MATRIX_STACK_DEPTH_NV" ).setInteger( GL_CURRENT_MATRIX_STACK_DEPTH_NV );
      self->addClassProperty( c_sdlglext, "CURRENT_MATRIX_NV" ).setInteger( GL_CURRENT_MATRIX_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_POINT_SIZE_NV" ).setInteger( GL_VERTEX_PROGRAM_POINT_SIZE_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_TWO_SIDE_NV" ).setInteger( GL_VERTEX_PROGRAM_TWO_SIDE_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_PARAMETER_NV" ).setInteger( GL_PROGRAM_PARAMETER_NV );
      self->addClassProperty( c_sdlglext, "ATTRIB_ARRAY_POINTER_NV" ).setInteger( GL_ATTRIB_ARRAY_POINTER_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_TARGET_NV" ).setInteger( GL_PROGRAM_TARGET_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_RESIDENT_NV" ).setInteger( GL_PROGRAM_RESIDENT_NV );
      self->addClassProperty( c_sdlglext, "TRACK_MATRIX_NV" ).setInteger( GL_TRACK_MATRIX_NV );
      self->addClassProperty( c_sdlglext, "TRACK_MATRIX_TRANSFORM_NV" ).setInteger( GL_TRACK_MATRIX_TRANSFORM_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_PROGRAM_BINDING_NV" ).setInteger( GL_VERTEX_PROGRAM_BINDING_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_ERROR_POSITION_NV" ).setInteger( GL_PROGRAM_ERROR_POSITION_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY0_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY0_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY1_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY1_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY2_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY2_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY3_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY3_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY4_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY4_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY5_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY5_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY6_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY6_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY7_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY7_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY8_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY8_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY9_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY9_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY10_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY10_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY11_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY11_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY12_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY12_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY13_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY13_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY14_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY14_NV );
      self->addClassProperty( c_sdlglext, "VERTEX_ATTRIB_ARRAY15_NV" ).setInteger( GL_VERTEX_ATTRIB_ARRAY15_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB0_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB0_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB1_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB1_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB2_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB2_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB3_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB3_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB4_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB4_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB5_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB5_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB6_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB6_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB7_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB7_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB8_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB8_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB9_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB9_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB10_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB10_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB11_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB11_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB12_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB12_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB13_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB13_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB14_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB14_4_NV );
      self->addClassProperty( c_sdlglext, "MAP1_VERTEX_ATTRIB15_4_NV" ).setInteger( GL_MAP1_VERTEX_ATTRIB15_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB0_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB0_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB1_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB1_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB2_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB2_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB3_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB3_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB4_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB4_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB5_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB5_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB6_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB6_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB7_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB7_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB8_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB8_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB9_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB9_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB10_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB10_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB11_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB11_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB12_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB12_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB13_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB13_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB14_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB14_4_NV );
      self->addClassProperty( c_sdlglext, "MAP2_VERTEX_ATTRIB15_4_NV" ).setInteger( GL_MAP2_VERTEX_ATTRIB15_4_NV );
      #endif

      #ifndef GL_SGIX_texture_coordinate_clamp
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_CLAMP_S_SGIX" ).setInteger( GL_TEXTURE_MAX_CLAMP_S_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_CLAMP_T_SGIX" ).setInteger( GL_TEXTURE_MAX_CLAMP_T_SGIX );
      self->addClassProperty( c_sdlglext, "TEXTURE_MAX_CLAMP_R_SGIX" ).setInteger( GL_TEXTURE_MAX_CLAMP_R_SGIX );
      #endif

      #ifndef GL_SGIX_scalebias_hint
      self->addClassProperty( c_sdlglext, "SCALEBIAS_HINT_SGIX" ).setInteger( GL_SCALEBIAS_HINT_SGIX );
      #endif

      #ifndef GL_OML_interlace
      self->addClassProperty( c_sdlglext, "INTERLACE_OML" ).setInteger( GL_INTERLACE_OML );
      self->addClassProperty( c_sdlglext, "INTERLACE_READ_OML" ).setInteger( GL_INTERLACE_READ_OML );
      #endif

      #ifndef GL_OML_subsample
      self->addClassProperty( c_sdlglext, "FORMAT_SUBSAMPLE_24_24_OML" ).setInteger( GL_FORMAT_SUBSAMPLE_24_24_OML );
      self->addClassProperty( c_sdlglext, "FORMAT_SUBSAMPLE_244_244_OML" ).setInteger( GL_FORMAT_SUBSAMPLE_244_244_OML );
      #endif

      #ifndef GL_OML_resample
      self->addClassProperty( c_sdlglext, "PACK_RESAMPLE_OML" ).setInteger( GL_PACK_RESAMPLE_OML );
      self->addClassProperty( c_sdlglext, "UNPACK_RESAMPLE_OML" ).setInteger( GL_UNPACK_RESAMPLE_OML );
      self->addClassProperty( c_sdlglext, "RESAMPLE_REPLICATE_OML" ).setInteger( GL_RESAMPLE_REPLICATE_OML );
      self->addClassProperty( c_sdlglext, "RESAMPLE_ZERO_FILL_OML" ).setInteger( GL_RESAMPLE_ZERO_FILL_OML );
      self->addClassProperty( c_sdlglext, "RESAMPLE_AVERAGE_OML" ).setInteger( GL_RESAMPLE_AVERAGE_OML );
      self->addClassProperty( c_sdlglext, "RESAMPLE_DECIMATE_OML" ).setInteger( GL_RESAMPLE_DECIMATE_OML );
      #endif

      #ifndef GL_NV_copy_depth_to_color
      self->addClassProperty( c_sdlglext, "DEPTH_STENCIL_TO_RGBA_NV" ).setInteger( GL_DEPTH_STENCIL_TO_RGBA_NV );
      self->addClassProperty( c_sdlglext, "DEPTH_STENCIL_TO_BGRA_NV" ).setInteger( GL_DEPTH_STENCIL_TO_BGRA_NV );
      #endif

      #ifndef GL_ATI_envmap_bumpmap
      self->addClassProperty( c_sdlglext, "BUMP_ROT_MATRIX_ATI" ).setInteger( GL_BUMP_ROT_MATRIX_ATI );
      self->addClassProperty( c_sdlglext, "BUMP_ROT_MATRIX_SIZE_ATI" ).setInteger( GL_BUMP_ROT_MATRIX_SIZE_ATI );
      self->addClassProperty( c_sdlglext, "BUMP_NUM_TEX_UNITS_ATI" ).setInteger( GL_BUMP_NUM_TEX_UNITS_ATI );
      self->addClassProperty( c_sdlglext, "BUMP_TEX_UNITS_ATI" ).setInteger( GL_BUMP_TEX_UNITS_ATI );
      self->addClassProperty( c_sdlglext, "DUDV_ATI" ).setInteger( GL_DUDV_ATI );
      self->addClassProperty( c_sdlglext, "DU8DV8_ATI" ).setInteger( GL_DU8DV8_ATI );
      self->addClassProperty( c_sdlglext, "BUMP_ENVMAP_ATI" ).setInteger( GL_BUMP_ENVMAP_ATI );
      self->addClassProperty( c_sdlglext, "BUMP_TARGET_ATI" ).setInteger( GL_BUMP_TARGET_ATI );
      #endif

      #ifndef GL_ATI_fragment_shader
      self->addClassProperty( c_sdlglext, "FRAGMENT_SHADER_ATI" ).setInteger( GL_FRAGMENT_SHADER_ATI );
      self->addClassProperty( c_sdlglext, "REG_0_ATI" ).setInteger( GL_REG_0_ATI );
      self->addClassProperty( c_sdlglext, "REG_1_ATI" ).setInteger( GL_REG_1_ATI );
      self->addClassProperty( c_sdlglext, "REG_2_ATI" ).setInteger( GL_REG_2_ATI );
      self->addClassProperty( c_sdlglext, "REG_3_ATI" ).setInteger( GL_REG_3_ATI );
      self->addClassProperty( c_sdlglext, "REG_4_ATI" ).setInteger( GL_REG_4_ATI );
      self->addClassProperty( c_sdlglext, "REG_5_ATI" ).setInteger( GL_REG_5_ATI );
      self->addClassProperty( c_sdlglext, "REG_6_ATI" ).setInteger( GL_REG_6_ATI );
      self->addClassProperty( c_sdlglext, "REG_7_ATI" ).setInteger( GL_REG_7_ATI );
      self->addClassProperty( c_sdlglext, "REG_8_ATI" ).setInteger( GL_REG_8_ATI );
      self->addClassProperty( c_sdlglext, "REG_9_ATI" ).setInteger( GL_REG_9_ATI );
      self->addClassProperty( c_sdlglext, "REG_10_ATI" ).setInteger( GL_REG_10_ATI );
      self->addClassProperty( c_sdlglext, "REG_11_ATI" ).setInteger( GL_REG_11_ATI );
      self->addClassProperty( c_sdlglext, "REG_12_ATI" ).setInteger( GL_REG_12_ATI );
      self->addClassProperty( c_sdlglext, "REG_13_ATI" ).setInteger( GL_REG_13_ATI );
      self->addClassProperty( c_sdlglext, "REG_14_ATI" ).setInteger( GL_REG_14_ATI );
      self->addClassProperty( c_sdlglext, "REG_15_ATI" ).setInteger( GL_REG_15_ATI );
      self->addClassProperty( c_sdlglext, "REG_16_ATI" ).setInteger( GL_REG_16_ATI );
      self->addClassProperty( c_sdlglext, "REG_17_ATI" ).setInteger( GL_REG_17_ATI );
      self->addClassProperty( c_sdlglext, "REG_18_ATI" ).setInteger( GL_REG_18_ATI );
      self->addClassProperty( c_sdlglext, "REG_19_ATI" ).setInteger( GL_REG_19_ATI );
      self->addClassProperty( c_sdlglext, "REG_20_ATI" ).setInteger( GL_REG_20_ATI );
      self->addClassProperty( c_sdlglext, "REG_21_ATI" ).setInteger( GL_REG_21_ATI );
      self->addClassProperty( c_sdlglext, "REG_22_ATI" ).setInteger( GL_REG_22_ATI );
      self->addClassProperty( c_sdlglext, "REG_23_ATI" ).setInteger( GL_REG_23_ATI );
      self->addClassProperty( c_sdlglext, "REG_24_ATI" ).setInteger( GL_REG_24_ATI );
      self->addClassProperty( c_sdlglext, "REG_25_ATI" ).setInteger( GL_REG_25_ATI );
      self->addClassProperty( c_sdlglext, "REG_26_ATI" ).setInteger( GL_REG_26_ATI );
      self->addClassProperty( c_sdlglext, "REG_27_ATI" ).setInteger( GL_REG_27_ATI );
      self->addClassProperty( c_sdlglext, "REG_28_ATI" ).setInteger( GL_REG_28_ATI );
      self->addClassProperty( c_sdlglext, "REG_29_ATI" ).setInteger( GL_REG_29_ATI );
      self->addClassProperty( c_sdlglext, "REG_30_ATI" ).setInteger( GL_REG_30_ATI );
      self->addClassProperty( c_sdlglext, "REG_31_ATI" ).setInteger( GL_REG_31_ATI );
      self->addClassProperty( c_sdlglext, "CON_0_ATI" ).setInteger( GL_CON_0_ATI );
      self->addClassProperty( c_sdlglext, "CON_1_ATI" ).setInteger( GL_CON_1_ATI );
      self->addClassProperty( c_sdlglext, "CON_2_ATI" ).setInteger( GL_CON_2_ATI );
      self->addClassProperty( c_sdlglext, "CON_3_ATI" ).setInteger( GL_CON_3_ATI );
      self->addClassProperty( c_sdlglext, "CON_4_ATI" ).setInteger( GL_CON_4_ATI );
      self->addClassProperty( c_sdlglext, "CON_5_ATI" ).setInteger( GL_CON_5_ATI );
      self->addClassProperty( c_sdlglext, "CON_6_ATI" ).setInteger( GL_CON_6_ATI );
      self->addClassProperty( c_sdlglext, "CON_7_ATI" ).setInteger( GL_CON_7_ATI );
      self->addClassProperty( c_sdlglext, "CON_8_ATI" ).setInteger( GL_CON_8_ATI );
      self->addClassProperty( c_sdlglext, "CON_9_ATI" ).setInteger( GL_CON_9_ATI );
      self->addClassProperty( c_sdlglext, "CON_10_ATI" ).setInteger( GL_CON_10_ATI );
      self->addClassProperty( c_sdlglext, "CON_11_ATI" ).setInteger( GL_CON_11_ATI );
      self->addClassProperty( c_sdlglext, "CON_12_ATI" ).setInteger( GL_CON_12_ATI );
      self->addClassProperty( c_sdlglext, "CON_13_ATI" ).setInteger( GL_CON_13_ATI );
      self->addClassProperty( c_sdlglext, "CON_14_ATI" ).setInteger( GL_CON_14_ATI );
      self->addClassProperty( c_sdlglext, "CON_15_ATI" ).setInteger( GL_CON_15_ATI );
      self->addClassProperty( c_sdlglext, "CON_16_ATI" ).setInteger( GL_CON_16_ATI );
      self->addClassProperty( c_sdlglext, "CON_17_ATI" ).setInteger( GL_CON_17_ATI );
      self->addClassProperty( c_sdlglext, "CON_18_ATI" ).setInteger( GL_CON_18_ATI );
      self->addClassProperty( c_sdlglext, "CON_19_ATI" ).setInteger( GL_CON_19_ATI );
      self->addClassProperty( c_sdlglext, "CON_20_ATI" ).setInteger( GL_CON_20_ATI );
      self->addClassProperty( c_sdlglext, "CON_21_ATI" ).setInteger( GL_CON_21_ATI );
      self->addClassProperty( c_sdlglext, "CON_22_ATI" ).setInteger( GL_CON_22_ATI );
      self->addClassProperty( c_sdlglext, "CON_23_ATI" ).setInteger( GL_CON_23_ATI );
      self->addClassProperty( c_sdlglext, "CON_24_ATI" ).setInteger( GL_CON_24_ATI );
      self->addClassProperty( c_sdlglext, "CON_25_ATI" ).setInteger( GL_CON_25_ATI );
      self->addClassProperty( c_sdlglext, "CON_26_ATI" ).setInteger( GL_CON_26_ATI );
      self->addClassProperty( c_sdlglext, "CON_27_ATI" ).setInteger( GL_CON_27_ATI );
      self->addClassProperty( c_sdlglext, "CON_28_ATI" ).setInteger( GL_CON_28_ATI );
      self->addClassProperty( c_sdlglext, "CON_29_ATI" ).setInteger( GL_CON_29_ATI );
      self->addClassProperty( c_sdlglext, "CON_30_ATI" ).setInteger( GL_CON_30_ATI );
      self->addClassProperty( c_sdlglext, "CON_31_ATI" ).setInteger( GL_CON_31_ATI );
      self->addClassProperty( c_sdlglext, "MOV_ATI" ).setInteger( GL_MOV_ATI );
      self->addClassProperty( c_sdlglext, "ADD_ATI" ).setInteger( GL_ADD_ATI );
      self->addClassProperty( c_sdlglext, "MUL_ATI" ).setInteger( GL_MUL_ATI );
      self->addClassProperty( c_sdlglext, "SUB_ATI" ).setInteger( GL_SUB_ATI );
      self->addClassProperty( c_sdlglext, "DOT3_ATI" ).setInteger( GL_DOT3_ATI );
      self->addClassProperty( c_sdlglext, "DOT4_ATI" ).setInteger( GL_DOT4_ATI );
      self->addClassProperty( c_sdlglext, "MAD_ATI" ).setInteger( GL_MAD_ATI );
      self->addClassProperty( c_sdlglext, "LERP_ATI" ).setInteger( GL_LERP_ATI );
      self->addClassProperty( c_sdlglext, "CND_ATI" ).setInteger( GL_CND_ATI );
      self->addClassProperty( c_sdlglext, "CND0_ATI" ).setInteger( GL_CND0_ATI );
      self->addClassProperty( c_sdlglext, "DOT2_ADD_ATI" ).setInteger( GL_DOT2_ADD_ATI );
      self->addClassProperty( c_sdlglext, "SECONDARY_INTERPOLATOR_ATI" ).setInteger( GL_SECONDARY_INTERPOLATOR_ATI );
      self->addClassProperty( c_sdlglext, "NUM_FRAGMENT_REGISTERS_ATI" ).setInteger( GL_NUM_FRAGMENT_REGISTERS_ATI );
      self->addClassProperty( c_sdlglext, "NUM_FRAGMENT_CONSTANTS_ATI" ).setInteger( GL_NUM_FRAGMENT_CONSTANTS_ATI );
      self->addClassProperty( c_sdlglext, "NUM_PASSES_ATI" ).setInteger( GL_NUM_PASSES_ATI );
      self->addClassProperty( c_sdlglext, "NUM_INSTRUCTIONS_PER_PASS_ATI" ).setInteger( GL_NUM_INSTRUCTIONS_PER_PASS_ATI );
      self->addClassProperty( c_sdlglext, "NUM_INSTRUCTIONS_TOTAL_ATI" ).setInteger( GL_NUM_INSTRUCTIONS_TOTAL_ATI );
      self->addClassProperty( c_sdlglext, "NUM_INPUT_INTERPOLATOR_COMPONENTS_ATI" ).setInteger( GL_NUM_INPUT_INTERPOLATOR_COMPONENTS_ATI );
      self->addClassProperty( c_sdlglext, "NUM_LOOPBACK_COMPONENTS_ATI" ).setInteger( GL_NUM_LOOPBACK_COMPONENTS_ATI );
      self->addClassProperty( c_sdlglext, "COLOR_ALPHA_PAIRING_ATI" ).setInteger( GL_COLOR_ALPHA_PAIRING_ATI );
      self->addClassProperty( c_sdlglext, "SWIZZLE_STR_ATI" ).setInteger( GL_SWIZZLE_STR_ATI );
      self->addClassProperty( c_sdlglext, "SWIZZLE_STQ_ATI" ).setInteger( GL_SWIZZLE_STQ_ATI );
      self->addClassProperty( c_sdlglext, "SWIZZLE_STR_DR_ATI" ).setInteger( GL_SWIZZLE_STR_DR_ATI );
      self->addClassProperty( c_sdlglext, "SWIZZLE_STQ_DQ_ATI" ).setInteger( GL_SWIZZLE_STQ_DQ_ATI );
      self->addClassProperty( c_sdlglext, "SWIZZLE_STRQ_ATI" ).setInteger( GL_SWIZZLE_STRQ_ATI );
      self->addClassProperty( c_sdlglext, "SWIZZLE_STRQ_DQ_ATI" ).setInteger( GL_SWIZZLE_STRQ_DQ_ATI );
      self->addClassProperty( c_sdlglext, "RED_BIT_ATI" ).setInteger( GL_RED_BIT_ATI );
      self->addClassProperty( c_sdlglext, "GREEN_BIT_ATI" ).setInteger( GL_GREEN_BIT_ATI );
      self->addClassProperty( c_sdlglext, "BLUE_BIT_ATI" ).setInteger( GL_BLUE_BIT_ATI );
      self->addClassProperty( c_sdlglext, "2X_BIT_ATI" ).setInteger( GL_2X_BIT_ATI );
      self->addClassProperty( c_sdlglext, "4X_BIT_ATI" ).setInteger( GL_4X_BIT_ATI );
      self->addClassProperty( c_sdlglext, "8X_BIT_ATI" ).setInteger( GL_8X_BIT_ATI );
      self->addClassProperty( c_sdlglext, "HALF_BIT_ATI" ).setInteger( GL_HALF_BIT_ATI );
      self->addClassProperty( c_sdlglext, "QUARTER_BIT_ATI" ).setInteger( GL_QUARTER_BIT_ATI );
      self->addClassProperty( c_sdlglext, "EIGHTH_BIT_ATI" ).setInteger( GL_EIGHTH_BIT_ATI );
      self->addClassProperty( c_sdlglext, "SATURATE_BIT_ATI" ).setInteger( GL_SATURATE_BIT_ATI );
      self->addClassProperty( c_sdlglext, "COMP_BIT_ATI" ).setInteger( GL_COMP_BIT_ATI );
      self->addClassProperty( c_sdlglext, "NEGATE_BIT_ATI" ).setInteger( GL_NEGATE_BIT_ATI );
      self->addClassProperty( c_sdlglext, "BIAS_BIT_ATI" ).setInteger( GL_BIAS_BIT_ATI );
      #endif

      #ifndef GL_ATI_pn_triangles
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_ATI" ).setInteger( GL_PN_TRIANGLES_ATI );
      self->addClassProperty( c_sdlglext, "MAX_PN_TRIANGLES_TESSELATION_LEVEL_ATI" ).setInteger( GL_MAX_PN_TRIANGLES_TESSELATION_LEVEL_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_POINT_MODE_ATI" ).setInteger( GL_PN_TRIANGLES_POINT_MODE_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_NORMAL_MODE_ATI" ).setInteger( GL_PN_TRIANGLES_NORMAL_MODE_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_TESSELATION_LEVEL_ATI" ).setInteger( GL_PN_TRIANGLES_TESSELATION_LEVEL_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_POINT_MODE_LINEAR_ATI" ).setInteger( GL_PN_TRIANGLES_POINT_MODE_LINEAR_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_POINT_MODE_CUBIC_ATI" ).setInteger( GL_PN_TRIANGLES_POINT_MODE_CUBIC_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_NORMAL_MODE_LINEAR_ATI" ).setInteger( GL_PN_TRIANGLES_NORMAL_MODE_LINEAR_ATI );
      self->addClassProperty( c_sdlglext, "PN_TRIANGLES_NORMAL_MODE_QUADRATIC_ATI" ).setInteger( GL_PN_TRIANGLES_NORMAL_MODE_QUADRATIC_ATI );
      #endif

      #ifndef GL_ATI_vertex_array_object
      self->addClassProperty( c_sdlglext, "STATIC_ATI" ).setInteger( GL_STATIC_ATI );
      self->addClassProperty( c_sdlglext, "DYNAMIC_ATI" ).setInteger( GL_DYNAMIC_ATI );
      self->addClassProperty( c_sdlglext, "PRESERVE_ATI" ).setInteger( GL_PRESERVE_ATI );
      self->addClassProperty( c_sdlglext, "DISCARD_ATI" ).setInteger( GL_DISCARD_ATI );
      self->addClassProperty( c_sdlglext, "OBJECT_BUFFER_SIZE_ATI" ).setInteger( GL_OBJECT_BUFFER_SIZE_ATI );
      self->addClassProperty( c_sdlglext, "OBJECT_BUFFER_USAGE_ATI" ).setInteger( GL_OBJECT_BUFFER_USAGE_ATI );
      self->addClassProperty( c_sdlglext, "ARRAY_OBJECT_BUFFER_ATI" ).setInteger( GL_ARRAY_OBJECT_BUFFER_ATI );
      self->addClassProperty( c_sdlglext, "ARRAY_OBJECT_OFFSET_ATI" ).setInteger( GL_ARRAY_OBJECT_OFFSET_ATI );
      #endif

      #ifndef GL_EXT_vertex_shader
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_EXT" ).setInteger( GL_VERTEX_SHADER_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_BINDING_EXT" ).setInteger( GL_VERTEX_SHADER_BINDING_EXT );
      self->addClassProperty( c_sdlglext, "OP_INDEX_EXT" ).setInteger( GL_OP_INDEX_EXT );
      self->addClassProperty( c_sdlglext, "OP_NEGATE_EXT" ).setInteger( GL_OP_NEGATE_EXT );
      self->addClassProperty( c_sdlglext, "OP_DOT3_EXT" ).setInteger( GL_OP_DOT3_EXT );
      self->addClassProperty( c_sdlglext, "OP_DOT4_EXT" ).setInteger( GL_OP_DOT4_EXT );
      self->addClassProperty( c_sdlglext, "OP_MUL_EXT" ).setInteger( GL_OP_MUL_EXT );
      self->addClassProperty( c_sdlglext, "OP_ADD_EXT" ).setInteger( GL_OP_ADD_EXT );
      self->addClassProperty( c_sdlglext, "OP_MADD_EXT" ).setInteger( GL_OP_MADD_EXT );
      self->addClassProperty( c_sdlglext, "OP_FRAC_EXT" ).setInteger( GL_OP_FRAC_EXT );
      self->addClassProperty( c_sdlglext, "OP_MAX_EXT" ).setInteger( GL_OP_MAX_EXT );
      self->addClassProperty( c_sdlglext, "OP_MIN_EXT" ).setInteger( GL_OP_MIN_EXT );
      self->addClassProperty( c_sdlglext, "OP_SET_GE_EXT" ).setInteger( GL_OP_SET_GE_EXT );
      self->addClassProperty( c_sdlglext, "OP_SET_LT_EXT" ).setInteger( GL_OP_SET_LT_EXT );
      self->addClassProperty( c_sdlglext, "OP_CLAMP_EXT" ).setInteger( GL_OP_CLAMP_EXT );
      self->addClassProperty( c_sdlglext, "OP_FLOOR_EXT" ).setInteger( GL_OP_FLOOR_EXT );
      self->addClassProperty( c_sdlglext, "OP_ROUND_EXT" ).setInteger( GL_OP_ROUND_EXT );
      self->addClassProperty( c_sdlglext, "OP_EXP_BASE_2_EXT" ).setInteger( GL_OP_EXP_BASE_2_EXT );
      self->addClassProperty( c_sdlglext, "OP_LOG_BASE_2_EXT" ).setInteger( GL_OP_LOG_BASE_2_EXT );
      self->addClassProperty( c_sdlglext, "OP_POWER_EXT" ).setInteger( GL_OP_POWER_EXT );
      self->addClassProperty( c_sdlglext, "OP_RECIP_EXT" ).setInteger( GL_OP_RECIP_EXT );
      self->addClassProperty( c_sdlglext, "OP_RECIP_SQRT_EXT" ).setInteger( GL_OP_RECIP_SQRT_EXT );
      self->addClassProperty( c_sdlglext, "OP_SUB_EXT" ).setInteger( GL_OP_SUB_EXT );
      self->addClassProperty( c_sdlglext, "OP_CROSS_PRODUCT_EXT" ).setInteger( GL_OP_CROSS_PRODUCT_EXT );
      self->addClassProperty( c_sdlglext, "OP_MULTIPLY_MATRIX_EXT" ).setInteger( GL_OP_MULTIPLY_MATRIX_EXT );
      self->addClassProperty( c_sdlglext, "OP_MOV_EXT" ).setInteger( GL_OP_MOV_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_VERTEX_EXT" ).setInteger( GL_OUTPUT_VERTEX_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_COLOR0_EXT" ).setInteger( GL_OUTPUT_COLOR0_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_COLOR1_EXT" ).setInteger( GL_OUTPUT_COLOR1_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD0_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD0_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD1_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD1_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD2_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD2_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD3_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD3_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD4_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD4_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD5_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD5_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD6_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD6_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD7_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD7_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD8_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD8_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD9_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD9_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD10_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD10_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD11_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD11_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD12_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD12_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD13_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD13_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD14_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD14_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD15_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD15_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD16_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD16_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD17_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD17_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD18_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD18_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD19_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD19_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD20_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD20_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD21_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD21_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD22_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD22_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD23_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD23_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD24_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD24_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD25_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD25_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD26_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD26_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD27_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD27_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD28_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD28_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD29_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD29_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD30_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD30_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_TEXTURE_COORD31_EXT" ).setInteger( GL_OUTPUT_TEXTURE_COORD31_EXT );
      self->addClassProperty( c_sdlglext, "OUTPUT_FOG_EXT" ).setInteger( GL_OUTPUT_FOG_EXT );
      self->addClassProperty( c_sdlglext, "SCALAR_EXT" ).setInteger( GL_SCALAR_EXT );
      self->addClassProperty( c_sdlglext, "VECTOR_EXT" ).setInteger( GL_VECTOR_EXT );
      self->addClassProperty( c_sdlglext, "MATRIX_EXT" ).setInteger( GL_MATRIX_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_EXT" ).setInteger( GL_VARIANT_EXT );
      self->addClassProperty( c_sdlglext, "INVARIANT_EXT" ).setInteger( GL_INVARIANT_EXT );
      self->addClassProperty( c_sdlglext, "LOCAL_CONSTANT_EXT" ).setInteger( GL_LOCAL_CONSTANT_EXT );
      self->addClassProperty( c_sdlglext, "LOCAL_EXT" ).setInteger( GL_LOCAL_EXT );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_SHADER_INSTRUCTIONS_EXT" ).setInteger( GL_MAX_VERTEX_SHADER_INSTRUCTIONS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_SHADER_VARIANTS_EXT" ).setInteger( GL_MAX_VERTEX_SHADER_VARIANTS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_SHADER_INVARIANTS_EXT" ).setInteger( GL_MAX_VERTEX_SHADER_INVARIANTS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_SHADER_LOCAL_CONSTANTS_EXT" ).setInteger( GL_MAX_VERTEX_SHADER_LOCAL_CONSTANTS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_SHADER_LOCALS_EXT" ).setInteger( GL_MAX_VERTEX_SHADER_LOCALS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_OPTIMIZED_VERTEX_SHADER_INSTRUCTIONS_EXT" ).setInteger( GL_MAX_OPTIMIZED_VERTEX_SHADER_INSTRUCTIONS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_OPTIMIZED_VERTEX_SHADER_VARIANTS_EXT" ).setInteger( GL_MAX_OPTIMIZED_VERTEX_SHADER_VARIANTS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_OPTIMIZED_VERTEX_SHADER_LOCAL_CONSTANTS_EXT" ).setInteger( GL_MAX_OPTIMIZED_VERTEX_SHADER_LOCAL_CONSTANTS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_OPTIMIZED_VERTEX_SHADER_INVARIANTS_EXT" ).setInteger( GL_MAX_OPTIMIZED_VERTEX_SHADER_INVARIANTS_EXT );
      self->addClassProperty( c_sdlglext, "MAX_OPTIMIZED_VERTEX_SHADER_LOCALS_EXT" ).setInteger( GL_MAX_OPTIMIZED_VERTEX_SHADER_LOCALS_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_INSTRUCTIONS_EXT" ).setInteger( GL_VERTEX_SHADER_INSTRUCTIONS_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_VARIANTS_EXT" ).setInteger( GL_VERTEX_SHADER_VARIANTS_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_INVARIANTS_EXT" ).setInteger( GL_VERTEX_SHADER_INVARIANTS_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_LOCAL_CONSTANTS_EXT" ).setInteger( GL_VERTEX_SHADER_LOCAL_CONSTANTS_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_LOCALS_EXT" ).setInteger( GL_VERTEX_SHADER_LOCALS_EXT );
      self->addClassProperty( c_sdlglext, "VERTEX_SHADER_OPTIMIZED_EXT" ).setInteger( GL_VERTEX_SHADER_OPTIMIZED_EXT );
      self->addClassProperty( c_sdlglext, "X_EXT" ).setInteger( GL_X_EXT );
      self->addClassProperty( c_sdlglext, "Y_EXT" ).setInteger( GL_Y_EXT );
      self->addClassProperty( c_sdlglext, "Z_EXT" ).setInteger( GL_Z_EXT );
      self->addClassProperty( c_sdlglext, "W_EXT" ).setInteger( GL_W_EXT );
      self->addClassProperty( c_sdlglext, "NEGATIVE_X_EXT" ).setInteger( GL_NEGATIVE_X_EXT );
      self->addClassProperty( c_sdlglext, "NEGATIVE_Y_EXT" ).setInteger( GL_NEGATIVE_Y_EXT );
      self->addClassProperty( c_sdlglext, "NEGATIVE_Z_EXT" ).setInteger( GL_NEGATIVE_Z_EXT );
      self->addClassProperty( c_sdlglext, "NEGATIVE_W_EXT" ).setInteger( GL_NEGATIVE_W_EXT );
      self->addClassProperty( c_sdlglext, "ZERO_EXT" ).setInteger( GL_ZERO_EXT );
      self->addClassProperty( c_sdlglext, "ONE_EXT" ).setInteger( GL_ONE_EXT );
      self->addClassProperty( c_sdlglext, "NEGATIVE_ONE_EXT" ).setInteger( GL_NEGATIVE_ONE_EXT );
      self->addClassProperty( c_sdlglext, "NORMALIZED_RANGE_EXT" ).setInteger( GL_NORMALIZED_RANGE_EXT );
      self->addClassProperty( c_sdlglext, "FULL_RANGE_EXT" ).setInteger( GL_FULL_RANGE_EXT );
      self->addClassProperty( c_sdlglext, "CURRENT_VERTEX_EXT" ).setInteger( GL_CURRENT_VERTEX_EXT );
      self->addClassProperty( c_sdlglext, "MVP_MATRIX_EXT" ).setInteger( GL_MVP_MATRIX_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_VALUE_EXT" ).setInteger( GL_VARIANT_VALUE_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_DATATYPE_EXT" ).setInteger( GL_VARIANT_DATATYPE_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_ARRAY_STRIDE_EXT" ).setInteger( GL_VARIANT_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_ARRAY_TYPE_EXT" ).setInteger( GL_VARIANT_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_ARRAY_EXT" ).setInteger( GL_VARIANT_ARRAY_EXT );
      self->addClassProperty( c_sdlglext, "VARIANT_ARRAY_POINTER_EXT" ).setInteger( GL_VARIANT_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlglext, "INVARIANT_VALUE_EXT" ).setInteger( GL_INVARIANT_VALUE_EXT );
      self->addClassProperty( c_sdlglext, "INVARIANT_DATATYPE_EXT" ).setInteger( GL_INVARIANT_DATATYPE_EXT );
      self->addClassProperty( c_sdlglext, "LOCAL_CONSTANT_VALUE_EXT" ).setInteger( GL_LOCAL_CONSTANT_VALUE_EXT );
      self->addClassProperty( c_sdlglext, "LOCAL_CONSTANT_DATATYPE_EXT" ).setInteger( GL_LOCAL_CONSTANT_DATATYPE_EXT );
      #endif

      #ifndef GL_ATI_vertex_streams
      self->addClassProperty( c_sdlglext, "MAX_VERTEX_STREAMS_ATI" ).setInteger( GL_MAX_VERTEX_STREAMS_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM0_ATI" ).setInteger( GL_VERTEX_STREAM0_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM1_ATI" ).setInteger( GL_VERTEX_STREAM1_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM2_ATI" ).setInteger( GL_VERTEX_STREAM2_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM3_ATI" ).setInteger( GL_VERTEX_STREAM3_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM4_ATI" ).setInteger( GL_VERTEX_STREAM4_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM5_ATI" ).setInteger( GL_VERTEX_STREAM5_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM6_ATI" ).setInteger( GL_VERTEX_STREAM6_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_STREAM7_ATI" ).setInteger( GL_VERTEX_STREAM7_ATI );
      self->addClassProperty( c_sdlglext, "VERTEX_SOURCE_ATI" ).setInteger( GL_VERTEX_SOURCE_ATI );
      #endif

      #ifndef GL_ATI_element_array
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_ATI" ).setInteger( GL_ELEMENT_ARRAY_ATI );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_TYPE_ATI" ).setInteger( GL_ELEMENT_ARRAY_TYPE_ATI );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_POINTER_ATI" ).setInteger( GL_ELEMENT_ARRAY_POINTER_ATI );
      #endif

      #ifndef GL_SUN_mesh_array
      self->addClassProperty( c_sdlglext, "QUAD_MESH_SUN" ).setInteger( GL_QUAD_MESH_SUN );
      self->addClassProperty( c_sdlglext, "TRIANGLE_MESH_SUN" ).setInteger( GL_TRIANGLE_MESH_SUN );
      #endif

      #ifndef GL_SUN_slice_accum
      self->addClassProperty( c_sdlglext, "SLICE_ACCUM_SUN" ).setInteger( GL_SLICE_ACCUM_SUN );
      #endif

      #ifndef GL_NV_multisample_filter_hint
      self->addClassProperty( c_sdlglext, "MULTISAMPLE_FILTER_HINT_NV" ).setInteger( GL_MULTISAMPLE_FILTER_HINT_NV );
      #endif

      #ifndef GL_NV_depth_clamp
      self->addClassProperty( c_sdlglext, "DEPTH_CLAMP_NV" ).setInteger( GL_DEPTH_CLAMP_NV );
      #endif

      #ifndef GL_NV_occlusion_query
      self->addClassProperty( c_sdlglext, "PIXEL_COUNTER_BITS_NV" ).setInteger( GL_PIXEL_COUNTER_BITS_NV );
      self->addClassProperty( c_sdlglext, "CURRENT_OCCLUSION_QUERY_ID_NV" ).setInteger( GL_CURRENT_OCCLUSION_QUERY_ID_NV );
      self->addClassProperty( c_sdlglext, "PIXEL_COUNT_NV" ).setInteger( GL_PIXEL_COUNT_NV );
      self->addClassProperty( c_sdlglext, "PIXEL_COUNT_AVAILABLE_NV" ).setInteger( GL_PIXEL_COUNT_AVAILABLE_NV );
      #endif

      #ifndef GL_NV_point_sprite
      self->addClassProperty( c_sdlglext, "POINT_SPRITE_NV" ).setInteger( GL_POINT_SPRITE_NV );
      self->addClassProperty( c_sdlglext, "COORD_REPLACE_NV" ).setInteger( GL_COORD_REPLACE_NV );
      self->addClassProperty( c_sdlglext, "POINT_SPRITE_R_MODE_NV" ).setInteger( GL_POINT_SPRITE_R_MODE_NV );
      #endif

      #ifndef GL_NV_texture_shader3
      self->addClassProperty( c_sdlglext, "OFFSET_PROJECTIVE_TEXTURE_2D_NV" ).setInteger( GL_OFFSET_PROJECTIVE_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_PROJECTIVE_TEXTURE_2D_SCALE_NV" ).setInteger( GL_OFFSET_PROJECTIVE_TEXTURE_2D_SCALE_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_NV" ).setInteger( GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_SCALE_NV" ).setInteger( GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_SCALE_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_HILO_TEXTURE_2D_NV" ).setInteger( GL_OFFSET_HILO_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_HILO_TEXTURE_RECTANGLE_NV" ).setInteger( GL_OFFSET_HILO_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_HILO_PROJECTIVE_TEXTURE_2D_NV" ).setInteger( GL_OFFSET_HILO_PROJECTIVE_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "OFFSET_HILO_PROJECTIVE_TEXTURE_RECTANGLE_NV" ).setInteger( GL_OFFSET_HILO_PROJECTIVE_TEXTURE_RECTANGLE_NV );
      self->addClassProperty( c_sdlglext, "DEPENDENT_HILO_TEXTURE_2D_NV" ).setInteger( GL_DEPENDENT_HILO_TEXTURE_2D_NV );
      self->addClassProperty( c_sdlglext, "DEPENDENT_RGB_TEXTURE_3D_NV" ).setInteger( GL_DEPENDENT_RGB_TEXTURE_3D_NV );
      self->addClassProperty( c_sdlglext, "DEPENDENT_RGB_TEXTURE_CUBE_MAP_NV" ).setInteger( GL_DEPENDENT_RGB_TEXTURE_CUBE_MAP_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_PASS_THROUGH_NV" ).setInteger( GL_DOT_PRODUCT_PASS_THROUGH_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_TEXTURE_1D_NV" ).setInteger( GL_DOT_PRODUCT_TEXTURE_1D_NV );
      self->addClassProperty( c_sdlglext, "DOT_PRODUCT_AFFINE_DEPTH_REPLACE_NV" ).setInteger( GL_DOT_PRODUCT_AFFINE_DEPTH_REPLACE_NV );
      self->addClassProperty( c_sdlglext, "HILO8_NV" ).setInteger( GL_HILO8_NV );
      self->addClassProperty( c_sdlglext, "SIGNED_HILO8_NV" ).setInteger( GL_SIGNED_HILO8_NV );
      self->addClassProperty( c_sdlglext, "FORCE_BLUE_TO_ONE_NV" ).setInteger( GL_FORCE_BLUE_TO_ONE_NV );
      #endif

      #ifndef GL_EXT_stencil_two_side
      self->addClassProperty( c_sdlglext, "STENCIL_TEST_TWO_SIDE_EXT" ).setInteger( GL_STENCIL_TEST_TWO_SIDE_EXT );
      self->addClassProperty( c_sdlglext, "ACTIVE_STENCIL_FACE_EXT" ).setInteger( GL_ACTIVE_STENCIL_FACE_EXT );
      #endif

      #ifndef GL_ATI_text_fragment_shader
      self->addClassProperty( c_sdlglext, "TEXT_FRAGMENT_SHADER_ATI" ).setInteger( GL_TEXT_FRAGMENT_SHADER_ATI );
      #endif

      #ifndef GL_APPLE_client_storage
      self->addClassProperty( c_sdlglext, "UNPACK_CLIENT_STORAGE_APPLE" ).setInteger( GL_UNPACK_CLIENT_STORAGE_APPLE );
      #endif

      #ifndef GL_APPLE_element_array
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_APPLE" ).setInteger( GL_ELEMENT_ARRAY_APPLE );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_TYPE_APPLE" ).setInteger( GL_ELEMENT_ARRAY_TYPE_APPLE );
      self->addClassProperty( c_sdlglext, "ELEMENT_ARRAY_POINTER_APPLE" ).setInteger( GL_ELEMENT_ARRAY_POINTER_APPLE );
      #endif

      #ifndef GL_APPLE_fence
      self->addClassProperty( c_sdlglext, "DRAW_PIXELS_APPLE" ).setInteger( GL_DRAW_PIXELS_APPLE );
      self->addClassProperty( c_sdlglext, "FENCE_APPLE" ).setInteger( GL_FENCE_APPLE );
      #endif

      #ifndef GL_APPLE_vertex_array_object
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_BINDING_APPLE" ).setInteger( GL_VERTEX_ARRAY_BINDING_APPLE );
      #endif

      #ifndef GL_APPLE_vertex_array_range
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_APPLE" ).setInteger( GL_VERTEX_ARRAY_RANGE_APPLE );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_LENGTH_APPLE" ).setInteger( GL_VERTEX_ARRAY_RANGE_LENGTH_APPLE );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_STORAGE_HINT_APPLE" ).setInteger( GL_VERTEX_ARRAY_STORAGE_HINT_APPLE );
      self->addClassProperty( c_sdlglext, "VERTEX_ARRAY_RANGE_POINTER_APPLE" ).setInteger( GL_VERTEX_ARRAY_RANGE_POINTER_APPLE );
      self->addClassProperty( c_sdlglext, "STORAGE_CACHED_APPLE" ).setInteger( GL_STORAGE_CACHED_APPLE );
      self->addClassProperty( c_sdlglext, "STORAGE_SHARED_APPLE" ).setInteger( GL_STORAGE_SHARED_APPLE );
      #endif

      #ifndef GL_APPLE_ycbcr_422
      self->addClassProperty( c_sdlglext, "YCBCR_422_APPLE" ).setInteger( GL_YCBCR_422_APPLE );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_8_8_APPLE" ).setInteger( GL_UNSIGNED_SHORT_8_8_APPLE );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_8_8_REV_APPLE" ).setInteger( GL_UNSIGNED_SHORT_8_8_REV_APPLE );
      #endif

      #ifndef GL_S3_s3tc
      self->addClassProperty( c_sdlglext, "RGB_S3TC" ).setInteger( GL_RGB_S3TC );
      self->addClassProperty( c_sdlglext, "RGB4_S3TC" ).setInteger( GL_RGB4_S3TC );
      self->addClassProperty( c_sdlglext, "RGBA_S3TC" ).setInteger( GL_RGBA_S3TC );
      self->addClassProperty( c_sdlglext, "RGBA4_S3TC" ).setInteger( GL_RGBA4_S3TC );
      #endif

      #ifndef GL_ATI_draw_buffers
      self->addClassProperty( c_sdlglext, "MAX_DRAW_BUFFERS_ATI" ).setInteger( GL_MAX_DRAW_BUFFERS_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER0_ATI" ).setInteger( GL_DRAW_BUFFER0_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER1_ATI" ).setInteger( GL_DRAW_BUFFER1_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER2_ATI" ).setInteger( GL_DRAW_BUFFER2_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER3_ATI" ).setInteger( GL_DRAW_BUFFER3_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER4_ATI" ).setInteger( GL_DRAW_BUFFER4_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER5_ATI" ).setInteger( GL_DRAW_BUFFER5_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER6_ATI" ).setInteger( GL_DRAW_BUFFER6_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER7_ATI" ).setInteger( GL_DRAW_BUFFER7_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER8_ATI" ).setInteger( GL_DRAW_BUFFER8_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER9_ATI" ).setInteger( GL_DRAW_BUFFER9_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER10_ATI" ).setInteger( GL_DRAW_BUFFER10_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER11_ATI" ).setInteger( GL_DRAW_BUFFER11_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER12_ATI" ).setInteger( GL_DRAW_BUFFER12_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER13_ATI" ).setInteger( GL_DRAW_BUFFER13_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER14_ATI" ).setInteger( GL_DRAW_BUFFER14_ATI );
      self->addClassProperty( c_sdlglext, "DRAW_BUFFER15_ATI" ).setInteger( GL_DRAW_BUFFER15_ATI );
      #endif

      #ifndef GL_ATI_pixel_format_float
      self->addClassProperty( c_sdlglext, "TYPE_RGBA_FLOAT_ATI" ).setInteger( GL_TYPE_RGBA_FLOAT_ATI );
      self->addClassProperty( c_sdlglext, "COLOR_CLEAR_UNCLAMPED_VALUE_ATI" ).setInteger( GL_COLOR_CLEAR_UNCLAMPED_VALUE_ATI );
      #endif

      #ifndef GL_ATI_texture_env_combine3
      self->addClassProperty( c_sdlglext, "MODULATE_ADD_ATI" ).setInteger( GL_MODULATE_ADD_ATI );
      self->addClassProperty( c_sdlglext, "MODULATE_SIGNED_ADD_ATI" ).setInteger( GL_MODULATE_SIGNED_ADD_ATI );
      self->addClassProperty( c_sdlglext, "MODULATE_SUBTRACT_ATI" ).setInteger( GL_MODULATE_SUBTRACT_ATI );
      #endif

      #ifndef GL_ATI_texture_float
      self->addClassProperty( c_sdlglext, "RGBA_FLOAT32_ATI" ).setInteger( GL_RGBA_FLOAT32_ATI );
      self->addClassProperty( c_sdlglext, "RGB_FLOAT32_ATI" ).setInteger( GL_RGB_FLOAT32_ATI );
      self->addClassProperty( c_sdlglext, "ALPHA_FLOAT32_ATI" ).setInteger( GL_ALPHA_FLOAT32_ATI );
      self->addClassProperty( c_sdlglext, "INTENSITY_FLOAT32_ATI" ).setInteger( GL_INTENSITY_FLOAT32_ATI );
      self->addClassProperty( c_sdlglext, "LUMINANCE_FLOAT32_ATI" ).setInteger( GL_LUMINANCE_FLOAT32_ATI );
      self->addClassProperty( c_sdlglext, "LUMINANCE_ALPHA_FLOAT32_ATI" ).setInteger( GL_LUMINANCE_ALPHA_FLOAT32_ATI );
      self->addClassProperty( c_sdlglext, "RGBA_FLOAT16_ATI" ).setInteger( GL_RGBA_FLOAT16_ATI );
      self->addClassProperty( c_sdlglext, "RGB_FLOAT16_ATI" ).setInteger( GL_RGB_FLOAT16_ATI );
      self->addClassProperty( c_sdlglext, "ALPHA_FLOAT16_ATI" ).setInteger( GL_ALPHA_FLOAT16_ATI );
      self->addClassProperty( c_sdlglext, "INTENSITY_FLOAT16_ATI" ).setInteger( GL_INTENSITY_FLOAT16_ATI );
      self->addClassProperty( c_sdlglext, "LUMINANCE_FLOAT16_ATI" ).setInteger( GL_LUMINANCE_FLOAT16_ATI );
      self->addClassProperty( c_sdlglext, "LUMINANCE_ALPHA_FLOAT16_ATI" ).setInteger( GL_LUMINANCE_ALPHA_FLOAT16_ATI );
      #endif

      #ifndef GL_NV_float_buffer
      self->addClassProperty( c_sdlglext, "FLOAT_R_NV" ).setInteger( GL_FLOAT_R_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RG_NV" ).setInteger( GL_FLOAT_RG_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGB_NV" ).setInteger( GL_FLOAT_RGB_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGBA_NV" ).setInteger( GL_FLOAT_RGBA_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_R16_NV" ).setInteger( GL_FLOAT_R16_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_R32_NV" ).setInteger( GL_FLOAT_R32_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RG16_NV" ).setInteger( GL_FLOAT_RG16_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RG32_NV" ).setInteger( GL_FLOAT_RG32_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGB16_NV" ).setInteger( GL_FLOAT_RGB16_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGB32_NV" ).setInteger( GL_FLOAT_RGB32_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGBA16_NV" ).setInteger( GL_FLOAT_RGBA16_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGBA32_NV" ).setInteger( GL_FLOAT_RGBA32_NV );
      self->addClassProperty( c_sdlglext, "TEXTURE_FLOAT_COMPONENTS_NV" ).setInteger( GL_TEXTURE_FLOAT_COMPONENTS_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_CLEAR_COLOR_VALUE_NV" ).setInteger( GL_FLOAT_CLEAR_COLOR_VALUE_NV );
      self->addClassProperty( c_sdlglext, "FLOAT_RGBA_MODE_NV" ).setInteger( GL_FLOAT_RGBA_MODE_NV );
      #endif

      #ifndef GL_NV_fragment_program
      self->addClassProperty( c_sdlglext, "MAX_FRAGMENT_PROGRAM_LOCAL_PARAMETERS_NV" ).setInteger( GL_MAX_FRAGMENT_PROGRAM_LOCAL_PARAMETERS_NV );
      self->addClassProperty( c_sdlglext, "FRAGMENT_PROGRAM_NV" ).setInteger( GL_FRAGMENT_PROGRAM_NV );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_COORDS_NV" ).setInteger( GL_MAX_TEXTURE_COORDS_NV );
      self->addClassProperty( c_sdlglext, "MAX_TEXTURE_IMAGE_UNITS_NV" ).setInteger( GL_MAX_TEXTURE_IMAGE_UNITS_NV );
      self->addClassProperty( c_sdlglext, "FRAGMENT_PROGRAM_BINDING_NV" ).setInteger( GL_FRAGMENT_PROGRAM_BINDING_NV );
      self->addClassProperty( c_sdlglext, "PROGRAM_ERROR_STRING_NV" ).setInteger( GL_PROGRAM_ERROR_STRING_NV );
      #endif

      #ifndef GL_NV_half_float
      self->addClassProperty( c_sdlglext, "HALF_FLOAT_NV" ).setInteger( GL_HALF_FLOAT_NV );
      #endif

      #ifndef GL_NV_pixel_data_range
      self->addClassProperty( c_sdlglext, "WRITE_PIXEL_DATA_RANGE_NV" ).setInteger( GL_WRITE_PIXEL_DATA_RANGE_NV );
      self->addClassProperty( c_sdlglext, "READ_PIXEL_DATA_RANGE_NV" ).setInteger( GL_READ_PIXEL_DATA_RANGE_NV );
      self->addClassProperty( c_sdlglext, "WRITE_PIXEL_DATA_RANGE_LENGTH_NV" ).setInteger( GL_WRITE_PIXEL_DATA_RANGE_LENGTH_NV );
      self->addClassProperty( c_sdlglext, "READ_PIXEL_DATA_RANGE_LENGTH_NV" ).setInteger( GL_READ_PIXEL_DATA_RANGE_LENGTH_NV );
      self->addClassProperty( c_sdlglext, "WRITE_PIXEL_DATA_RANGE_POINTER_NV" ).setInteger( GL_WRITE_PIXEL_DATA_RANGE_POINTER_NV );
      self->addClassProperty( c_sdlglext, "READ_PIXEL_DATA_RANGE_POINTER_NV" ).setInteger( GL_READ_PIXEL_DATA_RANGE_POINTER_NV );
      #endif

      #ifndef GL_NV_primitive_restart
      self->addClassProperty( c_sdlglext, "PRIMITIVE_RESTART_NV" ).setInteger( GL_PRIMITIVE_RESTART_NV );
      self->addClassProperty( c_sdlglext, "PRIMITIVE_RESTART_INDEX_NV" ).setInteger( GL_PRIMITIVE_RESTART_INDEX_NV );
      #endif

      #ifndef GL_NV_texture_expand_normal
      self->addClassProperty( c_sdlglext, "TEXTURE_UNSIGNED_REMAP_MODE_NV" ).setInteger( GL_TEXTURE_UNSIGNED_REMAP_MODE_NV );
      #endif

      #ifndef GL_ATI_separate_stencil
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_FUNC_ATI" ).setInteger( GL_STENCIL_BACK_FUNC_ATI );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_FAIL_ATI" ).setInteger( GL_STENCIL_BACK_FAIL_ATI );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_PASS_DEPTH_FAIL_ATI" ).setInteger( GL_STENCIL_BACK_PASS_DEPTH_FAIL_ATI );
      self->addClassProperty( c_sdlglext, "STENCIL_BACK_PASS_DEPTH_PASS_ATI" ).setInteger( GL_STENCIL_BACK_PASS_DEPTH_PASS_ATI );
      #endif


      #ifndef GL_OES_read_format
      self->addClassProperty( c_sdlglext, "IMPLEMENTATION_COLOR_READ_TYPE_OES" ).setInteger( GL_IMPLEMENTATION_COLOR_READ_TYPE_OES );
      self->addClassProperty( c_sdlglext, "IMPLEMENTATION_COLOR_READ_FORMAT_OES" ).setInteger( GL_IMPLEMENTATION_COLOR_READ_FORMAT_OES );
      #endif

      #ifndef GL_EXT_depth_bounds_test
      self->addClassProperty( c_sdlglext, "DEPTH_BOUNDS_TEST_EXT" ).setInteger( GL_DEPTH_BOUNDS_TEST_EXT );
      self->addClassProperty( c_sdlglext, "DEPTH_BOUNDS_EXT" ).setInteger( GL_DEPTH_BOUNDS_EXT );
      #endif

      #ifndef GL_EXT_texture_mirror_clamp
      self->addClassProperty( c_sdlglext, "MIRROR_CLAMP_EXT" ).setInteger( GL_MIRROR_CLAMP_EXT );
      self->addClassProperty( c_sdlglext, "MIRROR_CLAMP_TO_EDGE_EXT" ).setInteger( GL_MIRROR_CLAMP_TO_EDGE_EXT );
      self->addClassProperty( c_sdlglext, "MIRROR_CLAMP_TO_BORDER_EXT" ).setInteger( GL_MIRROR_CLAMP_TO_BORDER_EXT );
      #endif

      #ifndef GL_EXT_blend_equation_separate
      #define GL_BLEND_EQUATION_RGB_EXT         GL_BLEND_EQUATION
      self->addClassProperty( c_sdlglext, "BLEND_EQUATION_ALPHA_EXT" ).setInteger( GL_BLEND_EQUATION_ALPHA_EXT );
      #endif

      #ifndef GL_MESA_pack_invert
      self->addClassProperty( c_sdlglext, "PACK_INVERT_MESA" ).setInteger( GL_PACK_INVERT_MESA );
      #endif

      #ifndef GL_MESA_ycbcr_texture
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_8_8_MESA" ).setInteger( GL_UNSIGNED_SHORT_8_8_MESA );
      self->addClassProperty( c_sdlglext, "UNSIGNED_SHORT_8_8_REV_MESA" ).setInteger( GL_UNSIGNED_SHORT_8_8_REV_MESA );
      self->addClassProperty( c_sdlglext, "YCBCR_MESA" ).setInteger( GL_YCBCR_MESA );
      #endif

      #ifndef GL_EXT_pixel_buffer_object
      self->addClassProperty( c_sdlglext, "PIXEL_PACK_BUFFER_EXT" ).setInteger( GL_PIXEL_PACK_BUFFER_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_UNPACK_BUFFER_EXT" ).setInteger( GL_PIXEL_UNPACK_BUFFER_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_PACK_BUFFER_BINDING_EXT" ).setInteger( GL_PIXEL_PACK_BUFFER_BINDING_EXT );
      self->addClassProperty( c_sdlglext, "PIXEL_UNPACK_BUFFER_BINDING_EXT" ).setInteger( GL_PIXEL_UNPACK_BUFFER_BINDING_EXT );
      #endif



      #ifndef GL_NV_fragment_program2
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_EXEC_INSTRUCTIONS_NV" ).setInteger( GL_MAX_PROGRAM_EXEC_INSTRUCTIONS_NV );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_CALL_DEPTH_NV" ).setInteger( GL_MAX_PROGRAM_CALL_DEPTH_NV );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_IF_DEPTH_NV" ).setInteger( GL_MAX_PROGRAM_IF_DEPTH_NV );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_LOOP_DEPTH_NV" ).setInteger( GL_MAX_PROGRAM_LOOP_DEPTH_NV );
      self->addClassProperty( c_sdlglext, "MAX_PROGRAM_LOOP_COUNT_NV" ).setInteger( GL_MAX_PROGRAM_LOOP_COUNT_NV );
      #endif

      #ifndef GL_EXT_framebuffer_object
      self->addClassProperty( c_sdlglext, "INVALID_FRAMEBUFFER_OPERATION_EXT" ).setInteger( GL_INVALID_FRAMEBUFFER_OPERATION_EXT );
      self->addClassProperty( c_sdlglext, "MAX_RENDERBUFFER_SIZE_EXT" ).setInteger( GL_MAX_RENDERBUFFER_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_BINDING_EXT" ).setInteger( GL_FRAMEBUFFER_BINDING_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_BINDING_EXT" ).setInteger( GL_RENDERBUFFER_BINDING_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT" ).setInteger( GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT" ).setInteger( GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL_EXT" ).setInteger( GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE_EXT" ).setInteger( GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT" ).setInteger( GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_COMPLETE_EXT" ).setInteger( GL_FRAMEBUFFER_COMPLETE_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_FORMATS_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT" ).setInteger( GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_UNSUPPORTED_EXT" ).setInteger( GL_FRAMEBUFFER_UNSUPPORTED_EXT );
      self->addClassProperty( c_sdlglext, "MAX_COLOR_ATTACHMENTS_EXT" ).setInteger( GL_MAX_COLOR_ATTACHMENTS_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT0_EXT" ).setInteger( GL_COLOR_ATTACHMENT0_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT1_EXT" ).setInteger( GL_COLOR_ATTACHMENT1_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT2_EXT" ).setInteger( GL_COLOR_ATTACHMENT2_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT3_EXT" ).setInteger( GL_COLOR_ATTACHMENT3_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT4_EXT" ).setInteger( GL_COLOR_ATTACHMENT4_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT5_EXT" ).setInteger( GL_COLOR_ATTACHMENT5_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT6_EXT" ).setInteger( GL_COLOR_ATTACHMENT6_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT7_EXT" ).setInteger( GL_COLOR_ATTACHMENT7_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT8_EXT" ).setInteger( GL_COLOR_ATTACHMENT8_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT9_EXT" ).setInteger( GL_COLOR_ATTACHMENT9_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT10_EXT" ).setInteger( GL_COLOR_ATTACHMENT10_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT11_EXT" ).setInteger( GL_COLOR_ATTACHMENT11_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT12_EXT" ).setInteger( GL_COLOR_ATTACHMENT12_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT13_EXT" ).setInteger( GL_COLOR_ATTACHMENT13_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT14_EXT" ).setInteger( GL_COLOR_ATTACHMENT14_EXT );
      self->addClassProperty( c_sdlglext, "COLOR_ATTACHMENT15_EXT" ).setInteger( GL_COLOR_ATTACHMENT15_EXT );
      self->addClassProperty( c_sdlglext, "DEPTH_ATTACHMENT_EXT" ).setInteger( GL_DEPTH_ATTACHMENT_EXT );
      self->addClassProperty( c_sdlglext, "STENCIL_ATTACHMENT_EXT" ).setInteger( GL_STENCIL_ATTACHMENT_EXT );
      self->addClassProperty( c_sdlglext, "FRAMEBUFFER_EXT" ).setInteger( GL_FRAMEBUFFER_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_EXT" ).setInteger( GL_RENDERBUFFER_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_WIDTH_EXT" ).setInteger( GL_RENDERBUFFER_WIDTH_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_HEIGHT_EXT" ).setInteger( GL_RENDERBUFFER_HEIGHT_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_INTERNAL_FORMAT_EXT" ).setInteger( GL_RENDERBUFFER_INTERNAL_FORMAT_EXT );
      self->addClassProperty( c_sdlglext, "STENCIL_INDEX1_EXT" ).setInteger( GL_STENCIL_INDEX1_EXT );
      self->addClassProperty( c_sdlglext, "STENCIL_INDEX4_EXT" ).setInteger( GL_STENCIL_INDEX4_EXT );
      self->addClassProperty( c_sdlglext, "STENCIL_INDEX8_EXT" ).setInteger( GL_STENCIL_INDEX8_EXT );
      self->addClassProperty( c_sdlglext, "STENCIL_INDEX16_EXT" ).setInteger( GL_STENCIL_INDEX16_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_RED_SIZE_EXT" ).setInteger( GL_RENDERBUFFER_RED_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_GREEN_SIZE_EXT" ).setInteger( GL_RENDERBUFFER_GREEN_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_BLUE_SIZE_EXT" ).setInteger( GL_RENDERBUFFER_BLUE_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_ALPHA_SIZE_EXT" ).setInteger( GL_RENDERBUFFER_ALPHA_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_DEPTH_SIZE_EXT" ).setInteger( GL_RENDERBUFFER_DEPTH_SIZE_EXT );
      self->addClassProperty( c_sdlglext, "RENDERBUFFER_STENCIL_SIZE_EXT" ).setInteger( GL_RENDERBUFFER_STENCIL_SIZE_EXT );
      #endif
   }

   return self;
}
/* end of sdlttf.cpp */

