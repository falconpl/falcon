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
#include "gl_ext.h"
#include "gl_mod.h"

#include <SDL.h>
#include <SDL_opengl.h>


/*#
   @module sdlopengl OpenGL
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

   //=================================================================
   // Encapsulation SDLOpenGL
   //

   /*#
      @class OpenGL
      @brief Main SDL OpenGL encapsulation class.

      This class is the namespace for OpenGL functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlopengl
      module.
   */
   {

      self->addExtFunc( "Viewport", Falcon::Ext::opengl_Viewport );
      self->addExtFunc( "SetAttribute", Falcon::Ext::sdlgl_SetAttribute );
      self->addExtFunc( "GetAttribute", Falcon::Ext::sdlgl_GetAttribute );
      self->addExtFunc( "LoadIdentity", Falcon::Ext::opengl_LoadIdentity );
      self->addExtFunc( "MatrixMode", Falcon::Ext::opengl_MatrixMode );
      self->addExtFunc( "ClearColor", Falcon::Ext::opengl_ClearColor );
      self->addExtFunc( "ClearDepth", Falcon::Ext::opengl_ClearDepth );
      self->addExtFunc( "Enable", Falcon::Ext::opengl_Enable );
      self->addExtFunc( "Clear", Falcon::Ext::opengl_Clear );
      self->addExtFunc( "Begin", Falcon::Ext::opengl_Begin );
      self->addExtFunc( "End", Falcon::Ext::opengl_End );
      self->addExtFunc( "SwapBuffers", Falcon::Ext::opengl_SwapBuffers );
      self->addExtFunc( "Color3d", Falcon::Ext::opengl_Color3d );
      self->addExtFunc( "Vertex3d", Falcon::Ext::opengl_Vertex3d );
      self->addExtFunc( "Translate", Falcon::Ext::opengl_Translate );
      self->addExtFunc( "Rotate", Falcon::Ext::opengl_Rotate );
      self->addExtFunc( "GenTextures", Falcon::Ext::opengl_GenTextures );
      self->addExtFunc( "ShadeModel", Falcon::Ext::opengl_ShadeModel );
      self->addExtFunc( "Hint", Falcon::Ext::opengl_Hint );
      self->addExtFunc( "DepthFunc", Falcon::Ext::opengl_DepthFunc );
      self->addExtFunc( "TexCoord2d", Falcon::Ext::opengl_TexCoord2d );
      self->addExtFunc( "BindTexture", Falcon::Ext::opengl_BindTexture );
      self->addExtFunc( "TexParameteri", Falcon::Ext::opengl_TexParameteri );
      self->addExtFunc( "TexImage2D", Falcon::Ext::opengl_TexImage2D );
      self->addExtFunc( "GetError", Falcon::Ext::opengl_GetError );
      //self->addExtFunc( "GetError", Falcon::Ext::opengl_GetError );
   }
   {
      Falcon::Symbol *c_sdlgl = self->addClass( "GL" );
      /*SDL_GLattr constants*/
      self->addClassProperty( c_sdlgl, "RED_SIZE" ).setInteger( (SDL_GLattr)SDL_GL_RED_SIZE );
      self->addClassProperty( c_sdlgl, "GREEN_SIZE" ).setInteger( SDL_GL_GREEN_SIZE );
      self->addClassProperty( c_sdlgl, "BLUE_SIZE" ).setInteger( SDL_GL_BLUE_SIZE );
      self->addClassProperty( c_sdlgl, "ALPHA_SIZE" ).setInteger( SDL_GL_ALPHA_SIZE );
      self->addClassProperty( c_sdlgl, "BUFFER_SIZE" ).setInteger( SDL_GL_BUFFER_SIZE );
      self->addClassProperty( c_sdlgl, "DOUBLEBUFFER" ).setInteger( SDL_GL_DOUBLEBUFFER );
      self->addClassProperty( c_sdlgl, "DEPTH_SIZE" ).setInteger( SDL_GL_DEPTH_SIZE );
      self->addClassProperty( c_sdlgl, "STENCIL_SIZE" ).setInteger( SDL_GL_STENCIL_SIZE );
      self->addClassProperty( c_sdlgl, "ACCUM_RED_SIZE" ).setInteger( SDL_GL_ACCUM_RED_SIZE );
      self->addClassProperty( c_sdlgl, "ACCUM_GREEN_SIZE" ).setInteger( SDL_GL_ACCUM_GREEN_SIZE );
      self->addClassProperty( c_sdlgl, "ACCUM_BLUE_SIZE" ).setInteger( SDL_GL_ACCUM_BLUE_SIZE );
      self->addClassProperty( c_sdlgl, "ACCUM_ALPHA_SIZE" ).setInteger( SDL_GL_ACCUM_ALPHA_SIZE );
      self->addClassProperty( c_sdlgl, "STEREO" ).setInteger( SDL_GL_STEREO );
      self->addClassProperty( c_sdlgl, "MULTISAMPLEBUFFERS" ).setInteger( SDL_GL_MULTISAMPLEBUFFERS );
      self->addClassProperty( c_sdlgl, "MULTISAMPLESAMPLES" ).setInteger( SDL_GL_MULTISAMPLESAMPLES );
      self->addClassProperty( c_sdlgl, "ACCELERATED_VISUAL" ).setInteger( SDL_GL_ACCELERATED_VISUAL );
      self->addClassProperty( c_sdlgl, "SWAP_CONTROL" ).setInteger( SDL_GL_SWAP_CONTROL );
      /* Version */
      self->addClassProperty( c_sdlgl, "VERSION_1_1" ).setInteger( GL_VERSION_1_1 );

      /* AccumOp */
      self->addClassProperty( c_sdlgl, "ACCUM" ).setInteger( GL_ACCUM );
      self->addClassProperty( c_sdlgl, "LOAD" ).setInteger( GL_LOAD );
      self->addClassProperty( c_sdlgl, "RETURN" ).setInteger( GL_RETURN );
      self->addClassProperty( c_sdlgl, "MULT" ).setInteger( GL_MULT );
      self->addClassProperty( c_sdlgl, "ADD" ).setInteger( GL_ADD );

      /* AlphaFunction */
      self->addClassProperty( c_sdlgl, "NEVER" ).setInteger( GL_NEVER );
      self->addClassProperty( c_sdlgl, "LESS" ).setInteger( GL_LESS );
      self->addClassProperty( c_sdlgl, "EQUAL" ).setInteger( GL_EQUAL );
      self->addClassProperty( c_sdlgl, "LEQUAL" ).setInteger( GL_LEQUAL );
      self->addClassProperty( c_sdlgl, "GREATER" ).setInteger( GL_GREATER );
      self->addClassProperty( c_sdlgl, "NOTEQUAL" ).setInteger( GL_NOTEQUAL );
      self->addClassProperty( c_sdlgl, "GEQUAL" ).setInteger( GL_GEQUAL );
      self->addClassProperty( c_sdlgl, "ALWAYS" ).setInteger( GL_ALWAYS );

      /* AttribMask */
      self->addClassProperty( c_sdlgl, "CURRENT_BIT" ).setInteger( GL_CURRENT_BIT );
      self->addClassProperty( c_sdlgl, "POINT_BIT" ).setInteger( GL_POINT_BIT );
      self->addClassProperty( c_sdlgl, "LINE_BIT" ).setInteger( GL_LINE_BIT );
      self->addClassProperty( c_sdlgl, "POLYGON_BIT" ).setInteger( GL_POLYGON_BIT );
      self->addClassProperty( c_sdlgl, "POLYGON_STIPPLE_BIT" ).setInteger( GL_POLYGON_STIPPLE_BIT );
      self->addClassProperty( c_sdlgl, "PIXEL_MODE_BIT" ).setInteger( GL_PIXEL_MODE_BIT );
      self->addClassProperty( c_sdlgl, "LIGHTING_BIT" ).setInteger( GL_LIGHTING_BIT );
      self->addClassProperty( c_sdlgl, "FOG_BIT" ).setInteger( GL_FOG_BIT );
      self->addClassProperty( c_sdlgl, "DEPTH_BUFFER_BIT" ).setInteger( GL_DEPTH_BUFFER_BIT );
      self->addClassProperty( c_sdlgl, "ACCUM_BUFFER_BIT" ).setInteger( GL_ACCUM_BUFFER_BIT );
      self->addClassProperty( c_sdlgl, "STENCIL_BUFFER_BIT" ).setInteger( GL_STENCIL_BUFFER_BIT );
      self->addClassProperty( c_sdlgl, "VIEWPORT_BIT" ).setInteger( GL_VIEWPORT_BIT );
      self->addClassProperty( c_sdlgl, "TRANSFORM_BIT" ).setInteger( GL_TRANSFORM_BIT );
      self->addClassProperty( c_sdlgl, "ENABLE_BIT" ).setInteger( GL_ENABLE_BIT );
      self->addClassProperty( c_sdlgl, "COLOR_BUFFER_BIT" ).setInteger( GL_COLOR_BUFFER_BIT );
      self->addClassProperty( c_sdlgl, "HINT_BIT" ).setInteger( GL_HINT_BIT );
      self->addClassProperty( c_sdlgl, "EVAL_BIT" ).setInteger( GL_EVAL_BIT );
      self->addClassProperty( c_sdlgl, "LIST_BIT" ).setInteger( GL_LIST_BIT );
      self->addClassProperty( c_sdlgl, "TEXTURE_BIT" ).setInteger( GL_TEXTURE_BIT );
      self->addClassProperty( c_sdlgl, "SCISSOR_BIT" ).setInteger( GL_SCISSOR_BIT );
      self->addClassProperty( c_sdlgl, "ALL_ATTRIB_BITS" ).setInteger( GL_ALL_ATTRIB_BITS );

      /* BeginMode */
      self->addClassProperty( c_sdlgl, "POINTS" ).setInteger( GL_POINTS );
      self->addClassProperty( c_sdlgl, "LINES" ).setInteger( GL_LINES );
      self->addClassProperty( c_sdlgl, "LINE_LOOP" ).setInteger( GL_LINE_LOOP );
      self->addClassProperty( c_sdlgl, "LINE_STRIP" ).setInteger( GL_LINE_STRIP );
      self->addClassProperty( c_sdlgl, "TRIANGLES" ).setInteger( GL_TRIANGLES );
      self->addClassProperty( c_sdlgl, "TRIANGLE_STRIP" ).setInteger( GL_TRIANGLE_STRIP );
      self->addClassProperty( c_sdlgl, "TRIANGLE_FAN" ).setInteger( GL_TRIANGLE_FAN );
      self->addClassProperty( c_sdlgl, "QUADS" ).setInteger( GL_QUADS );
      self->addClassProperty( c_sdlgl, "QUAD_STRIP" ).setInteger( GL_QUAD_STRIP );
      self->addClassProperty( c_sdlgl, "POLYGON" ).setInteger( GL_POLYGON );

      /* BlendingFactorDest */
      self->addClassProperty( c_sdlgl, "ZERO" ).setInteger( GL_ZERO );
      self->addClassProperty( c_sdlgl, "ONE" ).setInteger( GL_ONE );
      self->addClassProperty( c_sdlgl, "SRC_COLOR" ).setInteger( GL_SRC_COLOR );
      self->addClassProperty( c_sdlgl, "ONE_MINUS_SRC_COLOR" ).setInteger( GL_ONE_MINUS_SRC_COLOR );
      self->addClassProperty( c_sdlgl, "SRC_ALPHA" ).setInteger( GL_SRC_ALPHA );
      self->addClassProperty( c_sdlgl, "ONE_MINUS_SRC_ALPHA" ).setInteger( GL_ONE_MINUS_SRC_ALPHA );
      self->addClassProperty( c_sdlgl, "DST_ALPHA" ).setInteger( GL_DST_ALPHA );
      self->addClassProperty( c_sdlgl, "ONE_MINUS_DST_ALPHA" ).setInteger( GL_ONE_MINUS_DST_ALPHA );

      /* BlendingFactorSrc */
      /*      GL_ZERO */
      /*      GL_ONE */
      self->addClassProperty( c_sdlgl, "DST_COLOR" ).setInteger( GL_DST_COLOR );
      self->addClassProperty( c_sdlgl, "ONE_MINUS_DST_COLOR" ).setInteger( GL_ONE_MINUS_DST_COLOR );
      self->addClassProperty( c_sdlgl, "SRC_ALPHA_SATURATE" ).setInteger( GL_SRC_ALPHA_SATURATE );
      /*      GL_SRC_ALPHA */
      /*      GL_ONE_MINUS_SRC_ALPHA */
      /*      GL_DST_ALPHA */
      /*      GL_ONE_MINUS_DST_ALPHA */

      /* Boolean */
      self->addClassProperty( c_sdlgl, "TRUE" ).setInteger( GL_TRUE );
      self->addClassProperty( c_sdlgl, "FALSE" ).setInteger( GL_FALSE );

      /* ClearBufferMask */
      /*      GL_COLOR_BUFFER_BIT */
      /*      GL_ACCUM_BUFFER_BIT */
      /*      GL_STENCIL_BUFFER_BIT */
      /*      GL_DEPTH_BUFFER_BIT */

      /* ClientArrayType */
      /*      GL_VERTEX_ARRAY */
      /*      GL_NORMAL_ARRAY */
      /*      GL_COLOR_ARRAY */
      /*      GL_INDEX_ARRAY */
      /*      GL_TEXTURE_COORD_ARRAY */
      /*      GL_EDGE_FLAG_ARRAY */

      /* ClipPlaneName */
      self->addClassProperty( c_sdlgl, "CLIP_PLANE0" ).setInteger( GL_CLIP_PLANE0 );
      self->addClassProperty( c_sdlgl, "CLIP_PLANE1" ).setInteger( GL_CLIP_PLANE1 );
      self->addClassProperty( c_sdlgl, "CLIP_PLANE2" ).setInteger( GL_CLIP_PLANE2 );
      self->addClassProperty( c_sdlgl, "CLIP_PLANE3" ).setInteger( GL_CLIP_PLANE3 );
      self->addClassProperty( c_sdlgl, "CLIP_PLANE4" ).setInteger( GL_CLIP_PLANE4 );
      self->addClassProperty( c_sdlgl, "CLIP_PLANE5" ).setInteger( GL_CLIP_PLANE5 );

      /* ColorMaterialFace */
      /*      GL_FRONT */
      /*      GL_BACK */
      /*      GL_FRONT_AND_BACK */

      /* ColorMaterialParameter */
      /*      GL_AMBIENT */
      /*      GL_DIFFUSE */
      /*      GL_SPECULAR */
      /*      GL_EMISSION */
      /*      GL_AMBIENT_AND_DIFFUSE */

      /* ColorPointerType */
      /*      GL_BYTE */
      /*      GL_UNSIGNED_BYTE */
      /*      GL_SHORT */
      /*      GL_UNSIGNED_SHORT */
      /*      GL_INT */
      /*      GL_UNSIGNED_INT */
      /*      GL_FLOAT */
      /*      GL_DOUBLE */

      /* CullFaceMode */
      /*      GL_FRONT */
      /*      GL_BACK */
      /*      GL_FRONT_AND_BACK */

      /* DataType */
      self->addClassProperty( c_sdlgl, "BYTE" ).setInteger( GL_BYTE );
      self->addClassProperty( c_sdlgl, "UNSIGNED_BYTE" ).setInteger( GL_UNSIGNED_BYTE );
      self->addClassProperty( c_sdlgl, "SHORT" ).setInteger( GL_SHORT );
      self->addClassProperty( c_sdlgl, "UNSIGNED_SHORT" ).setInteger( GL_UNSIGNED_SHORT );
      self->addClassProperty( c_sdlgl, "INT" ).setInteger( GL_INT );
      self->addClassProperty( c_sdlgl, "UNSIGNED_INT" ).setInteger( GL_UNSIGNED_INT );
      self->addClassProperty( c_sdlgl, "FLOAT" ).setInteger( GL_FLOAT );
      self->addClassProperty( c_sdlgl, "2_BYTES" ).setInteger( GL_2_BYTES );
      self->addClassProperty( c_sdlgl, "3_BYTES" ).setInteger( GL_3_BYTES );
      self->addClassProperty( c_sdlgl, "4_BYTES" ).setInteger( GL_4_BYTES );
      self->addClassProperty( c_sdlgl, "DOUBLE" ).setInteger( GL_DOUBLE );

      /* DepthFunction */
      /*      GL_NEVER */
      /*      GL_LESS */
      /*      GL_EQUAL */
      /*      GL_LEQUAL */
      /*      GL_GREATER */
      /*      GL_NOTEQUAL */
      /*      GL_GEQUAL */
      /*      GL_ALWAYS */

      /* DrawBufferMode */
      self->addClassProperty( c_sdlgl, "NONE" ).setInteger( GL_NONE );
      self->addClassProperty( c_sdlgl, "FRONT_LEFT" ).setInteger( GL_FRONT_LEFT );
      self->addClassProperty( c_sdlgl, "FRONT_RIGHT" ).setInteger( GL_FRONT_RIGHT );
      self->addClassProperty( c_sdlgl, "BACK_LEFT" ).setInteger( GL_BACK_LEFT );
      self->addClassProperty( c_sdlgl, "BACK_RIGHT" ).setInteger( GL_BACK_RIGHT );
      self->addClassProperty( c_sdlgl, "FRONT" ).setInteger( GL_FRONT );
      self->addClassProperty( c_sdlgl, "BACK" ).setInteger( GL_BACK );
      self->addClassProperty( c_sdlgl, "LEFT" ).setInteger( GL_LEFT );
      self->addClassProperty( c_sdlgl, "RIGHT" ).setInteger( GL_RIGHT );
      self->addClassProperty( c_sdlgl, "FRONT_AND_BACK" ).setInteger( GL_FRONT_AND_BACK );
      self->addClassProperty( c_sdlgl, "AUX0" ).setInteger( GL_AUX0 );
      self->addClassProperty( c_sdlgl, "AUX1" ).setInteger( GL_AUX1 );
      self->addClassProperty( c_sdlgl, "AUX2" ).setInteger( GL_AUX2 );
      self->addClassProperty( c_sdlgl, "AUX3" ).setInteger( GL_AUX3 );

      /* Enable */
      /*      GL_FOG */
      /*      GL_LIGHTING */
      /*      GL_TEXTURE_1D */
      /*      GL_TEXTURE_2D */
      /*      GL_LINE_STIPPLE */
      /*      GL_POLYGON_STIPPLE */
      /*      GL_CULL_FACE */
      /*      GL_ALPHA_TEST */
      /*      GL_BLEND */
      /*      GL_INDEX_LOGIC_OP */
      /*      GL_COLOR_LOGIC_OP */
      /*      GL_DITHER */
      /*      GL_STENCIL_TEST */
      /*      GL_DEPTH_TEST */
      /*      GL_CLIP_PLANE0 */
      /*      GL_CLIP_PLANE1 */
      /*      GL_CLIP_PLANE2 */
      /*      GL_CLIP_PLANE3 */
      /*      GL_CLIP_PLANE4 */
      /*      GL_CLIP_PLANE5 */
      /*      GL_LIGHT0 */
      /*      GL_LIGHT1 */
      /*      GL_LIGHT2 */
      /*      GL_LIGHT3 */
      /*      GL_LIGHT4 */
      /*      GL_LIGHT5 */
      /*      GL_LIGHT6 */
      /*      GL_LIGHT7 */
      /*      GL_TEXTURE_GEN_S */
      /*      GL_TEXTURE_GEN_T */
      /*      GL_TEXTURE_GEN_R */
      /*      GL_TEXTURE_GEN_Q */
      /*      GL_MAP1_VERTEX_3 */
      /*      GL_MAP1_VERTEX_4 */
      /*      GL_MAP1_COLOR_4 */
      /*      GL_MAP1_INDEX */
      /*      GL_MAP1_NORMAL */
      /*      GL_MAP1_TEXTURE_COORD_1 */
      /*      GL_MAP1_TEXTURE_COORD_2 */
      /*      GL_MAP1_TEXTURE_COORD_3 */
      /*      GL_MAP1_TEXTURE_COORD_4 */
      /*      GL_MAP2_VERTEX_3 */
      /*      GL_MAP2_VERTEX_4 */
      /*      GL_MAP2_COLOR_4 */
      /*      GL_MAP2_INDEX */
      /*      GL_MAP2_NORMAL */
      /*      GL_MAP2_TEXTURE_COORD_1 */
      /*      GL_MAP2_TEXTURE_COORD_2 */
      /*      GL_MAP2_TEXTURE_COORD_3 */
      /*      GL_MAP2_TEXTURE_COORD_4 */
      /*      GL_POINT_SMOOTH */
      /*      GL_LINE_SMOOTH */
      /*      GL_POLYGON_SMOOTH */
      /*      GL_SCISSOR_TEST */
      /*      GL_COLOR_MATERIAL */
      /*      GL_NORMALIZE */
      /*      GL_AUTO_NORMAL */
      /*      GL_VERTEX_ARRAY */
      /*      GL_NORMAL_ARRAY */
      /*      GL_COLOR_ARRAY */
      /*      GL_INDEX_ARRAY */
      /*      GL_TEXTURE_COORD_ARRAY */
      /*      GL_EDGE_FLAG_ARRAY */
      /*      GL_POLYGON_OFFSET_POINT */
      /*      GL_POLYGON_OFFSET_LINE */
      /*      GL_POLYGON_OFFSET_FILL */

      /* ErrorCode */
      self->addClassProperty( c_sdlgl, "NO_ERROR" ).setInteger( GL_NO_ERROR );
      self->addClassProperty( c_sdlgl, "INVALID_ENUM" ).setInteger( GL_INVALID_ENUM );
      self->addClassProperty( c_sdlgl, "INVALID_VALUE" ).setInteger( GL_INVALID_VALUE );
      self->addClassProperty( c_sdlgl, "INVALID_OPERATION" ).setInteger( GL_INVALID_OPERATION );
      self->addClassProperty( c_sdlgl, "STACK_OVERFLOW" ).setInteger( GL_STACK_OVERFLOW );
      self->addClassProperty( c_sdlgl, "STACK_UNDERFLOW" ).setInteger( GL_STACK_UNDERFLOW );
      self->addClassProperty( c_sdlgl, "OUT_OF_MEMORY" ).setInteger( GL_OUT_OF_MEMORY );

      /* FeedBackMode */
      self->addClassProperty( c_sdlgl, "2D" ).setInteger( GL_2D );
      self->addClassProperty( c_sdlgl, "3D" ).setInteger( GL_3D );
      self->addClassProperty( c_sdlgl, "3D_COLOR" ).setInteger( GL_3D_COLOR );
      self->addClassProperty( c_sdlgl, "3D_COLOR_TEXTURE" ).setInteger( GL_3D_COLOR_TEXTURE );
      self->addClassProperty( c_sdlgl, "4D_COLOR_TEXTURE" ).setInteger( GL_4D_COLOR_TEXTURE );

      /* FeedBackToken */
      self->addClassProperty( c_sdlgl, "PASS_THROUGH_TOKEN" ).setInteger( GL_PASS_THROUGH_TOKEN );
      self->addClassProperty( c_sdlgl, "POINT_TOKEN" ).setInteger( GL_POINT_TOKEN );
      self->addClassProperty( c_sdlgl, "LINE_TOKEN" ).setInteger( GL_LINE_TOKEN );
      self->addClassProperty( c_sdlgl, "POLYGON_TOKEN" ).setInteger( GL_POLYGON_TOKEN );
      self->addClassProperty( c_sdlgl, "BITMAP_TOKEN" ).setInteger( GL_BITMAP_TOKEN );
      self->addClassProperty( c_sdlgl, "DRAW_PIXEL_TOKEN" ).setInteger( GL_DRAW_PIXEL_TOKEN );
      self->addClassProperty( c_sdlgl, "COPY_PIXEL_TOKEN" ).setInteger( GL_COPY_PIXEL_TOKEN );
      self->addClassProperty( c_sdlgl, "LINE_RESET_TOKEN" ).setInteger( GL_LINE_RESET_TOKEN );

      /* FogMode */
      /*      GL_LINEAR */
      self->addClassProperty( c_sdlgl, "EXP" ).setInteger( GL_EXP );
      self->addClassProperty( c_sdlgl, "EXP2" ).setInteger( GL_EXP2 );


      /* FogParameter */
      /*      GL_FOG_COLOR */
      /*      GL_FOG_DENSITY */
      /*      GL_FOG_END */
      /*      GL_FOG_INDEX */
      /*      GL_FOG_MODE */
      /*      GL_FOG_START */

      /* FrontFaceDirection */
      self->addClassProperty( c_sdlgl, "CW" ).setInteger( GL_CW );
      self->addClassProperty( c_sdlgl, "CCW" ).setInteger( GL_CCW );

      /* GetMapTarget */
      self->addClassProperty( c_sdlgl, "COEFF" ).setInteger( GL_COEFF );
      self->addClassProperty( c_sdlgl, "ORDER" ).setInteger( GL_ORDER );
      self->addClassProperty( c_sdlgl, "DOMAIN" ).setInteger( GL_DOMAIN );

      /* GetPixelMap */
      /*      GL_PIXEL_MAP_I_TO_I */
      /*      GL_PIXEL_MAP_S_TO_S */
      /*      GL_PIXEL_MAP_I_TO_R */
      /*      GL_PIXEL_MAP_I_TO_G */
      /*      GL_PIXEL_MAP_I_TO_B */
      /*      GL_PIXEL_MAP_I_TO_A */
      /*      GL_PIXEL_MAP_R_TO_R */
      /*      GL_PIXEL_MAP_G_TO_G */
      /*      GL_PIXEL_MAP_B_TO_B */
      /*      GL_PIXEL_MAP_A_TO_A */

      /* GetPointerTarget */
      /*      GL_VERTEX_ARRAY_POINTER */
      /*      GL_NORMAL_ARRAY_POINTER */
      /*      GL_COLOR_ARRAY_POINTER */
      /*      GL_INDEX_ARRAY_POINTER */
      /*      GL_TEXTURE_COORD_ARRAY_POINTER */
      /*      GL_EDGE_FLAG_ARRAY_POINTER */

      /* GetTarget */
      self->addClassProperty( c_sdlgl, "CURRENT_COLOR" ).setInteger( GL_CURRENT_COLOR );
      self->addClassProperty( c_sdlgl, "CURRENT_INDEX" ).setInteger( GL_CURRENT_INDEX );
      self->addClassProperty( c_sdlgl, "CURRENT_NORMAL" ).setInteger( GL_CURRENT_NORMAL );
      self->addClassProperty( c_sdlgl, "CURRENT_TEXTURE_COORDS" ).setInteger( GL_CURRENT_TEXTURE_COORDS );
      self->addClassProperty( c_sdlgl, "CURRENT_RASTER_COLOR" ).setInteger( GL_CURRENT_RASTER_COLOR );
      self->addClassProperty( c_sdlgl, "CURRENT_RASTER_INDEX" ).setInteger( GL_CURRENT_RASTER_INDEX );
      self->addClassProperty( c_sdlgl, "CURRENT_RASTER_TEXTURE_COORDS" ).setInteger( GL_CURRENT_RASTER_TEXTURE_COORDS );
      self->addClassProperty( c_sdlgl, "CURRENT_RASTER_POSITION" ).setInteger( GL_CURRENT_RASTER_POSITION );
      self->addClassProperty( c_sdlgl, "CURRENT_RASTER_POSITION_VALID" ).setInteger( GL_CURRENT_RASTER_POSITION_VALID );
      self->addClassProperty( c_sdlgl, "CURRENT_RASTER_DISTANCE" ).setInteger( GL_CURRENT_RASTER_DISTANCE );
      self->addClassProperty( c_sdlgl, "POINT_SMOOTH" ).setInteger( GL_POINT_SMOOTH );
      self->addClassProperty( c_sdlgl, "POINT_SIZE" ).setInteger( GL_POINT_SIZE );
      self->addClassProperty( c_sdlgl, "POINT_SIZE_RANGE" ).setInteger( GL_POINT_SIZE_RANGE );
      self->addClassProperty( c_sdlgl, "POINT_SIZE_GRANULARITY" ).setInteger( GL_POINT_SIZE_GRANULARITY );
      self->addClassProperty( c_sdlgl, "LINE_SMOOTH" ).setInteger( GL_LINE_SMOOTH );
      self->addClassProperty( c_sdlgl, "LINE_WIDTH" ).setInteger( GL_LINE_WIDTH );
      self->addClassProperty( c_sdlgl, "LINE_WIDTH_RANGE" ).setInteger( GL_LINE_WIDTH_RANGE );
      self->addClassProperty( c_sdlgl, "LINE_WIDTH_GRANULARITY" ).setInteger( GL_LINE_WIDTH_GRANULARITY );
      self->addClassProperty( c_sdlgl, "LINE_STIPPLE" ).setInteger( GL_LINE_STIPPLE );
      self->addClassProperty( c_sdlgl, "LINE_STIPPLE_PATTERN" ).setInteger( GL_LINE_STIPPLE_PATTERN );
      self->addClassProperty( c_sdlgl, "LINE_STIPPLE_REPEAT" ).setInteger( GL_LINE_STIPPLE_REPEAT );
      self->addClassProperty( c_sdlgl, "LIST_MODE" ).setInteger( GL_LIST_MODE );
      self->addClassProperty( c_sdlgl, "MAX_LIST_NESTING" ).setInteger( GL_MAX_LIST_NESTING );
      self->addClassProperty( c_sdlgl, "LIST_BASE" ).setInteger( GL_LIST_BASE );
      self->addClassProperty( c_sdlgl, "LIST_INDEX" ).setInteger( GL_LIST_INDEX );
      self->addClassProperty( c_sdlgl, "POLYGON_MODE" ).setInteger( GL_POLYGON_MODE );
      self->addClassProperty( c_sdlgl, "POLYGON_SMOOTH" ).setInteger( GL_POLYGON_SMOOTH );
      self->addClassProperty( c_sdlgl, "POLYGON_STIPPLE" ).setInteger( GL_POLYGON_STIPPLE );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG" ).setInteger( GL_EDGE_FLAG );
      self->addClassProperty( c_sdlgl, "CULL_FACE" ).setInteger( GL_CULL_FACE );
      self->addClassProperty( c_sdlgl, "CULL_FACE_MODE" ).setInteger( GL_CULL_FACE_MODE );
      self->addClassProperty( c_sdlgl, "FRONT_FACE" ).setInteger( GL_FRONT_FACE );
      self->addClassProperty( c_sdlgl, "LIGHTING" ).setInteger( GL_LIGHTING );
      self->addClassProperty( c_sdlgl, "LIGHT_MODEL_LOCAL_VIEWER" ).setInteger( GL_LIGHT_MODEL_LOCAL_VIEWER );
      self->addClassProperty( c_sdlgl, "LIGHT_MODEL_TWO_SIDE" ).setInteger( GL_LIGHT_MODEL_TWO_SIDE );
      self->addClassProperty( c_sdlgl, "LIGHT_MODEL_AMBIENT" ).setInteger( GL_LIGHT_MODEL_AMBIENT );
      self->addClassProperty( c_sdlgl, "SHADE_MODEL" ).setInteger( GL_SHADE_MODEL );
      self->addClassProperty( c_sdlgl, "COLOR_MATERIAL_FACE" ).setInteger( GL_COLOR_MATERIAL_FACE );
      self->addClassProperty( c_sdlgl, "COLOR_MATERIAL_PARAMETER" ).setInteger( GL_COLOR_MATERIAL_PARAMETER );
      self->addClassProperty( c_sdlgl, "COLOR_MATERIAL" ).setInteger( GL_COLOR_MATERIAL );
      self->addClassProperty( c_sdlgl, "FOG" ).setInteger( GL_FOG );
      self->addClassProperty( c_sdlgl, "FOG_INDEX" ).setInteger( GL_FOG_INDEX );
      self->addClassProperty( c_sdlgl, "FOG_DENSITY" ).setInteger( GL_FOG_DENSITY );
      self->addClassProperty( c_sdlgl, "FOG_START" ).setInteger( GL_FOG_START );
      self->addClassProperty( c_sdlgl, "FOG_END" ).setInteger( GL_FOG_END );
      self->addClassProperty( c_sdlgl, "FOG_MODE" ).setInteger( GL_FOG_MODE );
      self->addClassProperty( c_sdlgl, "FOG_COLOR" ).setInteger( GL_FOG_COLOR );
      self->addClassProperty( c_sdlgl, "DEPTH_RANGE" ).setInteger( GL_DEPTH_RANGE );
      self->addClassProperty( c_sdlgl, "DEPTH_TEST" ).setInteger( GL_DEPTH_TEST );
      self->addClassProperty( c_sdlgl, "DEPTH_WRITEMASK" ).setInteger( GL_DEPTH_WRITEMASK );
      self->addClassProperty( c_sdlgl, "DEPTH_CLEAR_VALUE" ).setInteger( GL_DEPTH_CLEAR_VALUE );
      self->addClassProperty( c_sdlgl, "DEPTH_FUNC" ).setInteger( GL_DEPTH_FUNC );
      self->addClassProperty( c_sdlgl, "ACCUM_CLEAR_VALUE" ).setInteger( GL_ACCUM_CLEAR_VALUE );
      self->addClassProperty( c_sdlgl, "STENCIL_TEST" ).setInteger( GL_STENCIL_TEST );
      self->addClassProperty( c_sdlgl, "STENCIL_CLEAR_VALUE" ).setInteger( GL_STENCIL_CLEAR_VALUE );
      self->addClassProperty( c_sdlgl, "STENCIL_FUNC" ).setInteger( GL_STENCIL_FUNC );
      self->addClassProperty( c_sdlgl, "STENCIL_VALUE_MASK" ).setInteger( GL_STENCIL_VALUE_MASK );
      self->addClassProperty( c_sdlgl, "STENCIL_FAIL" ).setInteger( GL_STENCIL_FAIL );
      self->addClassProperty( c_sdlgl, "STENCIL_PASS_DEPTH_FAIL" ).setInteger( GL_STENCIL_PASS_DEPTH_FAIL );
      self->addClassProperty( c_sdlgl, "STENCIL_PASS_DEPTH_PASS" ).setInteger( GL_STENCIL_PASS_DEPTH_PASS );
      self->addClassProperty( c_sdlgl, "STENCIL_REF" ).setInteger( GL_STENCIL_REF );
      self->addClassProperty( c_sdlgl, "STENCIL_WRITEMASK" ).setInteger( GL_STENCIL_WRITEMASK );
      self->addClassProperty( c_sdlgl, "MATRIX_MODE" ).setInteger( GL_MATRIX_MODE );
      self->addClassProperty( c_sdlgl, "NORMALIZE" ).setInteger( GL_NORMALIZE );
      self->addClassProperty( c_sdlgl, "VIEWPORT" ).setInteger( GL_VIEWPORT );
      self->addClassProperty( c_sdlgl, "MODELVIEW_STACK_DEPTH" ).setInteger( GL_MODELVIEW_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "PROJECTION_STACK_DEPTH" ).setInteger( GL_PROJECTION_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "TEXTURE_STACK_DEPTH" ).setInteger( GL_TEXTURE_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "MODELVIEW_MATRIX" ).setInteger( GL_MODELVIEW_MATRIX );
      self->addClassProperty( c_sdlgl, "PROJECTION_MATRIX" ).setInteger( GL_PROJECTION_MATRIX );
      self->addClassProperty( c_sdlgl, "TEXTURE_MATRIX" ).setInteger( GL_TEXTURE_MATRIX );
      self->addClassProperty( c_sdlgl, "ATTRIB_STACK_DEPTH" ).setInteger( GL_ATTRIB_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "CLIENT_ATTRIB_STACK_DEPTH" ).setInteger( GL_CLIENT_ATTRIB_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "ALPHA_TEST" ).setInteger( GL_ALPHA_TEST );
      self->addClassProperty( c_sdlgl, "ALPHA_TEST_FUNC" ).setInteger( GL_ALPHA_TEST_FUNC );
      self->addClassProperty( c_sdlgl, "ALPHA_TEST_REF" ).setInteger( GL_ALPHA_TEST_REF );
      self->addClassProperty( c_sdlgl, "DITHER" ).setInteger( GL_DITHER );
      self->addClassProperty( c_sdlgl, "BLEND_DST" ).setInteger( GL_BLEND_DST );
      self->addClassProperty( c_sdlgl, "BLEND_SRC" ).setInteger( GL_BLEND_SRC );
      self->addClassProperty( c_sdlgl, "BLEND" ).setInteger( GL_BLEND );
      self->addClassProperty( c_sdlgl, "LOGIC_OP_MODE" ).setInteger( GL_LOGIC_OP_MODE );
      self->addClassProperty( c_sdlgl, "INDEX_LOGIC_OP" ).setInteger( GL_INDEX_LOGIC_OP );
      self->addClassProperty( c_sdlgl, "COLOR_LOGIC_OP" ).setInteger( GL_COLOR_LOGIC_OP );
      self->addClassProperty( c_sdlgl, "AUX_BUFFERS" ).setInteger( GL_AUX_BUFFERS );
      self->addClassProperty( c_sdlgl, "DRAW_BUFFER" ).setInteger( GL_DRAW_BUFFER );
      self->addClassProperty( c_sdlgl, "READ_BUFFER" ).setInteger( GL_READ_BUFFER );
      self->addClassProperty( c_sdlgl, "SCISSOR_BOX" ).setInteger( GL_SCISSOR_BOX );
      self->addClassProperty( c_sdlgl, "SCISSOR_TEST" ).setInteger( GL_SCISSOR_TEST );
      self->addClassProperty( c_sdlgl, "INDEX_CLEAR_VALUE" ).setInteger( GL_INDEX_CLEAR_VALUE );
      self->addClassProperty( c_sdlgl, "INDEX_WRITEMASK" ).setInteger( GL_INDEX_WRITEMASK );
      self->addClassProperty( c_sdlgl, "COLOR_CLEAR_VALUE" ).setInteger( GL_COLOR_CLEAR_VALUE );
      self->addClassProperty( c_sdlgl, "COLOR_WRITEMASK" ).setInteger( GL_COLOR_WRITEMASK );
      self->addClassProperty( c_sdlgl, "INDEX_MODE" ).setInteger( GL_INDEX_MODE );
      self->addClassProperty( c_sdlgl, "RGBA_MODE" ).setInteger( GL_RGBA_MODE );
      self->addClassProperty( c_sdlgl, "DOUBLEBUFFER" ).setInteger( GL_DOUBLEBUFFER );
      self->addClassProperty( c_sdlgl, "STEREO" ).setInteger( GL_STEREO );
      self->addClassProperty( c_sdlgl, "RENDER_MODE" ).setInteger( GL_RENDER_MODE );
      self->addClassProperty( c_sdlgl, "PERSPECTIVE_CORRECTION_HINT" ).setInteger( GL_PERSPECTIVE_CORRECTION_HINT );
      self->addClassProperty( c_sdlgl, "POINT_SMOOTH_HINT" ).setInteger( GL_POINT_SMOOTH_HINT );
      self->addClassProperty( c_sdlgl, "LINE_SMOOTH_HINT" ).setInteger( GL_LINE_SMOOTH_HINT );
      self->addClassProperty( c_sdlgl, "POLYGON_SMOOTH_HINT" ).setInteger( GL_POLYGON_SMOOTH_HINT );
      self->addClassProperty( c_sdlgl, "FOG_HINT" ).setInteger( GL_FOG_HINT );
      self->addClassProperty( c_sdlgl, "TEXTURE_GEN_S" ).setInteger( GL_TEXTURE_GEN_S );
      self->addClassProperty( c_sdlgl, "TEXTURE_GEN_T" ).setInteger( GL_TEXTURE_GEN_T );
      self->addClassProperty( c_sdlgl, "TEXTURE_GEN_R" ).setInteger( GL_TEXTURE_GEN_R );
      self->addClassProperty( c_sdlgl, "TEXTURE_GEN_Q" ).setInteger( GL_TEXTURE_GEN_Q );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_I" ).setInteger( GL_PIXEL_MAP_I_TO_I );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_S_TO_S" ).setInteger( GL_PIXEL_MAP_S_TO_S );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_R" ).setInteger( GL_PIXEL_MAP_I_TO_R );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_G" ).setInteger( GL_PIXEL_MAP_I_TO_G );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_B" ).setInteger( GL_PIXEL_MAP_I_TO_B );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_A" ).setInteger( GL_PIXEL_MAP_I_TO_A );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_R_TO_R" ).setInteger( GL_PIXEL_MAP_R_TO_R );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_G_TO_G" ).setInteger( GL_PIXEL_MAP_G_TO_G );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_B_TO_B" ).setInteger( GL_PIXEL_MAP_B_TO_B );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_A_TO_A" ).setInteger( GL_PIXEL_MAP_A_TO_A );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_I_SIZE" ).setInteger( GL_PIXEL_MAP_I_TO_I_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_S_TO_S_SIZE" ).setInteger( GL_PIXEL_MAP_S_TO_S_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_R_SIZE" ).setInteger( GL_PIXEL_MAP_I_TO_R_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_G_SIZE" ).setInteger( GL_PIXEL_MAP_I_TO_G_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_B_SIZE" ).setInteger( GL_PIXEL_MAP_I_TO_B_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_I_TO_A_SIZE" ).setInteger( GL_PIXEL_MAP_I_TO_A_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_R_TO_R_SIZE" ).setInteger( GL_PIXEL_MAP_R_TO_R_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_G_TO_G_SIZE" ).setInteger( GL_PIXEL_MAP_G_TO_G_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_B_TO_B_SIZE" ).setInteger( GL_PIXEL_MAP_B_TO_B_SIZE );
      self->addClassProperty( c_sdlgl, "PIXEL_MAP_A_TO_A_SIZE" ).setInteger( GL_PIXEL_MAP_A_TO_A_SIZE );
      self->addClassProperty( c_sdlgl, "UNPACK_SWAP_BYTES" ).setInteger( GL_UNPACK_SWAP_BYTES );
      self->addClassProperty( c_sdlgl, "UNPACK_LSB_FIRST" ).setInteger( GL_UNPACK_LSB_FIRST );
      self->addClassProperty( c_sdlgl, "UNPACK_ROW_LENGTH" ).setInteger( GL_UNPACK_ROW_LENGTH );
      self->addClassProperty( c_sdlgl, "UNPACK_SKIP_ROWS" ).setInteger( GL_UNPACK_SKIP_ROWS );
      self->addClassProperty( c_sdlgl, "UNPACK_SKIP_PIXELS" ).setInteger( GL_UNPACK_SKIP_PIXELS );
      self->addClassProperty( c_sdlgl, "UNPACK_ALIGNMENT" ).setInteger( GL_UNPACK_ALIGNMENT );
      self->addClassProperty( c_sdlgl, "PACK_SWAP_BYTES" ).setInteger( GL_PACK_SWAP_BYTES );
      self->addClassProperty( c_sdlgl, "PACK_LSB_FIRST" ).setInteger( GL_PACK_LSB_FIRST );
      self->addClassProperty( c_sdlgl, "PACK_ROW_LENGTH" ).setInteger( GL_PACK_ROW_LENGTH );
      self->addClassProperty( c_sdlgl, "PACK_SKIP_ROWS" ).setInteger( GL_PACK_SKIP_ROWS );
      self->addClassProperty( c_sdlgl, "PACK_SKIP_PIXELS" ).setInteger( GL_PACK_SKIP_PIXELS );
      self->addClassProperty( c_sdlgl, "PACK_ALIGNMENT" ).setInteger( GL_PACK_ALIGNMENT );
      self->addClassProperty( c_sdlgl, "MAP_COLOR" ).setInteger( GL_MAP_COLOR );
      self->addClassProperty( c_sdlgl, "MAP_STENCIL" ).setInteger( GL_MAP_STENCIL );
      self->addClassProperty( c_sdlgl, "INDEX_SHIFT" ).setInteger( GL_INDEX_SHIFT );
      self->addClassProperty( c_sdlgl, "INDEX_OFFSET" ).setInteger( GL_INDEX_OFFSET );
      self->addClassProperty( c_sdlgl, "RED_SCALE" ).setInteger( GL_RED_SCALE );
      self->addClassProperty( c_sdlgl, "RED_BIAS" ).setInteger( GL_RED_BIAS );
      self->addClassProperty( c_sdlgl, "ZOOM_X" ).setInteger( GL_ZOOM_X );
      self->addClassProperty( c_sdlgl, "ZOOM_Y" ).setInteger( GL_ZOOM_Y );
      self->addClassProperty( c_sdlgl, "GREEN_SCALE" ).setInteger( GL_GREEN_SCALE );
      self->addClassProperty( c_sdlgl, "GREEN_BIAS" ).setInteger( GL_GREEN_BIAS );
      self->addClassProperty( c_sdlgl, "BLUE_SCALE" ).setInteger( GL_BLUE_SCALE );
      self->addClassProperty( c_sdlgl, "BLUE_BIAS" ).setInteger( GL_BLUE_BIAS );
      self->addClassProperty( c_sdlgl, "ALPHA_SCALE" ).setInteger( GL_ALPHA_SCALE );
      self->addClassProperty( c_sdlgl, "ALPHA_BIAS" ).setInteger( GL_ALPHA_BIAS );
      self->addClassProperty( c_sdlgl, "DEPTH_SCALE" ).setInteger( GL_DEPTH_SCALE );
      self->addClassProperty( c_sdlgl, "DEPTH_BIAS" ).setInteger( GL_DEPTH_BIAS );
      self->addClassProperty( c_sdlgl, "MAX_EVAL_ORDER" ).setInteger( GL_MAX_EVAL_ORDER );
      self->addClassProperty( c_sdlgl, "MAX_LIGHTS" ).setInteger( GL_MAX_LIGHTS );
      self->addClassProperty( c_sdlgl, "MAX_CLIP_PLANES" ).setInteger( GL_MAX_CLIP_PLANES );
      self->addClassProperty( c_sdlgl, "MAX_TEXTURE_SIZE" ).setInteger( GL_MAX_TEXTURE_SIZE );
      self->addClassProperty( c_sdlgl, "MAX_PIXEL_MAP_TABLE" ).setInteger( GL_MAX_PIXEL_MAP_TABLE );
      self->addClassProperty( c_sdlgl, "MAX_ATTRIB_STACK_DEPTH" ).setInteger( GL_MAX_ATTRIB_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "MAX_MODELVIEW_STACK_DEPTH" ).setInteger( GL_MAX_MODELVIEW_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "MAX_NAME_STACK_DEPTH" ).setInteger( GL_MAX_NAME_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "MAX_PROJECTION_STACK_DEPTH" ).setInteger( GL_MAX_PROJECTION_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "MAX_TEXTURE_STACK_DEPTH" ).setInteger( GL_MAX_TEXTURE_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "MAX_VIEWPORT_DIMS" ).setInteger( GL_MAX_VIEWPORT_DIMS );
      self->addClassProperty( c_sdlgl, "MAX_CLIENT_ATTRIB_STACK_DEPTH" ).setInteger( GL_MAX_CLIENT_ATTRIB_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "SUBPIXEL_BITS" ).setInteger( GL_SUBPIXEL_BITS );
      self->addClassProperty( c_sdlgl, "INDEX_BITS" ).setInteger( GL_INDEX_BITS );
      self->addClassProperty( c_sdlgl, "RED_BITS" ).setInteger( GL_RED_BITS );
      self->addClassProperty( c_sdlgl, "GREEN_BITS" ).setInteger( GL_GREEN_BITS );
      self->addClassProperty( c_sdlgl, "BLUE_BITS" ).setInteger( GL_BLUE_BITS );
      self->addClassProperty( c_sdlgl, "ALPHA_BITS" ).setInteger( GL_ALPHA_BITS );
      self->addClassProperty( c_sdlgl, "DEPTH_BITS" ).setInteger( GL_DEPTH_BITS );
      self->addClassProperty( c_sdlgl, "STENCIL_BITS" ).setInteger( GL_STENCIL_BITS );
      self->addClassProperty( c_sdlgl, "ACCUM_RED_BITS" ).setInteger( GL_ACCUM_RED_BITS );
      self->addClassProperty( c_sdlgl, "ACCUM_GREEN_BITS" ).setInteger( GL_ACCUM_GREEN_BITS );
      self->addClassProperty( c_sdlgl, "ACCUM_BLUE_BITS" ).setInteger( GL_ACCUM_BLUE_BITS );
      self->addClassProperty( c_sdlgl, "ACCUM_ALPHA_BITS" ).setInteger( GL_ACCUM_ALPHA_BITS );
      self->addClassProperty( c_sdlgl, "NAME_STACK_DEPTH" ).setInteger( GL_NAME_STACK_DEPTH );
      self->addClassProperty( c_sdlgl, "AUTO_NORMAL" ).setInteger( GL_AUTO_NORMAL );
      self->addClassProperty( c_sdlgl, "MAP1_COLOR_4" ).setInteger( GL_MAP1_COLOR_4 );
      self->addClassProperty( c_sdlgl, "MAP1_INDEX" ).setInteger( GL_MAP1_INDEX );
      self->addClassProperty( c_sdlgl, "MAP1_NORMAL" ).setInteger( GL_MAP1_NORMAL );
      self->addClassProperty( c_sdlgl, "MAP1_TEXTURE_COORD_1" ).setInteger( GL_MAP1_TEXTURE_COORD_1 );
      self->addClassProperty( c_sdlgl, "MAP1_TEXTURE_COORD_2" ).setInteger( GL_MAP1_TEXTURE_COORD_2 );
      self->addClassProperty( c_sdlgl, "MAP1_TEXTURE_COORD_3" ).setInteger( GL_MAP1_TEXTURE_COORD_3 );
      self->addClassProperty( c_sdlgl, "MAP1_TEXTURE_COORD_4" ).setInteger( GL_MAP1_TEXTURE_COORD_4 );
      self->addClassProperty( c_sdlgl, "MAP1_VERTEX_3" ).setInteger( GL_MAP1_VERTEX_3 );
      self->addClassProperty( c_sdlgl, "MAP1_VERTEX_4" ).setInteger( GL_MAP1_VERTEX_4 );
      self->addClassProperty( c_sdlgl, "MAP2_COLOR_4" ).setInteger( GL_MAP2_COLOR_4 );
      self->addClassProperty( c_sdlgl, "MAP2_INDEX" ).setInteger( GL_MAP2_INDEX );
      self->addClassProperty( c_sdlgl, "MAP2_NORMAL" ).setInteger( GL_MAP2_NORMAL );
      self->addClassProperty( c_sdlgl, "MAP2_TEXTURE_COORD_1" ).setInteger( GL_MAP2_TEXTURE_COORD_1 );
      self->addClassProperty( c_sdlgl, "MAP2_TEXTURE_COORD_2" ).setInteger( GL_MAP2_TEXTURE_COORD_2 );
      self->addClassProperty( c_sdlgl, "MAP2_TEXTURE_COORD_3" ).setInteger( GL_MAP2_TEXTURE_COORD_3 );
      self->addClassProperty( c_sdlgl, "MAP2_TEXTURE_COORD_4" ).setInteger( GL_MAP2_TEXTURE_COORD_4 );
      self->addClassProperty( c_sdlgl, "MAP2_VERTEX_3" ).setInteger( GL_MAP2_VERTEX_3 );
      self->addClassProperty( c_sdlgl, "MAP2_VERTEX_4" ).setInteger( GL_MAP2_VERTEX_4 );
      self->addClassProperty( c_sdlgl, "MAP1_GRID_DOMAIN" ).setInteger( GL_MAP1_GRID_DOMAIN );
      self->addClassProperty( c_sdlgl, "MAP1_GRID_SEGMENTS" ).setInteger( GL_MAP1_GRID_SEGMENTS );
      self->addClassProperty( c_sdlgl, "MAP2_GRID_DOMAIN" ).setInteger( GL_MAP2_GRID_DOMAIN );
      self->addClassProperty( c_sdlgl, "MAP2_GRID_SEGMENTS" ).setInteger( GL_MAP2_GRID_SEGMENTS );
      self->addClassProperty( c_sdlgl, "TEXTURE_1D" ).setInteger( GL_TEXTURE_1D );
      self->addClassProperty( c_sdlgl, "TEXTURE_2D" ).setInteger( GL_TEXTURE_2D );
      self->addClassProperty( c_sdlgl, "FEEDBACK_BUFFER_POINTER" ).setInteger( GL_FEEDBACK_BUFFER_POINTER );
      self->addClassProperty( c_sdlgl, "FEEDBACK_BUFFER_SIZE" ).setInteger( GL_FEEDBACK_BUFFER_SIZE );
      self->addClassProperty( c_sdlgl, "FEEDBACK_BUFFER_TYPE" ).setInteger( GL_FEEDBACK_BUFFER_TYPE );
      self->addClassProperty( c_sdlgl, "SELECTION_BUFFER_POINTER" ).setInteger( GL_SELECTION_BUFFER_POINTER );
      self->addClassProperty( c_sdlgl, "SELECTION_BUFFER_SIZE" ).setInteger( GL_SELECTION_BUFFER_SIZE );
      /*      GL_TEXTURE_BINDING_1D */
      /*      GL_TEXTURE_BINDING_2D */
      /*      GL_VERTEX_ARRAY */
      /*      GL_NORMAL_ARRAY */
      /*      GL_COLOR_ARRAY */
      /*      GL_INDEX_ARRAY */
      /*      GL_TEXTURE_COORD_ARRAY */
      /*      GL_EDGE_FLAG_ARRAY */
      /*      GL_VERTEX_ARRAY_SIZE */
      /*      GL_VERTEX_ARRAY_TYPE */
      /*      GL_VERTEX_ARRAY_STRIDE */
      /*      GL_NORMAL_ARRAY_TYPE */
      /*      GL_NORMAL_ARRAY_STRIDE */
      /*      GL_COLOR_ARRAY_SIZE */
      /*      GL_COLOR_ARRAY_TYPE */
      /*      GL_COLOR_ARRAY_STRIDE */
      /*      GL_INDEX_ARRAY_TYPE */
      /*      GL_INDEX_ARRAY_STRIDE */
      /*      GL_TEXTURE_COORD_ARRAY_SIZE */
      /*      GL_TEXTURE_COORD_ARRAY_TYPE */
      /*      GL_TEXTURE_COORD_ARRAY_STRIDE */
      /*      GL_EDGE_FLAG_ARRAY_STRIDE */
      /*      GL_POLYGON_OFFSET_FACTOR */
      /*      GL_POLYGON_OFFSET_UNITS */

      /* GetTextureParameter */
      /*      GL_TEXTURE_MAG_FILTER */
      /*      GL_TEXTURE_MIN_FILTER */
      /*      GL_TEXTURE_WRAP_S */
      /*      GL_TEXTURE_WRAP_T */
      self->addClassProperty( c_sdlgl, "TEXTURE_WIDTH" ).setInteger( GL_TEXTURE_WIDTH );
      self->addClassProperty( c_sdlgl, "TEXTURE_HEIGHT" ).setInteger( GL_TEXTURE_HEIGHT );
      self->addClassProperty( c_sdlgl, "TEXTURE_INTERNAL_FORMAT" ).setInteger( GL_TEXTURE_INTERNAL_FORMAT );
      self->addClassProperty( c_sdlgl, "TEXTURE_BORDER_COLOR" ).setInteger( GL_TEXTURE_BORDER_COLOR );
      self->addClassProperty( c_sdlgl, "TEXTURE_BORDER" ).setInteger( GL_TEXTURE_BORDER );
      /*      GL_TEXTURE_RED_SIZE */
      /*      GL_TEXTURE_GREEN_SIZE */
      /*      GL_TEXTURE_BLUE_SIZE */
      /*      GL_TEXTURE_ALPHA_SIZE */
      /*      GL_TEXTURE_LUMINANCE_SIZE */
      /*      GL_TEXTURE_INTENSITY_SIZE */
      /*      GL_TEXTURE_PRIORITY */
      /*      GL_TEXTURE_RESIDENT */

      /* HintMode */
      self->addClassProperty( c_sdlgl, "DONT_CARE" ).setInteger( GL_DONT_CARE );
      self->addClassProperty( c_sdlgl, "FASTEST" ).setInteger( GL_FASTEST );
      self->addClassProperty( c_sdlgl, "NICEST" ).setInteger( GL_NICEST );

      /* HintTarget */
      /*      GL_PERSPECTIVE_CORRECTION_HINT */
      /*      GL_POINT_SMOOTH_HINT */
      /*      GL_LINE_SMOOTH_HINT */
      /*      GL_POLYGON_SMOOTH_HINT */
      /*      GL_FOG_HINT */
      /*      GL_PHONG_HINT */

      /* IndexPointerType */
      /*      GL_SHORT */
      /*      GL_INT */
      /*      GL_FLOAT */
      /*      GL_DOUBLE */

      /* LightModelParameter */
      /*      GL_LIGHT_MODEL_AMBIENT */
      /*      GL_LIGHT_MODEL_LOCAL_VIEWER */
      /*      GL_LIGHT_MODEL_TWO_SIDE */

      /* LightName */
      self->addClassProperty( c_sdlgl, "LIGHT0" ).setInteger( GL_LIGHT0 );
      self->addClassProperty( c_sdlgl, "LIGHT1" ).setInteger( GL_LIGHT1 );
      self->addClassProperty( c_sdlgl, "LIGHT2" ).setInteger( GL_LIGHT2 );
      self->addClassProperty( c_sdlgl, "LIGHT3" ).setInteger( GL_LIGHT3 );
      self->addClassProperty( c_sdlgl, "LIGHT4" ).setInteger( GL_LIGHT4 );
      self->addClassProperty( c_sdlgl, "LIGHT5" ).setInteger( GL_LIGHT5 );
      self->addClassProperty( c_sdlgl, "LIGHT6" ).setInteger( GL_LIGHT6 );
      self->addClassProperty( c_sdlgl, "LIGHT7" ).setInteger( GL_LIGHT7 );

      /* LightParameter */
      self->addClassProperty( c_sdlgl, "AMBIENT" ).setInteger( GL_AMBIENT );
      self->addClassProperty( c_sdlgl, "DIFFUSE" ).setInteger( GL_DIFFUSE );
      self->addClassProperty( c_sdlgl, "SPECULAR" ).setInteger( GL_SPECULAR );
      self->addClassProperty( c_sdlgl, "POSITION" ).setInteger( GL_POSITION );
      self->addClassProperty( c_sdlgl, "SPOT_DIRECTION" ).setInteger( GL_SPOT_DIRECTION );
      self->addClassProperty( c_sdlgl, "SPOT_EXPONENT" ).setInteger( GL_SPOT_EXPONENT );
      self->addClassProperty( c_sdlgl, "SPOT_CUTOFF" ).setInteger( GL_SPOT_CUTOFF );
      self->addClassProperty( c_sdlgl, "CONSTANT_ATTENUATION" ).setInteger( GL_CONSTANT_ATTENUATION );
      self->addClassProperty( c_sdlgl, "LINEAR_ATTENUATION" ).setInteger( GL_LINEAR_ATTENUATION );
      self->addClassProperty( c_sdlgl, "QUADRATIC_ATTENUATION" ).setInteger( GL_QUADRATIC_ATTENUATION );

      /* InterleavedArrays */
      /*      GL_V2F */
      /*      GL_V3F */
      /*      GL_C4UB_V2F */
      /*      GL_C4UB_V3F */
      /*      GL_C3F_V3F */
      /*      GL_N3F_V3F */
      /*      GL_C4F_N3F_V3F */
      /*      GL_T2F_V3F */
      /*      GL_T4F_V4F */
      /*      GL_T2F_C4UB_V3F */
      /*      GL_T2F_C3F_V3F */
      /*      GL_T2F_N3F_V3F */
      /*      GL_T2F_C4F_N3F_V3F */
      /*      GL_T4F_C4F_N3F_V4F */

      /* ListMode */
      self->addClassProperty( c_sdlgl, "COMPILE" ).setInteger( GL_COMPILE );
      self->addClassProperty( c_sdlgl, "COMPILE_AND_EXECUTE" ).setInteger( GL_COMPILE_AND_EXECUTE );

      /* ListNameType */
      /*      GL_BYTE */
      /*      GL_UNSIGNED_BYTE */
      /*      GL_SHORT */
      /*      GL_UNSIGNED_SHORT */
      /*      GL_INT */
      /*      GL_UNSIGNED_INT */
      /*      GL_FLOAT */
      /*      GL_2_BYTES */
      /*      GL_3_BYTES */
      /*      GL_4_BYTES */

      /* LogicOp */
      self->addClassProperty( c_sdlgl, "CLEAR" ).setInteger( GL_CLEAR );
      self->addClassProperty( c_sdlgl, "AND" ).setInteger( GL_AND );
      self->addClassProperty( c_sdlgl, "AND_REVERSE" ).setInteger( GL_AND_REVERSE );
      self->addClassProperty( c_sdlgl, "COPY" ).setInteger( GL_COPY );
      self->addClassProperty( c_sdlgl, "AND_INVERTED" ).setInteger( GL_AND_INVERTED );
      self->addClassProperty( c_sdlgl, "NOOP" ).setInteger( GL_NOOP );
      self->addClassProperty( c_sdlgl, "XOR" ).setInteger( GL_XOR );
      self->addClassProperty( c_sdlgl, "OR" ).setInteger( GL_OR );
      self->addClassProperty( c_sdlgl, "NOR" ).setInteger( GL_NOR );
      self->addClassProperty( c_sdlgl, "EQUIV" ).setInteger( GL_EQUIV );
      self->addClassProperty( c_sdlgl, "INVERT" ).setInteger( GL_INVERT );
      self->addClassProperty( c_sdlgl, "OR_REVERSE" ).setInteger( GL_OR_REVERSE );
      self->addClassProperty( c_sdlgl, "COPY_INVERTED" ).setInteger( GL_COPY_INVERTED );
      self->addClassProperty( c_sdlgl, "OR_INVERTED" ).setInteger( GL_OR_INVERTED );
      self->addClassProperty( c_sdlgl, "NAND" ).setInteger( GL_NAND );
      self->addClassProperty( c_sdlgl, "SET" ).setInteger( GL_SET );

      /* MapTarget */
      /*      GL_MAP1_COLOR_4 */
      /*      GL_MAP1_INDEX */
      /*      GL_MAP1_NORMAL */
      /*      GL_MAP1_TEXTURE_COORD_1 */
      /*      GL_MAP1_TEXTURE_COORD_2 */
      /*      GL_MAP1_TEXTURE_COORD_3 */
      /*      GL_MAP1_TEXTURE_COORD_4 */
      /*      GL_MAP1_VERTEX_3 */
      /*      GL_MAP1_VERTEX_4 */
      /*      GL_MAP2_COLOR_4 */
      /*      GL_MAP2_INDEX */
      /*      GL_MAP2_NORMAL */
      /*      GL_MAP2_TEXTURE_COORD_1 */
      /*      GL_MAP2_TEXTURE_COORD_2 */
      /*      GL_MAP2_TEXTURE_COORD_3 */
      /*      GL_MAP2_TEXTURE_COORD_4 */
      /*      GL_MAP2_VERTEX_3 */
      /*      GL_MAP2_VERTEX_4 */

      /* MaterialFace */
      /*      GL_FRONT */
      /*      GL_BACK */
      /*      GL_FRONT_AND_BACK */

      /* MaterialParameter */
      self->addClassProperty( c_sdlgl, "EMISSION" ).setInteger( GL_EMISSION );
      self->addClassProperty( c_sdlgl, "SHININESS" ).setInteger( GL_SHININESS );
      self->addClassProperty( c_sdlgl, "AMBIENT_AND_DIFFUSE" ).setInteger( GL_AMBIENT_AND_DIFFUSE );
      self->addClassProperty( c_sdlgl, "COLOR_INDEXES" ).setInteger( GL_COLOR_INDEXES );
      /*      GL_AMBIENT */
      /*      GL_DIFFUSE */
      /*      GL_SPECULAR */

      /* MatrixMode */
      self->addClassProperty( c_sdlgl, "MODELVIEW" ).setInteger( GL_MODELVIEW );
      self->addClassProperty( c_sdlgl, "PROJECTION" ).setInteger( GL_PROJECTION );
      self->addClassProperty( c_sdlgl, "TEXTURE" ).setInteger( GL_TEXTURE );

      /* MeshMode1 */
      /*      GL_POINT */
      /*      GL_LINE */

      /* MeshMode2 */
      /*      GL_POINT */
      /*      GL_LINE */
      /*      GL_FILL */

      /* NormalPointerType */
      /*      GL_BYTE */
      /*      GL_SHORT */
      /*      GL_INT */
      /*      GL_FLOAT */
      /*      GL_DOUBLE */

      /* PixelCopyType */
      self->addClassProperty( c_sdlgl, "COLOR" ).setInteger( GL_COLOR );
      self->addClassProperty( c_sdlgl, "DEPTH" ).setInteger( GL_DEPTH );
      self->addClassProperty( c_sdlgl, "STENCIL" ).setInteger( GL_STENCIL );

      /* PixelFormat */
      self->addClassProperty( c_sdlgl, "COLOR_INDEX" ).setInteger( GL_COLOR_INDEX );
      self->addClassProperty( c_sdlgl, "STENCIL_INDEX" ).setInteger( GL_STENCIL_INDEX );
      self->addClassProperty( c_sdlgl, "DEPTH_COMPONENT" ).setInteger( GL_DEPTH_COMPONENT );
      self->addClassProperty( c_sdlgl, "RED" ).setInteger( GL_RED );
      self->addClassProperty( c_sdlgl, "GREEN" ).setInteger( GL_GREEN );
      self->addClassProperty( c_sdlgl, "BLUE" ).setInteger( GL_BLUE );
      self->addClassProperty( c_sdlgl, "ALPHA" ).setInteger( GL_ALPHA );
      self->addClassProperty( c_sdlgl, "RGB" ).setInteger( GL_RGB );
      self->addClassProperty( c_sdlgl, "RGBA" ).setInteger( GL_RGBA );
      self->addClassProperty( c_sdlgl, "LUMINANCE" ).setInteger( GL_LUMINANCE );
      self->addClassProperty( c_sdlgl, "LUMINANCE_ALPHA" ).setInteger( GL_LUMINANCE_ALPHA );

      /* PixelMap */
      /*      GL_PIXEL_MAP_I_TO_I */
      /*      GL_PIXEL_MAP_S_TO_S */
      /*      GL_PIXEL_MAP_I_TO_R */
      /*      GL_PIXEL_MAP_I_TO_G */
      /*      GL_PIXEL_MAP_I_TO_B */
      /*      GL_PIXEL_MAP_I_TO_A */
      /*      GL_PIXEL_MAP_R_TO_R */
      /*      GL_PIXEL_MAP_G_TO_G */
      /*      GL_PIXEL_MAP_B_TO_B */
      /*      GL_PIXEL_MAP_A_TO_A */

      /* PixelStore */
      /*      GL_UNPACK_SWAP_BYTES */
      /*      GL_UNPACK_LSB_FIRST */
      /*      GL_UNPACK_ROW_LENGTH */
      /*      GL_UNPACK_SKIP_ROWS */
      /*      GL_UNPACK_SKIP_PIXELS */
      /*      GL_UNPACK_ALIGNMENT */
      /*      GL_PACK_SWAP_BYTES */
      /*      GL_PACK_LSB_FIRST */
      /*      GL_PACK_ROW_LENGTH */
      /*      GL_PACK_SKIP_ROWS */
      /*      GL_PACK_SKIP_PIXELS */
      /*      GL_PACK_ALIGNMENT */

      /* PixelTransfer */
      /*      GL_MAP_COLOR */
      /*      GL_MAP_STENCIL */
      /*      GL_INDEX_SHIFT */
      /*      GL_INDEX_OFFSET */
      /*      GL_RED_SCALE */
      /*      GL_RED_BIAS */
      /*      GL_GREEN_SCALE */
      /*      GL_GREEN_BIAS */
      /*      GL_BLUE_SCALE */
      /*      GL_BLUE_BIAS */
      /*      GL_ALPHA_SCALE */
      /*      GL_ALPHA_BIAS */
      /*      GL_DEPTH_SCALE */
      /*      GL_DEPTH_BIAS */

      /* PixelType */
      self->addClassProperty( c_sdlgl, "BITMAP" ).setInteger( GL_BITMAP );
      /*      GL_BYTE */
      /*      GL_UNSIGNED_BYTE */
      /*      GL_SHORT */
      /*      GL_UNSIGNED_SHORT */
      /*      GL_INT */
      /*      GL_UNSIGNED_INT */
      /*      GL_FLOAT */

      /* PolygonMode */
      self->addClassProperty( c_sdlgl, "POINT" ).setInteger( GL_POINT );
      self->addClassProperty( c_sdlgl, "LINE" ).setInteger( GL_LINE );
      self->addClassProperty( c_sdlgl, "FILL" ).setInteger( GL_FILL );

      /* ReadBufferMode */
      /*      GL_FRONT_LEFT */
      /*      GL_FRONT_RIGHT */
      /*      GL_BACK_LEFT */
      /*      GL_BACK_RIGHT */
      /*      GL_FRONT */
      /*      GL_BACK */
      /*      GL_LEFT */
      /*      GL_RIGHT */
      /*      GL_AUX0 */
      /*      GL_AUX1 */
      /*      GL_AUX2 */
      /*      GL_AUX3 */

      /* RenderingMode */
      self->addClassProperty( c_sdlgl, "RENDER" ).setInteger( GL_RENDER );
      self->addClassProperty( c_sdlgl, "FEEDBACK" ).setInteger( GL_FEEDBACK );
      self->addClassProperty( c_sdlgl, "SELECT" ).setInteger( GL_SELECT );

      /* ShadingModel */
      self->addClassProperty( c_sdlgl, "FLAT" ).setInteger( GL_FLAT );
      self->addClassProperty( c_sdlgl, "SMOOTH" ).setInteger( GL_SMOOTH );


      /* StencilFunction */
      /*      GL_NEVER */
      /*      GL_LESS */
      /*      GL_EQUAL */
      /*      GL_LEQUAL */
      /*      GL_GREATER */
      /*      GL_NOTEQUAL */
      /*      GL_GEQUAL */
      /*      GL_ALWAYS */

      /* StencilOp */
      /*      GL_ZERO */
      self->addClassProperty( c_sdlgl, "KEEP" ).setInteger( GL_KEEP );
      self->addClassProperty( c_sdlgl, "REPLACE" ).setInteger( GL_REPLACE );
      self->addClassProperty( c_sdlgl, "INCR" ).setInteger( GL_INCR );
      self->addClassProperty( c_sdlgl, "DECR" ).setInteger( GL_DECR );
      /*      GL_INVERT */

      /* StringName */
      self->addClassProperty( c_sdlgl, "VENDOR" ).setInteger( GL_VENDOR );
      self->addClassProperty( c_sdlgl, "RENDERER" ).setInteger( GL_RENDERER );
      self->addClassProperty( c_sdlgl, "VERSION" ).setInteger( GL_VERSION );
      self->addClassProperty( c_sdlgl, "EXTENSIONS" ).setInteger( GL_EXTENSIONS );

      /* TextureCoordName */
      self->addClassProperty( c_sdlgl, "S" ).setInteger( GL_S );
      self->addClassProperty( c_sdlgl, "T" ).setInteger( GL_T );
      self->addClassProperty( c_sdlgl, "R" ).setInteger( GL_R );
      self->addClassProperty( c_sdlgl, "Q" ).setInteger( GL_Q );

      /* TexCoordPointerType */
      /*      GL_SHORT */
      /*      GL_INT */
      /*      GL_FLOAT */
      /*      GL_DOUBLE */

      /* TextureEnvMode */
      self->addClassProperty( c_sdlgl, "MODULATE" ).setInteger( GL_MODULATE );
      self->addClassProperty( c_sdlgl, "DECAL" ).setInteger( GL_DECAL );
      /*      GL_BLEND */
      /*      GL_REPLACE */

      /* TextureEnvParameter */
      self->addClassProperty( c_sdlgl, "TEXTURE_ENV_MODE" ).setInteger( GL_TEXTURE_ENV_MODE );
      self->addClassProperty( c_sdlgl, "TEXTURE_ENV_COLOR" ).setInteger( GL_TEXTURE_ENV_COLOR );

      /* TextureEnvTarget */
      self->addClassProperty( c_sdlgl, "TEXTURE_ENV" ).setInteger( GL_TEXTURE_ENV );

      /* TextureGenMode */
      self->addClassProperty( c_sdlgl, "EYE_LINEAR" ).setInteger( GL_EYE_LINEAR );
      self->addClassProperty( c_sdlgl, "OBJECT_LINEAR" ).setInteger( GL_OBJECT_LINEAR );
      self->addClassProperty( c_sdlgl, "SPHERE_MAP" ).setInteger( GL_SPHERE_MAP );

      /* TextureGenParameter */
      self->addClassProperty( c_sdlgl, "TEXTURE_GEN_MODE" ).setInteger( GL_TEXTURE_GEN_MODE );
      self->addClassProperty( c_sdlgl, "OBJECT_PLANE" ).setInteger( GL_OBJECT_PLANE );
      self->addClassProperty( c_sdlgl, "EYE_PLANE" ).setInteger( GL_EYE_PLANE );

      /* TextureMagFilter */
      self->addClassProperty( c_sdlgl, "NEAREST" ).setInteger( GL_NEAREST );
      self->addClassProperty( c_sdlgl, "LINEAR" ).setInteger( GL_LINEAR );

      /* TextureMinFilter */
      /*      GL_NEAREST */
      /*      GL_LINEAR */
      self->addClassProperty( c_sdlgl, "NEAREST_MIPMAP_NEAREST" ).setInteger( GL_NEAREST_MIPMAP_NEAREST );
      self->addClassProperty( c_sdlgl, "LINEAR_MIPMAP_NEAREST" ).setInteger( GL_LINEAR_MIPMAP_NEAREST );
      self->addClassProperty( c_sdlgl, "NEAREST_MIPMAP_LINEAR" ).setInteger( GL_NEAREST_MIPMAP_LINEAR );
      self->addClassProperty( c_sdlgl, "LINEAR_MIPMAP_LINEAR" ).setInteger( GL_LINEAR_MIPMAP_LINEAR );

      /* TextureParameterName */
      self->addClassProperty( c_sdlgl, "TEXTURE_MAG_FILTER" ).setInteger( GL_TEXTURE_MAG_FILTER );
      self->addClassProperty( c_sdlgl, "TEXTURE_MIN_FILTER" ).setInteger( GL_TEXTURE_MIN_FILTER );
      self->addClassProperty( c_sdlgl, "TEXTURE_WRAP_S" ).setInteger( GL_TEXTURE_WRAP_S );
      self->addClassProperty( c_sdlgl, "TEXTURE_WRAP_T" ).setInteger( GL_TEXTURE_WRAP_T );
      /*      GL_TEXTURE_BORDER_COLOR */
      /*      GL_TEXTURE_PRIORITY */

      /* TextureTarget */
      /*      GL_TEXTURE_1D */
      /*      GL_TEXTURE_2D */
      /*      GL_PROXY_TEXTURE_1D */
      /*      GL_PROXY_TEXTURE_2D */

      /* TextureWrapMode */
      self->addClassProperty( c_sdlgl, "CLAMP" ).setInteger( GL_CLAMP );
      self->addClassProperty( c_sdlgl, "REPEAT" ).setInteger( GL_REPEAT );

      /* VertexPointerType */
      /*      GL_SHORT */
      /*      GL_INT */
      /*      GL_FLOAT */
      /*      GL_DOUBLE */

      /* ClientAttribMask */
      self->addClassProperty( c_sdlgl, "CLIENT_PIXEL_STORE_BIT" ).setInteger( GL_CLIENT_PIXEL_STORE_BIT );
      self->addClassProperty( c_sdlgl, "CLIENT_VERTEX_ARRAY_BIT" ).setInteger( GL_CLIENT_VERTEX_ARRAY_BIT );
      self->addClassProperty( c_sdlgl, "CLIENT_ALL_ATTRIB_BITS" ).setInteger( GL_CLIENT_ALL_ATTRIB_BITS );

      /* polygon_offset */
      self->addClassProperty( c_sdlgl, "POLYGON_OFFSET_FACTOR" ).setInteger( GL_POLYGON_OFFSET_FACTOR );
      self->addClassProperty( c_sdlgl, "POLYGON_OFFSET_UNITS" ).setInteger( GL_POLYGON_OFFSET_UNITS );
      self->addClassProperty( c_sdlgl, "POLYGON_OFFSET_POINT" ).setInteger( GL_POLYGON_OFFSET_POINT );
      self->addClassProperty( c_sdlgl, "POLYGON_OFFSET_LINE" ).setInteger( GL_POLYGON_OFFSET_LINE );
      self->addClassProperty( c_sdlgl, "POLYGON_OFFSET_FILL" ).setInteger( GL_POLYGON_OFFSET_FILL );

      /* texture */
      self->addClassProperty( c_sdlgl, "ALPHA4" ).setInteger( GL_ALPHA4 );
      self->addClassProperty( c_sdlgl, "ALPHA8" ).setInteger( GL_ALPHA8 );
      self->addClassProperty( c_sdlgl, "ALPHA12" ).setInteger( GL_ALPHA12 );
      self->addClassProperty( c_sdlgl, "ALPHA16" ).setInteger( GL_ALPHA16 );
      self->addClassProperty( c_sdlgl, "LUMINANCE4" ).setInteger( GL_LUMINANCE4 );
      self->addClassProperty( c_sdlgl, "LUMINANCE8" ).setInteger( GL_LUMINANCE8 );
      self->addClassProperty( c_sdlgl, "LUMINANCE12" ).setInteger( GL_LUMINANCE12 );
      self->addClassProperty( c_sdlgl, "LUMINANCE16" ).setInteger( GL_LUMINANCE16 );
      self->addClassProperty( c_sdlgl, "LUMINANCE4_ALPHA4" ).setInteger( GL_LUMINANCE4_ALPHA4 );
      self->addClassProperty( c_sdlgl, "LUMINANCE6_ALPHA2" ).setInteger( GL_LUMINANCE6_ALPHA2 );
      self->addClassProperty( c_sdlgl, "LUMINANCE8_ALPHA8" ).setInteger( GL_LUMINANCE8_ALPHA8 );
      self->addClassProperty( c_sdlgl, "LUMINANCE12_ALPHA4" ).setInteger( GL_LUMINANCE12_ALPHA4 );
      self->addClassProperty( c_sdlgl, "LUMINANCE12_ALPHA12" ).setInteger( GL_LUMINANCE12_ALPHA12 );
      self->addClassProperty( c_sdlgl, "LUMINANCE16_ALPHA16" ).setInteger( GL_LUMINANCE16_ALPHA16 );
      self->addClassProperty( c_sdlgl, "INTENSITY" ).setInteger( GL_INTENSITY );
      self->addClassProperty( c_sdlgl, "INTENSITY4" ).setInteger( GL_INTENSITY4 );
      self->addClassProperty( c_sdlgl, "INTENSITY8" ).setInteger( GL_INTENSITY8 );
      self->addClassProperty( c_sdlgl, "INTENSITY12" ).setInteger( GL_INTENSITY12 );
      self->addClassProperty( c_sdlgl, "INTENSITY16" ).setInteger( GL_INTENSITY16 );
      self->addClassProperty( c_sdlgl, "R3_G3_B2" ).setInteger( GL_R3_G3_B2 );
      self->addClassProperty( c_sdlgl, "RGB4" ).setInteger( GL_RGB4 );
      self->addClassProperty( c_sdlgl, "RGB5" ).setInteger( GL_RGB5 );
      self->addClassProperty( c_sdlgl, "RGB8" ).setInteger( GL_RGB8 );
      self->addClassProperty( c_sdlgl, "RGB10" ).setInteger( GL_RGB10 );
      self->addClassProperty( c_sdlgl, "RGB12" ).setInteger( GL_RGB12 );
      self->addClassProperty( c_sdlgl, "RGB16" ).setInteger( GL_RGB16 );
      self->addClassProperty( c_sdlgl, "RGBA2" ).setInteger( GL_RGBA2 );
      self->addClassProperty( c_sdlgl, "RGBA4" ).setInteger( GL_RGBA4 );
      self->addClassProperty( c_sdlgl, "RGB5_A1" ).setInteger( GL_RGB5_A1 );
      self->addClassProperty( c_sdlgl, "RGBA8" ).setInteger( GL_RGBA8 );
      self->addClassProperty( c_sdlgl, "RGB10_A2" ).setInteger( GL_RGB10_A2 );
      self->addClassProperty( c_sdlgl, "RGBA12" ).setInteger( GL_RGBA12 );
      self->addClassProperty( c_sdlgl, "RGBA16" ).setInteger( GL_RGBA16 );
      self->addClassProperty( c_sdlgl, "TEXTURE_RED_SIZE" ).setInteger( GL_TEXTURE_RED_SIZE );
      self->addClassProperty( c_sdlgl, "TEXTURE_GREEN_SIZE" ).setInteger( GL_TEXTURE_GREEN_SIZE );
      self->addClassProperty( c_sdlgl, "TEXTURE_BLUE_SIZE" ).setInteger( GL_TEXTURE_BLUE_SIZE );
      self->addClassProperty( c_sdlgl, "TEXTURE_ALPHA_SIZE" ).setInteger( GL_TEXTURE_ALPHA_SIZE );
      self->addClassProperty( c_sdlgl, "TEXTURE_LUMINANCE_SIZE" ).setInteger( GL_TEXTURE_LUMINANCE_SIZE );
      self->addClassProperty( c_sdlgl, "TEXTURE_INTENSITY_SIZE" ).setInteger( GL_TEXTURE_INTENSITY_SIZE );
      self->addClassProperty( c_sdlgl, "PROXY_TEXTURE_1D" ).setInteger( GL_PROXY_TEXTURE_1D );
      self->addClassProperty( c_sdlgl, "PROXY_TEXTURE_2D" ).setInteger( GL_PROXY_TEXTURE_2D );

      /* texture_object */
      self->addClassProperty( c_sdlgl, "TEXTURE_PRIORITY" ).setInteger( GL_TEXTURE_PRIORITY );
      self->addClassProperty( c_sdlgl, "TEXTURE_RESIDENT" ).setInteger( GL_TEXTURE_RESIDENT );
      self->addClassProperty( c_sdlgl, "TEXTURE_BINDING_1D" ).setInteger( GL_TEXTURE_BINDING_1D );
      self->addClassProperty( c_sdlgl, "TEXTURE_BINDING_2D" ).setInteger( GL_TEXTURE_BINDING_2D );

      /* vertex_array */
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY" ).setInteger( GL_VERTEX_ARRAY );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY" ).setInteger( GL_NORMAL_ARRAY );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY" ).setInteger( GL_COLOR_ARRAY );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY" ).setInteger( GL_INDEX_ARRAY );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY" ).setInteger( GL_TEXTURE_COORD_ARRAY );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY" ).setInteger( GL_EDGE_FLAG_ARRAY );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_SIZE" ).setInteger( GL_VERTEX_ARRAY_SIZE );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_TYPE" ).setInteger( GL_VERTEX_ARRAY_TYPE );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_STRIDE" ).setInteger( GL_VERTEX_ARRAY_STRIDE );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_TYPE" ).setInteger( GL_NORMAL_ARRAY_TYPE );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_STRIDE" ).setInteger( GL_NORMAL_ARRAY_STRIDE );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_SIZE" ).setInteger( GL_COLOR_ARRAY_SIZE );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_TYPE" ).setInteger( GL_COLOR_ARRAY_TYPE );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_STRIDE" ).setInteger( GL_COLOR_ARRAY_STRIDE );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_TYPE" ).setInteger( GL_INDEX_ARRAY_TYPE );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_STRIDE" ).setInteger( GL_INDEX_ARRAY_STRIDE );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_SIZE" ).setInteger( GL_TEXTURE_COORD_ARRAY_SIZE );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_TYPE" ).setInteger( GL_TEXTURE_COORD_ARRAY_TYPE );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_STRIDE" ).setInteger( GL_TEXTURE_COORD_ARRAY_STRIDE );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY_STRIDE" ).setInteger( GL_EDGE_FLAG_ARRAY_STRIDE );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_POINTER" ).setInteger( GL_VERTEX_ARRAY_POINTER );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_POINTER" ).setInteger( GL_NORMAL_ARRAY_POINTER );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_POINTER" ).setInteger( GL_COLOR_ARRAY_POINTER );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_POINTER" ).setInteger( GL_INDEX_ARRAY_POINTER );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_POINTER" ).setInteger( GL_TEXTURE_COORD_ARRAY_POINTER );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY_POINTER" ).setInteger( GL_EDGE_FLAG_ARRAY_POINTER );
      self->addClassProperty( c_sdlgl, "V2F" ).setInteger( GL_V2F );
      self->addClassProperty( c_sdlgl, "V3F" ).setInteger( GL_V3F );
      self->addClassProperty( c_sdlgl, "C4UB_V2F" ).setInteger( GL_C4UB_V2F );
      self->addClassProperty( c_sdlgl, "C4UB_V3F" ).setInteger( GL_C4UB_V3F );
      self->addClassProperty( c_sdlgl, "C3F_V3F" ).setInteger( GL_C3F_V3F );
      self->addClassProperty( c_sdlgl, "N3F_V3F" ).setInteger( GL_N3F_V3F );
      self->addClassProperty( c_sdlgl, "C4F_N3F_V3F" ).setInteger( GL_C4F_N3F_V3F );
      self->addClassProperty( c_sdlgl, "T2F_V3F" ).setInteger( GL_T2F_V3F );
      self->addClassProperty( c_sdlgl, "T4F_V4F" ).setInteger( GL_T4F_V4F );
      self->addClassProperty( c_sdlgl, "T2F_C4UB_V3F" ).setInteger( GL_T2F_C4UB_V3F );
      self->addClassProperty( c_sdlgl, "T2F_C3F_V3F" ).setInteger( GL_T2F_C3F_V3F );
      self->addClassProperty( c_sdlgl, "T2F_N3F_V3F" ).setInteger( GL_T2F_N3F_V3F );
      self->addClassProperty( c_sdlgl, "T2F_C4F_N3F_V3F" ).setInteger( GL_T2F_C4F_N3F_V3F );
      self->addClassProperty( c_sdlgl, "T4F_C4F_N3F_V4F" ).setInteger( GL_T4F_C4F_N3F_V4F );

      /* Extensions */
      self->addClassProperty( c_sdlgl, "EXT_vertex_array" ).setInteger( GL_EXT_vertex_array );
      self->addClassProperty( c_sdlgl, "EXT_bgra" ).setInteger( GL_EXT_bgra );
      self->addClassProperty( c_sdlgl, "EXT_paletted_texture" ).setInteger( GL_EXT_paletted_texture );
      self->addClassProperty( c_sdlgl, "WIN_swap_hint" ).setInteger( GL_WIN_swap_hint );
      self->addClassProperty( c_sdlgl, "WIN_draw_range_elements" ).setInteger( GL_WIN_draw_range_elements );
      // GL_WIN_phong_shading
      // GL_WIN_specular_fog

      /* EXT_vertex_array */
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_EXT" ).setInteger( GL_VERTEX_ARRAY_EXT );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_EXT" ).setInteger( GL_NORMAL_ARRAY_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_EXT" ).setInteger( GL_COLOR_ARRAY_EXT );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_EXT" ).setInteger( GL_INDEX_ARRAY_EXT );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_EXT );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_EXT );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_SIZE_EXT" ).setInteger( GL_VERTEX_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_TYPE_EXT" ).setInteger( GL_VERTEX_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_STRIDE_EXT" ).setInteger( GL_VERTEX_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_COUNT_EXT" ).setInteger( GL_VERTEX_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_TYPE_EXT" ).setInteger( GL_NORMAL_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_STRIDE_EXT" ).setInteger( GL_NORMAL_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_COUNT_EXT" ).setInteger( GL_NORMAL_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_SIZE_EXT" ).setInteger( GL_COLOR_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_TYPE_EXT" ).setInteger( GL_COLOR_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_STRIDE_EXT" ).setInteger( GL_COLOR_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_COUNT_EXT" ).setInteger( GL_COLOR_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_TYPE_EXT" ).setInteger( GL_INDEX_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_STRIDE_EXT" ).setInteger( GL_INDEX_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_COUNT_EXT" ).setInteger( GL_INDEX_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_SIZE_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_TYPE_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_TYPE_EXT );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_STRIDE_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_COUNT_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY_STRIDE_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_STRIDE_EXT );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY_COUNT_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_COUNT_EXT );
      self->addClassProperty( c_sdlgl, "VERTEX_ARRAY_POINTER_EXT" ).setInteger( GL_VERTEX_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlgl, "NORMAL_ARRAY_POINTER_EXT" ).setInteger( GL_NORMAL_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_ARRAY_POINTER_EXT" ).setInteger( GL_COLOR_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlgl, "INDEX_ARRAY_POINTER_EXT" ).setInteger( GL_INDEX_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlgl, "TEXTURE_COORD_ARRAY_POINTER_EXT" ).setInteger( GL_TEXTURE_COORD_ARRAY_POINTER_EXT );
      self->addClassProperty( c_sdlgl, "EDGE_FLAG_ARRAY_POINTER_EXT" ).setInteger( GL_EDGE_FLAG_ARRAY_POINTER_EXT );
      #define GL_DOUBLE_EXT                     GL_DOUBLE

      /* EXT_bgra */
      self->addClassProperty( c_sdlgl, "BGR_EXT" ).setInteger( GL_BGR_EXT );
      self->addClassProperty( c_sdlgl, "BGRA_EXT" ).setInteger( GL_BGRA_EXT );

      /* EXT_paletted_texture */

      /* These must match the GL_COLOR_TABLE_*_SGI enumerants */
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_FORMAT_EXT" ).setInteger( GL_COLOR_TABLE_FORMAT_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_WIDTH_EXT" ).setInteger( GL_COLOR_TABLE_WIDTH_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_RED_SIZE_EXT" ).setInteger( GL_COLOR_TABLE_RED_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_GREEN_SIZE_EXT" ).setInteger( GL_COLOR_TABLE_GREEN_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_BLUE_SIZE_EXT" ).setInteger( GL_COLOR_TABLE_BLUE_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_ALPHA_SIZE_EXT" ).setInteger( GL_COLOR_TABLE_ALPHA_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_LUMINANCE_SIZE_EXT" ).setInteger( GL_COLOR_TABLE_LUMINANCE_SIZE_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_TABLE_INTENSITY_SIZE_EXT" ).setInteger( GL_COLOR_TABLE_INTENSITY_SIZE_EXT );

      self->addClassProperty( c_sdlgl, "COLOR_INDEX1_EXT" ).setInteger( GL_COLOR_INDEX1_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_INDEX2_EXT" ).setInteger( GL_COLOR_INDEX2_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_INDEX4_EXT" ).setInteger( GL_COLOR_INDEX4_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_INDEX8_EXT" ).setInteger( GL_COLOR_INDEX8_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_INDEX12_EXT" ).setInteger( GL_COLOR_INDEX12_EXT );
      self->addClassProperty( c_sdlgl, "COLOR_INDEX16_EXT" ).setInteger( GL_COLOR_INDEX16_EXT );

      /* WIN_draw_range_elements */
      self->addClassProperty( c_sdlgl, "MAX_ELEMENTS_VERTICES_WIN" ).setInteger( GL_MAX_ELEMENTS_VERTICES_WIN );
      self->addClassProperty( c_sdlgl, "MAX_ELEMENTS_INDICES_WIN" ).setInteger( GL_MAX_ELEMENTS_INDICES_WIN );

      /* WIN_phong_shading */
      self->addClassProperty( c_sdlgl, "PHONG_WIN" ).setInteger( GL_PHONG_WIN );
      self->addClassProperty( c_sdlgl, "PHONG_HINT_WIN" ).setInteger( GL_PHONG_HINT_WIN );

      /* WIN_specular_fog */
      self->addClassProperty( c_sdlgl, "FOG_SPECULAR_TEXTURE_WIN" ).setInteger( GL_FOG_SPECULAR_TEXTURE_WIN );
   }

   return self;
}
/* end of sdlttf.cpp */

