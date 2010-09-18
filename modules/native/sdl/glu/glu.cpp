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
#include "glu_ext.h"
#include "glu_mod.h"

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

   
   {
     
      self->addExtFunc( "Perspective", Falcon::Ext::openglu_Perspective );
      self->addExtFunc( "Build2DMipmaps", Falcon::Ext::openglu_Build2DMipmaps );
      self->addExtFunc( "ErrorString", Falcon::Ext::openglu_ErrorString );
      
      
   }
   {
      Falcon::Symbol *c_sdlglu = self->addClass( "GLU" );
      /****           Generic constants               ****/

      /* Version */
      self->addClassProperty( c_sdlglu, "VERSION_1_1" ).setInteger( GLU_VERSION_1_1 );
      self->addClassProperty( c_sdlglu, "VERSION_1_2" ).setInteger( GLU_VERSION_1_2 );

      /* Errors: (return value 0 = no error) */
      self->addClassProperty( c_sdlglu, "INVALID_ENUM" ).setInteger( GLU_INVALID_ENUM );
      self->addClassProperty( c_sdlglu, "INVALID_VALUE" ).setInteger( GLU_INVALID_VALUE );
      self->addClassProperty( c_sdlglu, "OUT_OF_MEMORY" ).setInteger( GLU_OUT_OF_MEMORY );
      self->addClassProperty( c_sdlglu, "INCOMPATIBLE_GL_VERSION" ).setInteger( GLU_INCOMPATIBLE_GL_VERSION );

      /* StringName */
      self->addClassProperty( c_sdlglu, "VERSION" ).setInteger( GLU_VERSION );
      self->addClassProperty( c_sdlglu, "EXTENSIONS" ).setInteger( GLU_EXTENSIONS );

      /* Boolean */
      self->addClassProperty( c_sdlglu, "TRUE" ).setInteger( GLU_TRUE );
      self->addClassProperty( c_sdlglu, "FALSE" ).setInteger( GLU_FALSE );


      /****           Quadric constants               ****/

      /* QuadricNormal */
      self->addClassProperty( c_sdlglu, "SMOOTH" ).setInteger( GLU_SMOOTH );
      self->addClassProperty( c_sdlglu, "FLAT" ).setInteger( GLU_FLAT );
      self->addClassProperty( c_sdlglu, "NONE" ).setInteger( GLU_NONE );

      /* QuadricDrawStyle */
      self->addClassProperty( c_sdlglu, "POINT" ).setInteger( GLU_POINT );
      self->addClassProperty( c_sdlglu, "LINE" ).setInteger( GLU_LINE );
      self->addClassProperty( c_sdlglu, "FILL" ).setInteger( GLU_FILL );
      self->addClassProperty( c_sdlglu, "SILHOUETTE" ).setInteger( GLU_SILHOUETTE );

      /* QuadricOrientation */
      self->addClassProperty( c_sdlglu, "OUTSIDE" ).setInteger( GLU_OUTSIDE );
      self->addClassProperty( c_sdlglu, "INSIDE" ).setInteger( GLU_INSIDE );

      /* Callback types: */
      /*      self->addClassProperty( c_sdlglu, "ERROR" ).setInteger( GLU_ERROR ); */


      /****           Tesselation constants           ****/

      self->addClassProperty( c_sdlglu, "TESS_MAX_COORD" ).setNumeric( GLU_TESS_MAX_COORD );

      /* TessProperty */
      self->addClassProperty( c_sdlglu, "TESS_WINDING_RULE" ).setInteger( GLU_TESS_WINDING_RULE );
      self->addClassProperty( c_sdlglu, "TESS_BOUNDARY_ONLY" ).setInteger( GLU_TESS_BOUNDARY_ONLY );
      self->addClassProperty( c_sdlglu, "TESS_TOLERANCE" ).setInteger( GLU_TESS_TOLERANCE );

      /* TessWinding */
      self->addClassProperty( c_sdlglu, "TESS_WINDING_ODD" ).setInteger( GLU_TESS_WINDING_ODD );
      self->addClassProperty( c_sdlglu, "TESS_WINDING_NONZERO" ).setInteger( GLU_TESS_WINDING_NONZERO );
      self->addClassProperty( c_sdlglu, "TESS_WINDING_POSITIVE" ).setInteger( GLU_TESS_WINDING_POSITIVE );
      self->addClassProperty( c_sdlglu, "TESS_WINDING_NEGATIVE" ).setInteger( GLU_TESS_WINDING_NEGATIVE );
      self->addClassProperty( c_sdlglu, "TESS_WINDING_ABS_GEQ_TWO" ).setInteger( GLU_TESS_WINDING_ABS_GEQ_TWO );

      /* TessCallback */
      self->addClassProperty( c_sdlglu, "TESS_BEGIN" ).setInteger( GLU_TESS_BEGIN );  
      self->addClassProperty( c_sdlglu, "TESS_VERTEX" ).setInteger( GLU_TESS_VERTEX );  
      self->addClassProperty( c_sdlglu, "TESS_END" ).setInteger( GLU_TESS_END );  
      self->addClassProperty( c_sdlglu, "TESS_ERROR" ).setInteger( GLU_TESS_ERROR );  
      self->addClassProperty( c_sdlglu, "TESS_EDGE_FLAG" ).setInteger( GLU_TESS_EDGE_FLAG );  
      self->addClassProperty( c_sdlglu, "TESS_COMBINE" ).setInteger( GLU_TESS_COMBINE );  
      self->addClassProperty( c_sdlglu, "TESS_BEGIN_DATA" ).setInteger( GLU_TESS_BEGIN_DATA );  
      self->addClassProperty( c_sdlglu, "TESS_VERTEX_DATA" ).setInteger( GLU_TESS_VERTEX_DATA );  
      self->addClassProperty( c_sdlglu, "TESS_END_DATA" ).setInteger( GLU_TESS_END_DATA );  
      self->addClassProperty( c_sdlglu, "TESS_ERROR_DATA" ).setInteger( GLU_TESS_ERROR_DATA );  
      self->addClassProperty( c_sdlglu, "TESS_EDGE_FLAG_DATA" ).setInteger( GLU_TESS_EDGE_FLAG_DATA );  
      self->addClassProperty( c_sdlglu, "TESS_COMBINE_DATA" ).setInteger( GLU_TESS_COMBINE_DATA );  

      /* TessError */
      self->addClassProperty( c_sdlglu, "TESS_ERROR1" ).setInteger( GLU_TESS_ERROR1 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR2" ).setInteger( GLU_TESS_ERROR2 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR3" ).setInteger( GLU_TESS_ERROR3 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR4" ).setInteger( GLU_TESS_ERROR4 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR5" ).setInteger( GLU_TESS_ERROR5 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR6" ).setInteger( GLU_TESS_ERROR6 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR7" ).setInteger( GLU_TESS_ERROR7 );
      self->addClassProperty( c_sdlglu, "TESS_ERROR8" ).setInteger( GLU_TESS_ERROR8 );

      self->addClassProperty( c_sdlglu, "TESS_MISSING_BEGIN_POLYGON" ).setInteger( GLU_TESS_MISSING_BEGIN_POLYGON );
      self->addClassProperty( c_sdlglu, "TESS_MISSING_BEGIN_CONTOUR" ).setInteger( GLU_TESS_MISSING_BEGIN_CONTOUR );
      self->addClassProperty( c_sdlglu, "TESS_MISSING_END_POLYGON" ).setInteger( GLU_TESS_MISSING_END_POLYGON );
      self->addClassProperty( c_sdlglu, "TESS_MISSING_END_CONTOUR" ).setInteger( GLU_TESS_MISSING_END_CONTOUR );
      self->addClassProperty( c_sdlglu, "TESS_COORD_TOO_LARGE" ).setInteger( GLU_TESS_COORD_TOO_LARGE );
      self->addClassProperty( c_sdlglu, "TESS_NEED_COMBINE_CALLBACK" ).setInteger( GLU_TESS_NEED_COMBINE_CALLBACK );

      /****           NURBS constants                 ****/

      /* NurbsProperty */
      self->addClassProperty( c_sdlglu, "AUTO_LOAD_MATRIX" ).setInteger( GLU_AUTO_LOAD_MATRIX );
      self->addClassProperty( c_sdlglu, "CULLING" ).setInteger( GLU_CULLING );
      self->addClassProperty( c_sdlglu, "SAMPLING_TOLERANCE" ).setInteger( GLU_SAMPLING_TOLERANCE );
      self->addClassProperty( c_sdlglu, "DISPLAY_MODE" ).setInteger( GLU_DISPLAY_MODE );
      self->addClassProperty( c_sdlglu, "PARAMETRIC_TOLERANCE" ).setInteger( GLU_PARAMETRIC_TOLERANCE );
      self->addClassProperty( c_sdlglu, "SAMPLING_METHOD" ).setInteger( GLU_SAMPLING_METHOD );
      self->addClassProperty( c_sdlglu, "U_STEP" ).setInteger( GLU_U_STEP );
      self->addClassProperty( c_sdlglu, "V_STEP" ).setInteger( GLU_V_STEP );

      /* NurbsSampling */
      self->addClassProperty( c_sdlglu, "PATH_LENGTH" ).setInteger( GLU_PATH_LENGTH );
      self->addClassProperty( c_sdlglu, "PARAMETRIC_ERROR" ).setInteger( GLU_PARAMETRIC_ERROR );
      self->addClassProperty( c_sdlglu, "DOMAIN_DISTANCE" ).setInteger( GLU_DOMAIN_DISTANCE );


      /* NurbsTrim */
      self->addClassProperty( c_sdlglu, "MAP1_TRIM_2" ).setInteger( GLU_MAP1_TRIM_2 );
      self->addClassProperty( c_sdlglu, "MAP1_TRIM_3" ).setInteger( GLU_MAP1_TRIM_3 );

      /* NurbsDisplay */
      /*      self->addClassProperty( c_sdlglu, "FILL" ).setInteger( GLU_FILL ); */
      self->addClassProperty( c_sdlglu, "OUTLINE_POLYGON" ).setInteger( GLU_OUTLINE_POLYGON );
      self->addClassProperty( c_sdlglu, "OUTLINE_PATCH" ).setInteger( GLU_OUTLINE_PATCH );

      /* NurbsCallback */
      /*      self->addClassProperty( c_sdlglu, "ERROR" ).setInteger( GLU_ERROR ); */

      /* NurbsErrors */
      self->addClassProperty( c_sdlglu, "NURBS_ERROR1" ).setInteger( GLU_NURBS_ERROR1 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR2" ).setInteger( GLU_NURBS_ERROR2 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR3" ).setInteger( GLU_NURBS_ERROR3 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR4" ).setInteger( GLU_NURBS_ERROR4 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR5" ).setInteger( GLU_NURBS_ERROR5 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR6" ).setInteger( GLU_NURBS_ERROR6 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR7" ).setInteger( GLU_NURBS_ERROR7 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR8" ).setInteger( GLU_NURBS_ERROR8 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR9" ).setInteger( GLU_NURBS_ERROR9 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR10" ).setInteger( GLU_NURBS_ERROR10 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR11" ).setInteger( GLU_NURBS_ERROR11 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR12" ).setInteger( GLU_NURBS_ERROR12 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR13" ).setInteger( GLU_NURBS_ERROR13 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR14" ).setInteger( GLU_NURBS_ERROR14 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR15" ).setInteger( GLU_NURBS_ERROR15 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR16" ).setInteger( GLU_NURBS_ERROR16 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR17" ).setInteger( GLU_NURBS_ERROR17 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR18" ).setInteger( GLU_NURBS_ERROR18 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR19" ).setInteger( GLU_NURBS_ERROR19 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR20" ).setInteger( GLU_NURBS_ERROR20 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR21" ).setInteger( GLU_NURBS_ERROR21 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR22" ).setInteger( GLU_NURBS_ERROR22 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR23" ).setInteger( GLU_NURBS_ERROR23 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR24" ).setInteger( GLU_NURBS_ERROR24 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR25" ).setInteger( GLU_NURBS_ERROR25 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR26" ).setInteger( GLU_NURBS_ERROR26 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR27" ).setInteger( GLU_NURBS_ERROR27 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR28" ).setInteger( GLU_NURBS_ERROR28 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR29" ).setInteger( GLU_NURBS_ERROR29 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR30" ).setInteger( GLU_NURBS_ERROR30 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR31" ).setInteger( GLU_NURBS_ERROR31 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR32" ).setInteger( GLU_NURBS_ERROR32 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR33" ).setInteger( GLU_NURBS_ERROR33 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR34" ).setInteger( GLU_NURBS_ERROR34 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR35" ).setInteger( GLU_NURBS_ERROR35 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR36" ).setInteger( GLU_NURBS_ERROR36 );
      self->addClassProperty( c_sdlglu, "NURBS_ERROR37" ).setInteger( GLU_NURBS_ERROR37 );

      /* Contours types -- obsolete! */
      self->addClassProperty( c_sdlglu, "CW" ).setInteger( GLU_CW );
      self->addClassProperty( c_sdlglu, "CCW" ).setInteger( GLU_CCW );
      self->addClassProperty( c_sdlglu, "INTERIOR" ).setInteger( GLU_INTERIOR );
      self->addClassProperty( c_sdlglu, "EXTERIOR" ).setInteger( GLU_EXTERIOR );
      self->addClassProperty( c_sdlglu, "UNKNOWN" ).setInteger( GLU_UNKNOWN );

      /* Names without "TESS_" prefix */
      self->addClassProperty( c_sdlglu, "BEGIN" ).setInteger( GLU_BEGIN );
      self->addClassProperty( c_sdlglu, "VERTEX" ).setInteger( GLU_VERTEX );
      self->addClassProperty( c_sdlglu, "END" ).setInteger( GLU_END );
      self->addClassProperty( c_sdlglu, "ERROR" ).setInteger( GLU_ERROR );
      self->addClassProperty( c_sdlglu, "EDGE_FLAG" ).setInteger( GLU_EDGE_FLAG );
   }

   return self;
}
/* end of sdlttf.cpp */

