/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.cpp

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Tue, 12 Aug 2009 00:06:56 +1100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL OpenGL binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "glu_ext.h"
#include "glu_mod.h"
#include <sdl_service.h>  // for the reset
#include <SDL.h>
#include <SDL_opengl.h>


/*# @beginmodule sdlopengl */

namespace Falcon {

static SDLService *s_service = 0;

namespace Ext {
   

   FALCON_FUNC openglu_Perspective( ::Falcon::VMachine *vm )
   {
      Item *i_fovy = vm->param(0);
      Item *i_aspect = vm->param(1);
      Item *i_zNear = vm->param(2);
      Item *i_zFar = vm->param(3);

      if ( ( i_fovy == 0 || !i_fovy->isOrdinal() ) ||
           ( i_aspect == 0 || !i_aspect->isOrdinal() ) ||
           ( i_zNear == 0 || !i_zNear->isOrdinal() ) ||
           ( i_zFar == 0 || !i_zFar->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N,N" ) ) ;
         return;
      }
      
      GLdouble fovy = (GLdouble) i_fovy->forceNumeric();
      GLdouble aspect = (GLdouble) i_aspect->forceNumeric();
      GLdouble zNear = (GLdouble) i_zNear->forceNumeric();
      GLdouble zFar = (GLdouble) i_zFar->forceNumeric();

      ::gluPerspective(fovy, aspect, zNear, zFar);
   }

   FALCON_FUNC openglu_Build2DMipmaps( VMachine *vm )
   {
      Item *i_target = vm->param(0);
      Item *i_components = vm->param(1);
      Item *i_width = vm->param(2);
      Item *i_height = vm->param(3);
      Item *i_format = vm->param(4);
      Item *i_type = vm->param(5);
      Item *i_pixels = vm->param(6);
      if ( ( i_target == 0 || !i_target->isOrdinal() ) ||
           ( i_components == 0 || !i_components->isOrdinal() ) ||
           ( i_width == 0 || !i_width->isOrdinal() ) ||
           ( i_height == 0 || !i_height->isOrdinal() ) ||
           ( i_format == 0 || !i_format->isOrdinal() ) ||
           ( i_type == 0 || !i_type->isOrdinal() ) ||
           ( i_pixels == 0 || !i_pixels->isMemBuf() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N,N,N,N,M" ) ) ;
         return;
      }
      
      GLenum  target = (GLenum ) i_target->forceInteger();
      GLint  components = (GLint ) i_components->forceInteger();
      GLsizei width = (GLsizei) i_width->forceInteger();
      GLsizei height = (GLsizei) i_height->forceInteger();
      GLenum  format = (GLenum ) i_format->forceInteger();
      GLenum  type = (GLenum ) i_type->forceInteger();
      const GLvoid *pixels = (const GLvoid *) i_pixels->asMemBuf()->data();
      ::gluBuild2DMipmaps(target, components, width, height, format, type, pixels);
   }

   FALCON_FUNC openglu_ErrorString( VMachine *vm )
   {
      Item *i_errCode = vm->param(0);
      if ( (i_errCode == 0 || !i_errCode->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }
      GLenum errCode = (GLenum) i_errCode->forceInteger();
      const GLubyte *error_string = ::gluErrorString(errCode);
      String *s_error_string = new String((char *)error_string);
      vm->retval(s_error_string);

   }
   
}
}

/* end of TTF_ext.cpp */
