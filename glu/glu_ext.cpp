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
   
}
}

/* end of TTF_ext.cpp */
