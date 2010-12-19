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

#include "glext_ext.h"
#include "glext_mod.h"
#include <sdl_service.h>  // for the reset
#include <SDL.h>
#include <SDL_opengl.h>


/*--# @beginmodule sdl.opengl */

namespace Falcon {

static SDLService *s_service = 0;

namespace Ext {
   
   

}
}

/* end of TTF_ext.cpp */
