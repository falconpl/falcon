/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_mod.cpp

   The SDL True Type binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL True Type binding support module - module specific extensions.
*/

#include <falcon/vm.h>
#include <falcon/membuf.h>
#include "sdlimage_mod.h"

#include <SDL_image.h>


/*# @beginmodule sdlttf */

namespace Falcon {
namespace Ext {

//=======================================
// Quit carrier
//
/*TTFQuitCarrier::~TTFQuitCarrier()
{
   TTF_Quit();
}

TTFFontCarrier::~TTFFontCarrier()
{
   TTF_CloseFont( m_font );
}*/

}
}

/* end of sdlttf_mod.cpp */
