/*
   FALCON - The Falcon Programming Language.
   FILE: sdlopengl_mod.h

   The SDL OpenGL binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Tue, 12 Aug 2009 00:06:56 +1100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL True Type binding support module - module specific extensions.
*/

#ifndef FALCON_SDLOPENGL_MOD
#define FALCON_SDLOPENGL_MOD

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>

#define FALCON_OPENGL_ERROR_BASE 2130

namespace Falcon{
namespace Ext{

/** Automatic quit system. */
class GLUQuitCarrier: public FalconData
{
public:
   GLUQuitCarrier() {}
   virtual ~GLUQuitCarrier();

   virtual void gcMark( uint32 ) {}
   virtual FalconData* clone() const { return 0; }
};

}
}
#endif

/* end of sdlopengl_mod.h */
