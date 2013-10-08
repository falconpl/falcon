/*
   FALCON - The Falcon Programming Language.
   FILE: uploaded_ext.h

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Apr 2010 11:24:16 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef WOPI_UPLOADED_EXT_H
#define WOPI_UPLOADED_EXT_H

#include <falcon/setup.h>
#include <falcon/engine.h>

namespace Falcon {
namespace WOPI {


   class Uploaded {

   };

void InitUploadedClass( Module* m );

}
}

#endif

/* end of uploaded_ext.h */
