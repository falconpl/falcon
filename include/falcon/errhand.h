/*
   FALCON - The Falcon Programming Language.
   FILE: flc_errhand.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio set 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_ERRHAND_H
#define flc_ERRHAND_H

#include <falcon/string.h>
#include <falcon/genericlist.h>
#include <falcon/basealloc.h>
#include <falcon/error.h>

namespace Falcon {

/** Error handler interface.
   This interface allows embedding applications to provide their own
   error handlers. The error handler is invoked whenever any component
   of the Falcon system detects an error that require user (or embedder)
   attention.

*/

class FALCON_DYN_CLASS ErrorHandler: public BaseAlloc
{
public:

    /** Commits the error to the final media.
      Once the caller has properly setup the error description,
      this method will be called to deliver the error to the
      final representation media (i.e. a stream, a window,
      a network channel etc.).

      The handler is due to destroy the Error. Semipermanent data,
      as the traceback list, if present, or the raised() item in
      the error structure are to be considered valid only till
      the end of the function call, as the engine may dispose
      them after.

      I.e. if the program wants to save the error
      for later evaluation it should clear the raised() item,
      as it may get destroyed by GC if the VM returns in control.
   */
   virtual void handleError( Error *error ) = 0;
};

}

#endif

/* end of flc_errhand.h */
