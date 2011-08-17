/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: fstream_posix.h

   Posix system specific FILE service support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 08 Jun 2011 12:54:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Posix system specific FILE service support.
*/

#ifndef _FALCON_FSTREAM_POSIX_H_
#define _FALCON_FSTREAM_POSIX_H_

#include <falcon/setup.h>

namespace Falcon {

class PosixFStreamData {
public:
   int fdFile;

   PosixFStreamData( int fd ):
      fdFile( fd )
   {}
};

}

#endif

/* end of fstream_posx.h */
