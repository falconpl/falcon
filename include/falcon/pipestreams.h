/*
   FALCON - The Falcon Programming Language.
   FILE: pipestreams.h

   System streams with pipe traits.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_PIPESTREAM_H_
#define _FALCON_PIPESTREAM_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/filedata.h>
#include <falcon/streamtraits.h>

#include <falcon/fstream.h>
#include <falcon/stdstreamtraits.h>

namespace Falcon {

class ReadPipeStream: public ReadOnlyFStream
{
public:
   ReadPipeStream( Sys::FileData *fsdata ):
      ReadOnlyFStream( fsdata )
     {}

   virtual ~ReadPipeStream()
   {}

   virtual StreamTraits* traits() const
   {
      return Engine::streamTraits()->readPipeTraits();
   }
};

class WritePipeStream: public WriteOnlyFStream
{
public:
   WritePipeStream( Sys::FileData *fsdata ):
        WriteOnlyFStream( fsdata )
   {}

   virtual ~WritePipeStream()
   {}

   virtual StreamTraits* traits() const
   {
      return Engine::streamTraits()->writePipeTraits();
   }
};

}
#endif

/* end of pipestream.h */
