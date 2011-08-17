/*
   FALCON - The Falcon Programming Language
   FILE: stdstreams_unix.cpp

   Unix specific standard streams factories.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Unix specific standard streams factories.
*/

#include <falcon/fstream.h>
#include <falcon/transcoding.h>
#include <falcon/stdstreams.h>
#include <falcon/streambuffer.h>

namespace Falcon {

Stream *stdInputStream()
{
   Stream *stdin = new StreamBuffer( new StdInStream );
   String enc;

   Falcon::Transcoder *coder;

   if ( Falcon::GetSystemEncoding( enc ) )
   {
      coder =  Falcon::TranscoderFactory( enc );
      if ( coder == 0 ) {
         return stdin;
      }
   }
   else
      return stdin;

   coder->setUnderlying( stdin, true );
   return coder;
}

Stream *stdOutputStream()
{
   Stream *stdout = new StreamBuffer( new StdOutStream );
   //Stream *stdout = new StdOutStream;
   String enc;

   Falcon::Transcoder *coder;

   if ( Falcon::GetSystemEncoding( enc ) )
   {
      coder =  Falcon::TranscoderFactory( enc );
      if ( coder == 0 ) {
         return stdout;
      }
   }
   else
      return stdout;

   coder->setUnderlying( stdout, true );
   return coder;
}

Stream *stdErrorStream()
{
   Stream *stderr = new StreamBuffer( new StdErrStream );
   String enc;

   Falcon::Transcoder *coder;

   if ( Falcon::GetSystemEncoding( enc ) )
   {
      coder =  Falcon::TranscoderFactory( enc );
      if ( coder == 0 ) {
         return stderr;
      }
   }
   else
      return stderr;

   coder->setUnderlying( stderr, true );
   return coder;
}

Stream *DefaultTextTranscoder( Stream *underlying, bool own )
{
   String encoding;
   if ( ! GetSystemEncoding( encoding ) )
      return underlying;

   Transcoder *encap = TranscoderFactory( encoding, underlying, own );
   // in unix there's no difference between text and binary, so there's no other transcoder to add.
   return encap;
}


Stream *AddSystemEOL( Stream *underlying, bool )
{
   return underlying;
}

}


/* end of stdstreams_unix.cpp */
