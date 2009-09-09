/*
   FALCON - The Falcon Programming Language
   FILE: vm_stdstreams_win.cpp

   Windows specific standard streams factories.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Windows specifi standard streams factories.
   Actually, the same as unix, but using also the CRLF transcoder.
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
         coder = new TranscoderByte(0);
      }
   }
   else
      coder = new TranscoderByte(0);

   coder->setUnderlying( stdin, true );
   /*Falcon::Transcoder *wincoder = new TranscoderEOL(0);
   wincoder->setUnderlying( coder, true );*/
   return coder;
}



Stream *stdOutputStream()
{
   Stream *stdout = new StreamBuffer( new StdOutStream );
   String enc;

   Falcon::Transcoder *coder;

   if ( Falcon::GetSystemEncoding( enc ) )
   {
      coder =  Falcon::TranscoderFactory( enc );
      if ( coder == 0 ) {
         coder = new TranscoderByte(0);
      }
   }
   else
      coder = new TranscoderByte(0);

   coder->setUnderlying( stdout, true );
   Falcon::Transcoder *wincoder = new TranscoderEOL(0);
   wincoder->setUnderlying( coder, true );
   return wincoder;

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
         coder = new TranscoderByte(0);
      }
   }
   else
      coder = new TranscoderByte(0);

   coder->setUnderlying( stderr, true );
   Falcon::Transcoder *wincoder = new TranscoderEOL(0);
   wincoder->setUnderlying( coder, true );
   return wincoder;
}

Stream *DefaultTextTranscoder( Stream *underlying, bool own )
{
   String encoding;
   if ( ! GetSystemEncoding( encoding ) )
   {
      Falcon::Transcoder *wincoder = new TranscoderEOL(0);
      wincoder->setUnderlying( underlying, own );
      return wincoder;
   }

   Transcoder *encap = TranscoderFactory( encoding );
   encap->setUnderlying( underlying, own );

   // the wincoder owns the underlying for sure.
   Falcon::Transcoder *wincoder = new TranscoderEOL(0);
   wincoder->setUnderlying( encap, true );

   return wincoder;
}

Stream *AddSystemEOL( Stream *underlying, bool own )
{
   return new TranscoderEOL( underlying, own );
}

}


/* end of stdstreams_win.cpp */
