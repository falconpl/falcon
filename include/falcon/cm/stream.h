/*
   FALCON - The Falcon Programming Language.
   FILE: stream.h

   Falcon core module -- Interface to Stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_STREAM_H
#define FALCON_CORE_STREAM_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/types.h>

#include <falcon/usercarrier.h>
#include <falcon/path.h>

namespace Falcon {

class Stream;
class StreamBuffer;
class Transcoder;

namespace Ext {


/** We keep th path, the auth data and the query. */
class FALCON_DYN_CLASS StreamCarrier
{
public:
   uint32 m_gcMark;
   
   Stream* m_stream;
   StreamBuffer* m_sbuf;
   Stream* m_underlying;
      
   StreamCarrier( Stream* stream );
   virtual ~StreamCarrier();
   
   void setBuffering( uint32 size );
   
   virtual void onFlushingOperation();
};


/*# @class Stream
   
 */
class ClassStream: public ClassUser
{
public:
   
   ClassStream();
   virtual ~ClassStream();

   //=============================================================
   // Using a different carrier.
   
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void* createInstance() const;
   
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   
   //=============================================================
   //
   
private:   
   
   //====================================================
   // Properties.
   //
   
   FALCON_DECLARE_PROPERTY( error )
   FALCON_DECLARE_PROPERTY( moved )
   FALCON_DECLARE_PROPERTY( position )
   FALCON_DECLARE_PROPERTY( status )
   FALCON_DECLARE_PROPERTY( eof )
   FALCON_DECLARE_PROPERTY( interrupted )
   FALCON_DECLARE_PROPERTY( bad )
   FALCON_DECLARE_PROPERTY( good )
   FALCON_DECLARE_PROPERTY( isopen )
   FALCON_DECLARE_PROPERTY( buffer )
   
   FALCON_DECLARE_METHOD( write, "data:S|M, count:[N], start:[N]" );
   FALCON_DECLARE_METHOD( read, "data:S|M, count:[N], start:[N]" );
   FALCON_DECLARE_METHOD( grab, "count:N" );
   FALCON_DECLARE_METHOD( close, "" );
   FALCON_DECLARE_METHOD( seekBeg, "position:N" );
   FALCON_DECLARE_METHOD( seekCur, "position:N" );
   FALCON_DECLARE_METHOD( seekEnd, "position:N" );
   FALCON_DECLARE_METHOD( seek, "position:N,whence:N" );
   FALCON_DECLARE_METHOD( tell, "" );
   FALCON_DECLARE_METHOD( flush, "" );
   FALCON_DECLARE_METHOD( trunc, "position:[N]" );
   FALCON_DECLARE_METHOD( ravail, "msecs:[N]" );
   FALCON_DECLARE_METHOD( wavail, "msecs:[N]" );
};

}
}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of stream.h */
