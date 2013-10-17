/*
   FALCON - The Falcon Programming Language.
   FILE: uploaded.h

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 16 Oct 2013 15:41:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_UPLOADED_H_
#define _FALCON_WOPI_UPLOADED_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon {
namespace WOPI {

/** Class representing an uploaded file.
 *
 */
class Uploaded
{
public:
   Uploaded();
   Uploaded( const String& fname, const String& mime, int64 size );
   Uploaded( const String& fname, const String& mime, int64 size, const String& storage );
   ~Uploaded();

   /**
    * Ensures that the whole uploaded file is transferred into the data() member.
    */
   void read();

   /**
    * Opens a read-only data stream pointing to the uploaded file data.
    */
   Stream* open();

   /**
    * Moves the temporary memory or disk storage to a permanent location.
    */
   void store( const String& target );
   void store( Stream* target );

   void gcMark( uint32 m ) { m_gcMark = m; }
   uint32 currentMark() const { return m_gcMark; }

   int64 filesize() const { return m_size; }
   void filesize( int64 value ) { m_size = value; }

   const String& filename() const { return m_filename; }
   void filename( const String& value ) { m_filename = value; }

   const String& mimeType() const { return m_mimeType; }
   void mimeType( const String& value ) { m_mimeType = value; }

   const String& storage() const { return m_storage; }
   void storage( const String& value ) { m_storage = value; }

   String* data() const { return m_data; }
   void data( String* value );

   const String& error() const { return m_error; }
   void error( const String& value ) { m_error = value; }

private:
   int64 m_size;
   String m_filename;
   String m_mimeType;
   String m_storage;
   String m_error;

   GCLock* m_dataLock;
   String* m_data;

   uint32 m_gcMark;
};

}
}

#endif

/* end of uploaded.h */

