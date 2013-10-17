/*
   FALCON - The Falcon Programming Language.
   FILE: classuploaded.cpp

   Web Oriented Programming Interface

   Wrapper for uploaded data files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Apr 2010 11:24:16 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/classuploaded.h>
#include <falcon/wopi/uploaded.h>

#include <falcon/collector.h>
#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>
#include <falcon/function.h>
#include <falcon/error.h>

/*#
   @beginmodule wopi
*/


namespace Falcon {
namespace WOPI {

//============================================
// Uploaded class - class managing uploads.
//============================================

namespace {

/*#
   @class Uploaded
   @brief Entity storing uploaded file data.

   Forms containing uploadable files are returned in the @a Request.posts
   dictionary. In those forms, the entries corresponding to uploaded
   files are stored as instances of this class.

   The class has support to access the temporary storage where the
   uploaded file has been placed, to read it in memory or to
   access the memory where the system has stored it.

   For more informations read the @a upload_control entry.

   @prop size The size of the uploaded file, in bytes.
   @prop mimeType The declared MIME type of the uploaded data. This is
         Determined by the remote browser uploading the file, so it cannot
         be trusted.
   @prop filename The name of the original file before upload. Can be used
         as an hint of the name it should be given locally, or for extra
         confirmation of the MIME type.
   @prop storage Complete path to an accessible file in the local filesystem.
         This is where the server has temporarily stored the file. It may
         be destroyed as soon as the script is closed. If the file was imported
         directly in memory, this field will be nil.
   @prop data Complete content of the file, as a byte-sized MemBuf. If the
         server was setup to store the file in memory as it is received,
         this field will be with a buffer string containing all the
         data in the file. If this option is not enabled, the property will
         be nil.
   @prop error A textual description of errors that may have occurred during
         the storage of the uploaded file, before the Falcon process
         is able to take care of it.
*/

static void get_size( const Class*, const String&, void *instance, Item& value )
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   value = (int64) upld->filesize();
}

static void get_mimeType( const Class*, const String&, void *instance, Item& value )
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   String* sval = new String(upld->mimeType());
   value = FALCON_GC_HANDLE( sval );
}

static void get_filename( const Class*, const String&, void *instance, Item& value )
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   String* sval = new String(upld->filename());
   value = FALCON_GC_HANDLE( sval );
}

static void get_storage( const Class*, const String&, void *instance, Item& value )
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   String* sval = new String(upld->storage());
   value = FALCON_GC_HANDLE( sval );
}

static void get_data( const Class*, const String&, void *instance, Item& value )
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   if( upld->data() == 0 ) {
      value.setNil();
   }
   else {
      value.setUser( upld->data()->handler(), upld->data() );
   }
}

static void get_error( const Class*, const String&, void *instance, Item& value )
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   String* sval = new String(upld->error());
   value = FALCON_GC_HANDLE( sval );
}

/*#
   @method read Uploaded
   @brief Reads an uploaded file from the temporary storage into memory.
   @return The contents of the uploaded file.
   @raise IoError on read error.
   @raise TypeError if the storage property is not a valid filename.
   
   If the uploaded file corresponding to this entry was stored in a temporary
   local file (in the @b storage property), this method reads it
   in a wide-enough MemBuf and stores it in the @b data property.

   It is possible to use this method to make sure that the whole file is
   in the @b data property after a size check.

   @note The server may prevent this operation to be completed if the
   file is too large. Where in doubt, prefer @b Uploaded.open, which has
   the same semantic but that is more flexible and resource-aware.
*/

FALCON_DECLARE_FUNCTION(read, "" )
FALCON_DEFINE_FUNCTION_P1(read)
{
   Uploaded* upld = static_cast<Uploaded*>(ctx->self().asInst());
   upld->read();
   ctx->returnFrame( Item(upld->data()->handler(), upld->data() ) );
}

/*#
   @method store Uploaded
   @brief Stores the uploaded data into a file.
   @param target The location where to store the file, or an open stream.
   @raise IoError on read or write error.
   @raise TypeError if the storage property is not a valid filename.
   
   If @b data is filled, this method saves
   its contents into the file indicated by the @b path parameter;
   if it was stored to a temporary file, a system file move
   is tried, if it fails, a file copy is tried, and the origin
   file is removed after the copy is succesful. 

   On failure, a relevant IoError is raised, but the operation
   doesn't fail in case the original file cannot be deleted.

   @note This method can be also used to move or copy an arbitrary
   file by storing a path directly in the @b storage
   property.
*/

FALCON_DECLARE_FUNCTION(store, "target:S|Stream" )
FALCON_DEFINE_FUNCTION_P1(store)
{
   static Class* streamClass = Engine::instance()->stdHandlers()->streamClass();
   Falcon::Item *i_path = ctx->param(0);
   
   if ( i_path == 0
       || ( ! i_path->isString() && ! i_path->isInstanceOf( streamClass ) )
   )
   {
      throw paramError();
   }
   
   Uploaded* upld = static_cast<Uploaded*>(ctx->self().asInst());
   if( i_path->isString() )
   {
      upld->store( *i_path->asString() );
   }
   else {
      Stream* stream = static_cast<Stream*>(i_path->asInst());
      upld->store(stream);
   }
   
   ctx->returnFrame();
}

/*#
   @method open Uploaded
   @brief Opens a read-only Falcon Stream pointing to the uploaed file.
   @return A Falcon stream.
   @raise IoError on open or write error.
   @raise TypeError if the storage property is not a valid filename.
   
   If @b data is filled, this method creates a memory
   read-only StringStream accessing the data as a file. If it was
   stored in a temporary file named as reported by the @b storage
   property, that file is open in read-only/shared mode.

   This method allows to obtain a valid readable stream no matter if the
   uploaded file was cached in memory or temporarily stored to disk.
*/
FALCON_DECLARE_FUNCTION(open, "" )
FALCON_DEFINE_FUNCTION_P1(open)
{
   Uploaded* upld = static_cast<Uploaded*>(ctx->self().asInst());
   Stream* stream = upld->open();
   ctx->returnFrame( Item(stream->handler(), stream) );
}

}

ClassUploaded::ClassUploaded():
         Class("Uploaded")
{
   addProperty( "size", &get_size );
   addProperty( "filename", &get_filename );
   addProperty( "mimeType", &get_mimeType );
   addProperty( "storage", &get_storage );
   addProperty( "data", &get_data );
   addProperty( "error", &get_error );

   addMethod( new Function_open );
   addMethod( new Function_read );
   addMethod( new Function_store );
}

ClassUploaded::~ClassUploaded()
{
}

void ClassUploaded::dispose( void* instance ) const
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   delete upld;
}

void* ClassUploaded::clone( void* ) const
{
   // uncloneable class
   return 0;
}

void* ClassUploaded::createInstance() const
{
   // virtual class
   return 0;
}

void ClassUploaded::gcMarkInstance( void* instance, uint32 mark ) const
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   upld->gcMark(mark);
}

bool ClassUploaded::gcCheckInstance( void* instance, uint32 mark ) const
{
   Uploaded* upld = static_cast<Uploaded*>(instance);
   return upld->currentMark() >= mark;
}

}
}

/* end of classuploaded.cpp */
