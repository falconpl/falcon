/*
   FALCON - The Falcon Programming Language.
   FILE: uploaded_ext.cpp

   Web Oriented Programming Interface

   Wrapper for uploaded data files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Apr 2010 11:24:16 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/uploaded_ext.h>

namespace Falcon {
namespace WOPI {

//============================================
// Uploaded class - class managing uploads.
//============================================

/*#
   @class Uploaded
   @brief Entity storing uploaded file data.

   Forms containing uploadable files are returned in the @a Request.posts
   dictionary. In those forms, the entries coresponding to uploaded
   files are stored as instances of this class.

   The class has support to access the temporary storage where the
   uploaded file has been placed, to read it in memory or to
   access the memory where the system has stored it.

   For more informations read the @a upload_control entry.

   @prop size The size of the uploaded file, in bytes.
   @prop mimeType The declared mime type of the uploaded data. This is
         Determined by the remote browser uploading the file, so it cannot
         be trusted.
   @prop filename The name of the original file before upload. Can be used
         as an hint of the name it should be given locally, or for extra
         confirmation of the mime type.
   @prop storage Complete path to an accessible file in the local filesystem.
         This is where the server has temporarily stored the file. It may
         be destroeyd as soon as the script is closed. If the file was impored
         directly in memory, this field will be nil.
   @prop data Complete content of the file, as a byte-sized MemBuf. If the
         server was setup to store the file in memory as it is received,
         this field will be valorized with a MemBuf containing all the
         data in the file. If this option is not enabled, the property will
         be nil.
*/

/*#
   @method read Uploaded
   @brief Reads an uploaded file from the temporary storage into memory.
   @return True if the file was actually read, false if it was already stored in memory.
   @raise IoError on read error.
   @raise TypeError if the storage property is not a valid filename.
   
   If the uploaded file coresponding to this entry was stored in a temporary
   local file (in the @b storage property), this method reads it
   in a wide-enough MemBuf and stores it in the @b data property.

   It is possible to use this method to make sure that the whole file is
   in the @b data property after a size check.

   @note The server may prevent this operation to be completed if the
   file is too large. Where in doubt, prefer @b Uploaded.open, which has
   the same semantic but that is more flexible and resource-aware.
*/

FALCON_FUNC Uploaded_read( Falcon::VMachine *vm )
{
   Falcon::CoreObject *self = vm->self().asObject();
   Falcon::Item i_data;

   if ( self->getProperty( "data", i_data ) &&
      ( i_data.isMemBuf() || i_data.isString() )
   )
   {
      vm->regA().setBoolean( false );
      return;
   }

   Falcon::Item i_storage;
   if ( ! self->getProperty( "storage", i_storage ) || ! i_storage.isString() )
   {
      // invalid storage?
      throw new Falcon::TypeError( Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ ).
         extra( ".storage" ) );
   }

   Falcon::FileStream fs;
   if ( ! fs.open(  *i_storage.asString(),
                     Falcon::BaseFileStream::e_omReadOnly,
                     Falcon::BaseFileStream::e_smShareRead ) )
   {
      throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
         sysError( (uint32) fs.lastError() ) );
   }

   Falcon::int64 filesize = fs.seekEnd(0);
   Falcon::MemBuf *mb = new Falcon::MemBuf_1( (uint32) filesize );

   fs.seekBegin( 0 );
   Falcon::int64 readIn = 0;
   while ( readIn < filesize )
   {
      Falcon::int32 len = fs.read( mb->data() + readIn, (int32)( filesize - readIn ) );
      if ( len < 0 )
      {
         throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
            sysError( (uint32) fs.lastError() ) );
         // anyhow, try to close to avoid system leaks.
         fs.close();
      }
      readIn += len;
   }

   self->setProperty( "data", mb );
   
   if ( ! fs.close() )
   {
      throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
            sysError( (uint32) fs.lastError() ) );
   }

   vm->regA().setBoolean( true );
}

/*#
   @method store Uploaded
   @brief Stores the uploaded data into a file.
   @param path The location where to store the file, or an open stream.
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

FALCON_FUNC Uploaded_store( Falcon::VMachine *vm )
{
   Falcon::Item *ip_path = vm->param(0);
   
   if ( ip_path == 0
       || ( ! ip_path->isString() && ! ip_path->isOfClass( "Stream" ) )
   )
   {
      throw
            new Falcon::ParamError( Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
                .extra( "S|Stream" ) );
   }
   
   Falcon::CoreObject *self = vm->self().asObject();

   //===============================================
   // Do we store from data?
   //
   Falcon::Item i_data;
   if ( self->getProperty( "data", i_data ) &&
      ( i_data.isMemBuf() || i_data.isString() )
   )
   {
      Falcon::FileStream* tgFile = 0;
      Falcon::Stream *tgStream;

      // try to save to ip_path;
      if ( ip_path->isString() )
      {
         // try to open the file.
         tgFile = new Falcon::FileStream;
         if ( ! tgFile->create( *ip_path->asString(),
               Falcon::BaseFileStream::e_aUserRead | Falcon::BaseFileStream::e_aUserWrite ) )
         {
            Falcon::int64 le = tgFile->lastError();
            delete tgFile;
            throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
               sysError( (uint32) le ) );
         }

         tgStream = tgFile;
      }
      else
         tgStream = static_cast<Falcon::Stream*>( ip_path->asObject()->getUserData() );
      
      Falcon::int64 written = 0;
      Falcon::uint32 len = i_data.isString() ? i_data.asString()->size()
                                                : i_data.asMemBuf()->size();
      Falcon::byte* data = i_data.isString() ? i_data.asString()->getRawStorage()
                                                : i_data.asMemBuf()->data();
      
      
      int wrt = 0;
      while ( written < len )
      {
         wrt = tgStream->write( data + written, (uint32)( len - written ) );
          if ( wrt < 0 )
         {
            Falcon::int64 le = tgStream->lastError();
            delete tgFile;
            throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
               sysError( (uint32) le ) );
         }
         written += wrt;
      }

      if ( ! tgStream->close() )
      {
         Falcon::int64 le = tgStream->lastError();
         delete tgFile;  // ok also if 0
         throw new Falcon::IoError(
            Falcon::ErrorParam( Falcon::e_io_error, __LINE__ )
            .sysError(  (uint32)le ) );
      }

      // success
      return;
   }


   // =============================================
   // Do we store from storage?

   Falcon::Item i_storage;
   if ( ! self->getProperty( "storage", i_storage ) || ! i_storage.isString() )
   {
      // invalid storage?
      throw new Falcon::TypeError( Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
            .extra( ".storage" ) );
   }

   const Falcon::String &fname = *i_storage.asString();

   // First, try to rename the file.
   if ( ip_path->isString() )
   {
      Falcon::int32 status;
      // fname -> dest
      if ( Falcon::Sys::fal_move( fname, *ip_path->asString(), status ) )
         return; // great, we did it
   }

   Falcon::FileStream infile;
   if ( ! infile.open( fname, Falcon::BaseFileStream::e_omReadOnly ) || ! infile.good() )
   {
      throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
               sysError( (uint32) infile.lastError() ) );
   }
   
   // no way? -- try to copy it.
   Falcon::FileStream* tgFile = 0;
   Falcon::Stream *tgStream;

   // try to save to ip_path;
   if ( ip_path->isString() )
   {
      // try to open the file.
      tgFile = new Falcon::FileStream();
      if ( ! tgFile->create( *ip_path->asString(),
               Falcon::BaseFileStream::e_aUserRead | Falcon::BaseFileStream::e_aUserWrite ) )
      {
         Falcon::int64 le = tgFile->lastError();
         delete tgFile;
         throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
            sysError( (uint32) le ) );
      }

      tgStream = tgFile;
   }
   else
      tgStream = static_cast<Falcon::Stream*>( ip_path->asObject()->getUserData() );

   Falcon::byte buffer[2048];
   int wrt = 0;
   while ( ! infile.eof() )
   {
      wrt = infile.read( buffer, 2048 );
       if ( wrt < 0 )
      {
         delete tgFile;
         throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
            sysError( (uint32) infile.lastError() ) );
      }
      
      wrt = tgStream->write( buffer, wrt );
         
      if ( wrt < 0 )
      {
         Falcon::int64 le = tgStream->lastError();
         delete tgFile;
         throw new Falcon::IoError( Falcon::ErrorParam( Falcon::e_io_error, __LINE__ ).
            sysError( (uint32) le ) );
      }
   }

   // silently try to unlink the source file
   infile.close();
   Falcon::int32 status;
   Falcon::Sys::fal_unlink( fname, status );

   // do not raise again if in error
   if ( ! tgStream->close() )
   {
      Falcon::int64 le = tgStream->lastError();
      delete tgFile;
      throw new Falcon::IoError(
         Falcon::ErrorParam( Falcon::e_io_error, __LINE__ )
         .sysError( (uint32) le ) );
   }

   delete tgFile;
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
FALCON_FUNC Uploaded_open( Falcon::VMachine *vm )
{
   Falcon::CoreObject *self = vm->self().asObject();
   Falcon::Item i_data;

   // try to read the data.
   Falcon::Stream *ret = 0;
   
   if ( self->getProperty( "data", i_data ) )
   {
      
      if ( i_data.isMemBuf() )
      {
         ret = new Falcon::StringStream();
         Falcon::MemBuf *mb = i_data.asMemBuf();
         ret->write( mb->data(), mb->size() );
         ret->seekBegin(0);
      }
      else if ( i_data.isString() )
      {
         ret = new Falcon::StringStream( *i_data.asString() );
      }
   }

   // try to load from storage if we didn't create a stream.
   if ( ret == 0 )
   {
      Falcon::Item i_storage;
      if ( ! self->getProperty( "storage", i_storage ) || ! i_storage.isString() )
      {
         // invalid storage?
         throw new Falcon::TypeError( Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
               .extra( ".storage" ) );
      }

      Falcon::FileStream *temp = new Falcon::FileStream();
      if ( ! temp->open( *i_storage.asString(), Falcon::BaseFileStream::e_omReadOnly ) )
      {
         Falcon::int64 le = temp->lastError();
         delete temp;
         throw new Falcon::IoError(
            Falcon::ErrorParam( Falcon::e_io_error, __LINE__ )
            .sysError( (uint32) le ) );
      }
      
      ret = temp;
   }

   // create the stream.
   Falcon::Item *stream_cls = vm->findWKI( "Stream" );
   fassert( stream_cls != 0 );
   fassert( stream_cls->isClass() );
   Falcon::CoreObject *oret = stream_cls->asClass()->createInstance();
   oret->setUserData( ret );
   vm->retval( oret );
}


void InitUploadedClass( Module* self )
{
   // Create a class for uploaded files
   Falcon::Symbol *c_upfile = self->addClass( "Uploaded" );
   c_upfile->setWKS( true );
   // we don't need an object manager as we don't have internal data.
   self->addClassProperty( c_upfile, "size" );
   self->addClassProperty( c_upfile, "mimeType" );
   self->addClassProperty( c_upfile, "filename" );
   self->addClassProperty( c_upfile, "storage" );
   self->addClassProperty( c_upfile, "data" );
   self->addClassProperty( c_upfile, "error" );
   self->addClassMethod( c_upfile, "read", &Uploaded_read );
   self->addClassMethod( c_upfile, "store", &Uploaded_store ).asSymbol()
      ->addParam( "path" );
}

}
}


/* end of uploaded_ext.cpp */