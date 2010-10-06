/*
   FALCON - The Falcon Programming Language.
   FILE: dir.cpp

   Directory management api.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Directory management api
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/sys.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/filestat.h>
#include <falcon/dir_sys.h>
#include <falcon/memory.h>
#include <falcon/fassert.h>

#include "core_module.h"
#include <falcon/eng_messages.h>

#include <string.h>

/*#
   @begingroup core_syssupport
*/

/*#
   @funset core_dir_funcs Directory functions
   @brief Directory and file names functions.

   Directory functions are currently under development. The general principle is
   that they should be, where possible, multiplatform and portable, with some
   extensions working only on specific OSes.  Also, the general idea on failure is
   that they should raise an error when success is expected, and return an error
   value when failure is likely. However, the behavior of the function listed in
   this section is subject to sudden change, as the contribution of the community
   (even if just in form of suggestion) is vital.

   @beginset core_dir_funcs
*/


namespace Falcon {

namespace core {

FileStatObject::FileStatObject( const CoreClass *cls ):
   ReflectObject( cls, new InnerData )
{
}

FileStatObject::FileStatObject( const FileStatObject &other ):
   ReflectObject( other.generator(), new InnerData( *other.getInnerData() ) )
{
}

FileStatObject::~FileStatObject()
{
   delete getInnerData();
}

void FileStatObject::gcMark( uint32 mark )
{
   memPool->markItem( getInnerData()->m_cache_mtime );
   memPool->markItem( getInnerData()->m_cache_atime );
   memPool->markItem( getInnerData()->m_cache_ctime );
}

FileStatObject* FileStatObject::clone() const
{
   return new FileStatObject( *this );
}

CoreObject* FileStatObjectFactory( const CoreClass *cls, void *, bool  )
{
   return new FileStatObject( cls );
}


FileStatObject::InnerData::InnerData( const InnerData &other ):
   m_fsdata( other.m_fsdata )
{
   fassert( m_cache_atime.asObjectSafe() );

   other.m_cache_mtime.clone( m_cache_mtime );
   other.m_cache_atime.clone( m_cache_atime );
   other.m_cache_ctime.clone( m_cache_ctime );
}

void FileStats_type_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatObject::InnerData *id = static_cast<FileStatObject::InnerData *>(user_data);
   property = (int64) id->m_fsdata.m_type;
}

void FileStats_ctime_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatObject::InnerData *id = static_cast<FileStatObject::InnerData *>(user_data);

   // is read only
   if ( id->m_fsdata.m_ctime != 0 )
   {
      if ( id->m_cache_ctime.isNil() )
      {
         VMachine* vm = VMachine::getCurrent();
         fassert( vm != 0 );
         Item *ts_class = vm->findWKI( "TimeStamp" );
         //if we wrote the std module, can't be zero.
         fassert( ts_class != 0 );
         id->m_cache_ctime = ts_class->asClass()->createInstance( new TimeStamp( *id->m_fsdata.m_ctime) );
      }
      else {
         TimeStamp *ts = dyncast<TimeStamp *>( id->m_cache_ctime.asObject()->getFalconData());
         *ts = *id->m_fsdata.m_ctime;
      }
   }
   else {
      id->m_cache_ctime.setNil();
   }

   property = id->m_cache_ctime;
}

void FileStats_mtime_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatObject::InnerData *id = static_cast< FileStatObject::InnerData *>(user_data);

   // is read only
   if ( id->m_fsdata.m_mtime != 0 )
   {
      if ( id->m_cache_mtime.isNil() ) {
         VMachine* vm = VMachine::getCurrent();
         fassert( vm != 0 );
         Item *ts_class = vm->findWKI( "TimeStamp" );
         //if we wrote the std module, can't be zero.
         fassert( ts_class != 0 );
         id->m_cache_mtime = ts_class->asClass()->createInstance( new TimeStamp( *id->m_fsdata.m_mtime) );
      }
      else {
         TimeStamp *ts = dyncast<TimeStamp *>( id->m_cache_mtime.asObject()->getFalconData());
         *ts = *id->m_fsdata.m_mtime;
      }
   }
   else {
      id->m_cache_mtime.setNil();
   }

   property = id->m_cache_mtime;
}

void FileStats_atime_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatObject::InnerData *id = static_cast<FileStatObject::InnerData *>(user_data);
   // is read only
   if ( id->m_fsdata.m_atime != 0 )
   {
      if ( id->m_cache_atime.isNil() )
      {
         VMachine* vm = VMachine::getCurrent();
         fassert( vm != 0 );
         Item *ts_class = vm->findWKI( "TimeStamp" );
         //if we wrote the std module, can't be zero.
         fassert( ts_class != 0 );
         id->m_cache_atime = ts_class->asClass()->createInstance( new TimeStamp( *id->m_fsdata.m_atime) );
      }
      else
      {
         TimeStamp *ts = dyncast<TimeStamp *>( id->m_cache_atime.asObject()->getFalconData());
         *ts = *id->m_fsdata.m_atime;
      }
   }
   else {
      id->m_cache_atime.setNil();
   }

   property = id->m_cache_atime;
}


/*#
   @class FileStat
   @optparam path If given, the filestats will be initialized with stats of the given file.
   @raise IoError if @b path is given but not found.
   @brief Class holding informations on system files.

   The FileStat class holds informations on a single directory entry.

   It is possible to pass a @b path parameter, in which case, if the given file is found,
   the contents of this class is filled with the stat data from the required file, otherwise
   an IoError is raised. The @a FileStat.read method would search for the required file
   without raising in case it is not found, so if it preferable not to raise on failure
   (i.e. because searching the most fitting of a list of possibly existing files), it is
   possiblo to create the FileStat object without parameters and the use the @b read method
   iteratively.

   @prop access POSIX access mode
   @prop atime Last access time, expressed as a @a TimeStamp instance.
   @prop attribs DOS Attributes
   @prop ctime Creation time or last attribute change time, expressed as a @a TimeStamp instance.
   @prop group Group ID of the given file.
   @prop mtime Last modify time, expressed as a @a TimeStamp instance.
   @prop owner Owner ID of the given file.
   @prop size File size.

   Both access and attribs properties are given a value respectively only on
   POSIX or MS-Windows systems; their value is the underlying numeric
   value the system provides. The ctime property has a different meaning
   in MS-Windows and POSIX system. In the former, is the time at which the
   file has been created; in the latter is the time when the file ownership flags
   have been last changed, which may or may not be the same as file creation time.

   Times are returned as a @a TimeStamp class instance; the time is always expressed
   as local system time.
*/

/*# @property ftype FileStat
   @brief Type of the file.

   Can be one of the following constants (declared in this class):

   - NORMAL
   - DIR
   - PIPE
   - LINK
   - DEVICE
   - SOCKET
   - UNKNOWN
*/

FALCON_FUNC FileStat_init( ::Falcon::VMachine *vm )
{
   // we're initialized with consistent data from the class factory function.
   if ( vm->paramCount() > 0 )
   {
      // would throw on param error...
      FileStat_read( vm );

      // we must raise from the constructor, if we didn't found the file.
      if( ! vm->regA().isTrue() )
      {
         throw new IoError( ErrorParam( e_nofile, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *vm->param(0)->asString() ) );
      }

   }
}

/*#
   @method read FileStat
   @brief Fills the data in this instance reading them from a system file.
   @param filename Relative or absolute path to a file for which stats must be read
   @return True on success, false if the file cannot be queried.

   Fills the contents of this object with informations on the given file.
   If the stats of the required file can be read, the function returns true.
*/

FALCON_FUNC FileStat_read ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra("S") );
   }

   FileStatObject *self = dyncast<FileStatObject*>(vm->self().asObject());
   FileStatObject::InnerData *id = self->getInnerData();
   vm->regA().setBoolean( Sys::fal_stats( *name->asString(), id->m_fsdata ) );
}

/*#
   @function fileType
   @brief Determines the type of a file.
   @param filename Relative or absolute path to a file.
   @optparam nf Don't follow symlinks

   @return A valid file type or FileStat.NOTFOUND if not found.

   This function is useful to know what of what kind of system entry
   is a certain file, or if it exists at all, without paying the overhead
   for a full FileStat object being instantiated.


   Returned values may be:
      - FileStat.NORMAL
      - FileStat.DIR
      - FileStat.PIPE
      - FileStat.LINK - Only if @b nf parameter is true.
      - FileStat.DEVICE
      - FileStat.SOCKET
      - FileStat.UNKNOWN

   or FileStat.NOTFOUND if the file doesn't exist.
*/
FALCON_FUNC  fileType( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *i_df = vm->param(1);
   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
      return;
   }

   FileStat::e_fileType type = FileStat::t_notFound;

   bool follow = !( i_df != 0 && i_df->isTrue() );
	int count = 99;

	String sName = *name->asString();
   while( true )
   {
   	Sys::fal_fileType( sName, type );

   	if ( follow && type == FileStat::t_link )
		{
		   String temp;
		   if ( ! Sys::fal_readlink( sName, temp ) || --count == 0 )
		   {
				type = FileStat::t_notFound;
				break;
			}
		   sName = temp;
		}
   	else
   	{
   		break;
   	}
   }

   // will already be -1 if not found
   vm->retval( type );
}

/*#
   @function dirReadLink
   @brief On systems supporting symbolic links, returns the linked file.
   @param linkPath A path to a symbolic link.

   If the target file in the linkPath parameter is a symbolic link and can
   be read, the return value will be a string containing the file the link
   is pointing to. Otherwise, the function will return nil. Function will
   return nil also on system where symbolic links are not supported.
*/
FALCON_FUNC  dirReadLink( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime ) );
      return;
   }

   String temp;
   if ( ! Sys::fal_readlink( *name->asString(), temp ) ) {
      vm->retnil();
   }
   else {
      CoreString *ret = new CoreString;
      ret->bufferize( temp );
      vm->retval( ret );
   }
}

/*#
   @function dirMakeLink
   @brief Creates a soft link to a file.
   @param source The original file path.
   @param dest The path to the link file.

   The path of both source and dest parameter is to be expressed in Falcon
   convention (forward slashes), and can be both relative to the working directory
   or absolute.

   Currently, the function works only on UNIX systems; Windows platforms have
   recently added this functionality, but they are still not supported by this
   function at the moment.

   On success, the function returns true. On failure, it returns false. In case the
   function is not supported, it just returns false. (Comments are welcome).
*/

FALCON_FUNC  dirMakeLink( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *dest = vm->param(1);

   if ( name == 0 || ! name->isString() || dest == 0 || ! dest->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime ) );
   }

   if ( ! Sys::fal_readlink( *name->asString(), *dest->asString() ) ) {
      vm->regA().setBoolean( false );
   }
   else {
      vm->regA().setBoolean( true );
   }
}

/*
   @function fileNameSplit
   @brief Splits a filename in four elements.
   @param path A string containing a path.
   @return An array of four elements containing the splitted string.

   This function analyzes the given filename and separates it in disk/server
   specification, path, file name and extension, returning them in a 4 string
   element array. If one of the elements is not present in the filename, the
   corresponding location is set to nil.

   The extension dot, the disk/server specification colon and the last slash of the
   path are removed from the returned strings.

   @note This function is an interal shortcut to the @a Path class.
*/

FALCON_FUNC  fileNameSplit ( ::Falcon::VMachine *vm )
{
   // a filename is always in this format:
   // [DISK_SPEC or SERVER:][/path/to/file/]file.part[.ext]

   // returns an array of 4 elements. ALWAYS.
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   Path path( *name->asString() );
   if ( ! path.isValid() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_invalid_path ) ) );
   }

   CoreArray *parts = new CoreArray( 4 );
   parts->length( 4 );
   CoreString *res = new CoreString;
   CoreString *loc = new CoreString;
   CoreString *file = new CoreString;
   CoreString *ext = new CoreString;

   path.split( *res, *loc, *file, *ext );
   parts->at(0) = res;
   parts->at(1) = loc;
   parts->at(2) = file;
   parts->at(3) = ext;

   vm->retval( parts );
}


/*#
   @function fileNameMerge
   @brief Merges a filename split up in four elements.
   @param spec The disk or server specification, an array containing all the
      elements of the file path, or nil.
   @param path Path to the file, or nil.
   @param filename Filename, or nil.
   @param ext extension, or nil.
   @return A complete absolute path.

   The final path is composed by adding a colon after the disk/server
   specification, a slash after the path and a dot before the extension.

   It is also possible to pass all the four elements in an array, in place
   of the @b spec parameter.

   @note This function is an interal shortcut to the @a Path class.
*/
FALCON_FUNC  fileNameMerge ( ::Falcon::VMachine *vm )
{
   const String *unitspec = 0;
   const String *fname = 0;
   const String *fpath = 0;
   const String *fext = 0;

   String sDummy;
   Item *p0 = vm->param(0);

   if ( p0 != 0 && p0->isArray() )
   {
      const CoreArray &array = *p0->asArray();
      if( array.length() >= 0 && array[0].isString() ) {
         unitspec = array[0].asString();
      }
      else
         unitspec = &sDummy;

      if( array.length() >= 1 && array[1].isString() ) {
         fpath = array[1].asString();
      }
      else
         fpath = &sDummy;

      if( array.length() >= 2 && array[2].isString() ) {
         fname = array[2].asString();
      }
      else
         fname = &sDummy;

      if( array.length() >= 3 && array[3].isString() ) {
         fext = array[3].asString();
      }
      else
         fext = &sDummy;

   }
   else {
      Item *p1 = vm->param(1);
      Item *p2 = vm->param(2);
      Item *p3 = vm->param(3);

      unitspec = p0 != 0 && p0->isString() ? p0->asString() : &sDummy;
      fpath = p1 != 0 && p1->isString() ? p1->asString() : &sDummy;
      fname = p2 != 0 && p2->isString() ? p2->asString() : &sDummy;
      fext = p3 != 0 && p3->isString() ? p3->asString() : &sDummy;
   }

   Path p;
   p.join( *unitspec, *fpath, *fname, *fext );
   if ( ! p.isValid() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_invalid_path ) ) );
   }

   vm->retval( new CoreString( p.get() ) );
}

/*#
   @function fileName
   @brief Determines the name of a file in a complete path.
   @param path A string containing a path.
   @return The filename part in the path.

   The function determines the filename part of a complete path name. The returned
   filename includes the extension, if present. The filename does not need to
   represent a file actually existing in the system.
   If the filename part cannot be determined, an empty string is returned.
*/

FALCON_FUNC  fileName ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   String *name = filename->asString();
   int32 len = name->length()-1;
   int32 pos = len;
   while( pos >= 0 )
   {
      if ( name->getCharAt( pos ) == '/' )
      {
         vm->retval( new CoreString( *name, pos + 1 ) );
         return;
      }
      pos--;
   }

   // shallow copy
   vm->retval( *filename );
}


/*#
   @function filePath
   @brief Return the path specification part in a complete filename.
   @param fullpath A string containing a path.
   @return The path part.

   The function determines the filename part of a complete path name. The
   returned filename includes the host or disk specification, if present. The
   filename does not need to represent a file actually existing in the system.

   @note The filename should be in Falcon format (URI convention with forward slashes).
   @see Path
*/

FALCON_FUNC  filePath ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }


   String *name = filename->asString();
   int32 len = name->length();
   int32 pos = len-1;

   while( pos > 0 ) {
      if ( name->getCharAt( pos ) == '/' ) {
         vm->retval( new CoreString( *name, 0, pos ) );
         return;
      }
      pos--;
   }

   if ( name->getCharAt( pos ) == '/' )
      vm->retval( new CoreString( "/" ) );
   else
      vm->retval( new CoreString );
}

/*#
   @function fileExt
   @brief Return the extension in a complete filename.
   @param fullpath A string containing a path.
   @return The extension part.

   The function determines the element past a dot after the
   filename part. Filenames can start with a "."; they are not considered
   extensions. 

   @note The filename should be in Falcon format (URI convention with forward slashes).
   @see Path
*/

FALCON_FUNC  fileExt ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   String *name = filename->asString();
   uint32 pos = name->rfind( "/" );
   uint32 len = name->length();
   if ( pos + 2 < len )
   {
      // there may be an extension.
      ++pos;
      // discard initial "."
      while( pos < len && name->getCharAt( pos ) != '.' )
      {
         ++pos;
      }

      // find a dot here.
      uint32 pdot = name->rfind( ".", pos );
      if ( pdot + 1 < len )
      {
         vm->retval( new CoreString( *name, pdot+1 ) );
         return;
      }
   }
   
   // else return ""
   vm->retval( new CoreString );
}

/*#
   @function fileUnit
   @brief Return the unit specificator in a complete filename.
   @param fullpath A string containing a path.
   @return The unit specificator part.

   Returns the unit specification in a complete path.

   @see Path
*/

FALCON_FUNC  fileUnit ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   String *name = filename->asString();
   uint32 pos = name->find( ":" );
   
   if ( pos != String::npos && pos > 0 )
   {
      uint32 start = 0;
      while( start < pos && name->getCharAt(start) == '/' )
      {
         ++start;
      }
      
      vm->retval( new CoreString( *name, start, pos ) );
      return;
   }
   
   // else return ""
   vm->retval( new CoreString );
}


/*#
   @class Directory
   @brief Special iterator to access directory listings.
   @param dirname A relative or absolute path to a directory
   @raise IoError on failure.

   The Directory class allow to iterate through the contents of a local file directory.

   The caller should repeatedly call the read() method until nil is returned. In
   case an error is raised, the error() method may be called to get informations on
   the cause that raised the error.

   After the read is complete, the caller should call close() to free the resources
   associated with the object. The garbage collector will eventually take care of
   it, but it is better to close the object as soon as possible.
*/

FALCON_FUNC  Directory_init ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
      return;
   }

   int32 fsError;
   DirEntry *dir = Sys::fal_openDir( *name->asString(), fsError );

   if( dir != 0 )
   {
      CoreObject *self = vm->self().asObjectSafe();
      self->setUserData( dir );
   }
   else {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
         .origin( e_orig_runtime )
         .extra( *name->asString() )
         .sysError( (uint32) Sys::_lastError() ) );
   }
}


/*#
   @method read Directory
   @brief Returns the next entry in the directory.
   @return A string representing the next entry, or oob(0) when no new entries are left.

   The usage is
   @code
   dir =  Directory( "." )
   while entry = dir.read(): > entry
   dir.close()
   @endcode
   or
   @code
   dir = Directory( "." )
   for entry in dir.read: > entry
   dir.close()
   @endcode

*/
FALCON_FUNC  Directory_read ( ::Falcon::VMachine *vm )
{
   DirEntry *dir = dyncast<DirEntry *>(vm->self().asObject()->getFalconData());

   String reply;
   if ( dir->read( reply ) )
   {
      CoreString *ret = new CoreString;
      ret->bufferize( reply );
      vm->retval( ret );
   }
   else
   {
      if ( dir->lastError() != 0 )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
            .origin( e_orig_runtime )
            .sysError( (uint32) Sys::_lastError() ) );
      }
      vm->retval( (int64)0 );
      vm->regA().setOob( true );
   }
}

/*#
   @method close Directory
   @brief Closes the directory object.

   This method should be called when the item is not needed anymore to free
   system resources.

   However, the directory listing is closed at garbage collecting.
*/
FALCON_FUNC  Directory_close ( ::Falcon::VMachine *vm )
{
   DirEntry *dir = dyncast<DirEntry *>(vm->self().asObject()->getFalconData());
   dir->close();
   if ( dir->lastError() != 0 ) {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
            .origin( e_orig_runtime )
            .sysError( (uint32) Sys::_lastError() ) );
   }
}


static bool Directory_descend_next_descend ( ::Falcon::VMachine *vm )
{
   if( vm->regA().isOob() && vm->regA().isInteger() )
   {
      // in case of 0 or 1 we should break.
      if ( vm->regA().asInteger() == 0 || vm->regA().asInteger() == 1 )
         return false;
   }

   // we're in a directory. descend.
   int32 fsError;
   DirEntry *dir = Sys::fal_openDir( *vm->param(0)->asString(), fsError );
   vm->regB().setNil();

   if( dir != 0 )
   {
      Item* i_dir = vm->findWKI( "Directory" );
      fassert( i_dir != 0 && i_dir->isClass() );

      // we don't want to be called anymore.
      // when this frame returns, resume previous frame.
      vm->returnHandler( 0 );

      CoreClass* dircls = i_dir->asClass();
      CoreObject *self = dircls->createInstance( dir );
      vm->param(1)->asArray()->append( self );
      // and be sure that the VM will execute this last time
      return false;
   }

   throw new IoError( ErrorParam( e_io_error, __LINE__ )
         .origin( e_orig_runtime )
         .extra( *vm->param(0)->asString() )
         .sysError( (uint32) Sys::_lastError() ) );
}

static bool Directory_descend_next ( ::Falcon::VMachine *vm )
{
   // get the real self
   CoreArray* pushed = vm->local(0)->asArray();
   fassert( pushed->length() > 0 );
   Item& self = pushed->at( pushed->length() -1 );
   DirEntry *dir = dyncast<DirEntry *>(self.asObjectSafe()->getFalconData());

   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      // we're out of here.
      return false;
   }

   String fnext;

   // skip this and parent dir
   while( fnext == "" || fnext == "." || fnext == ".." )
   {
      if( ! dir->read( fnext ) )
      {
         // pop this level.
         dir->close();
         pushed->length( pushed->length()-1);
         // please, repeat us if this was not the last level
         return pushed->length() != 0;
      }
   }

   // is this a directory?
   if ( dir->path().size() != 0 )
      fnext.prepend( dir->path() + "/" );

   FileStat fs;
   Sys::fal_stats( fnext, fs );
   if( fs.m_type == FileStat::t_dir )
   {
      // yes? -- prepare the new callback frame
      if ( vm->param(0) != 0 && ! vm->param(0)->isNil() )
      {
         vm->pushParameter( (new CoreString( fnext ))->bufferize() );
         vm->pushParameter( pushed );
         vm->callFrame( *vm->param(0), 2 );

         // prepare the descent
         vm->returnHandler( &Directory_descend_next_descend );
      }
      else
      {
         // we're in a directory. descend.
         int32 fsError;
         DirEntry *dir = Sys::fal_openDir( fnext, fsError );
         vm->regB().setNil();

         if( dir != 0 )
         {
            Item* i_dir = vm->findWKI( "Directory" );
            fassert( i_dir != 0 && i_dir->isClass() );
            CoreClass* dircls = i_dir->asClass();
            CoreObject *self = dircls->createInstance( dir );
            pushed->append( self );

            return true;
         }

         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                  .origin( e_orig_runtime )
                  .extra( *vm->param(0)->asString() )
                  .sysError( (uint32) Sys::_lastError() ) );

      }
   }
   else {
      // should we call the file handler?
      Item* i_ffunc = vm->param(1);
      if ( i_ffunc != 0 && ! i_ffunc->isNil() )
      {
         vm->pushParameter( (new CoreString( fnext ))->bufferize() );
         vm->callFrame( *i_ffunc, 1 );
         // no need for extra params
      }
   }

   return true;
}

/*#
   @method descend Directory
   @brief Descends into subdirectories, iteratively calling a function.
   @optparam dfunc Function to be called upon directories.
   @optparam ffunc Function to be called upon files.

   This function calls iteratively a function on directory entries.
   If an entry is detected to be a directory, it is passed to
   @b dfunc as the only parameter. If @b ffunc is also provided,
   then it will receive all the non-directory entries. Entries
   coresponding to the current directory and the parent directory
   will never be sent to the handler functions.

   @note The parameters for @b dfunc and @b ffunc will always
   be relative to the directory on which this object has been
   created.

   Retunring an out of band 0, any of the callbacks involved may
   stop the processing and return immediately. An out of band 1
   will skip the currently processed item and proceed.
   The @b dfunc handler is called before descending into the found
   subdirectory; this gives the handlers the chance to skip directories
   or interrupt the search.

   @note After a complete descend, this directory will be closed and won't
   be usable anymore.
*/
FALCON_FUNC  Directory_descend ( ::Falcon::VMachine *vm )
{
   Item *i_dfunc = vm->param(0);
   Item *i_ffunc = vm->param(1);

   if ( ( i_dfunc != 0 && ! i_dfunc->isCallable() && ! i_dfunc->isNil() )
        || ( i_ffunc != 0 && ! i_ffunc->isCallable() && ! i_dfunc->isNil() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "[C],[C]" ) );
      return;
   }

   // nothing to do?
   if( (i_dfunc == 0 || i_dfunc->isNil() ) &&
       (i_ffunc == 0 || i_ffunc->isNil() ))
   {
      // should we signal an error? --- I don't think
      return;
   }

   vm->addLocals(1);
   *vm->local(0) = new CoreArray(1);
   vm->local(0)->asArray()->append( vm->self() );

   // be sure we won't loop out
   vm->regA().setNil();
   vm->returnHandler( &Directory_descend_next );
}


/*#
   @method error Directory
   @brief Returns the last system error code that the directory operation causes.
   @return A system error code.

   The error code may be rendered into a string using the @a systemErrorDescription function.
*/

FALCON_FUNC  Directory_error( ::Falcon::VMachine *vm )
{
   DirEntry *dir =  dyncast<DirEntry *>(vm->self().asObject()->getFalconData());
   vm->retval( (int)dir->lastError() );
}

/*#
   @function dirMake
   @brief Creates a directory.
   @param dirname The name of the directory to be created.
   @optparam bFull Create also the full pat to the given directory.
   @raise IoError on system error.

   On success, this function creates the given directory with normal
   attributes.

   It is possible to specify both a relative or absolute path; both
   the relative and absolute path can contain a subtree specification,
   that is, a set of directories separated by forward slashes, which
   lead to the directory that should be created. For example:

   @code
      dirMake( "top/middle/bottom" )
   @endcode

   instructs @b dirMake to create the directory bottom in a subdirectory
   "middle", which should already exist. Passing @b true as second parameter,
   dirMake will also try to create directories leading to the final destination,
   if they are missing.
*/

FALCON_FUNC  dirMake ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   const String &strName = *name->asString();
   bool descend = vm->param(1) == 0 ? false : vm->param(1)->isTrue();
   int32 fsError = 0;

   if ( ! Sys::fal_mkdir( strName, fsError, descend ) )
   {
      throw  new IoError( ErrorParam( 1011, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot create directory" ).extra( strName ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}

/*#
   @function fileCopy
   @ingroup core_syssupport
   @param source Source file to be copied
   @param dest Destination file.
   @brief Copies a whole file from one position to another.
   @raise IoError on system error.

   This function performs a file copy. The function is still
   experimental and needs addition of VM interruption protocol
   compliancy, as well as the possibility to preserve or change
   the system attributes in the target copy.
*/
FALCON_FUNC  fileCopy ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   Item *filedest = vm->param(1);

   if ( filename == 0 || ! filename->isString() ||
        filedest == 0 || ! filedest->isString()
      )
   {
      throw new ParamError(
         ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra("S,S") );
   }

   const String &source = *filename->asString();
   const String &dest = *filedest->asString();

   ::Falcon::BaseFileStream::t_shareMode shMode = ::Falcon::BaseFileStream::e_smShareFull;

   FileStream instream, outstream;
   instream.open( source, ::Falcon::BaseFileStream::e_omReadOnly, shMode );
   if ( ! instream.good() )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ ).
         extra( source ).
         sysError( (uint32) instream.lastError() ) );
   }

   outstream.create( dest, (Falcon::BaseFileStream::t_attributes) 0644, shMode );
   if ( ! outstream.good() )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ ).
         extra( dest ).
         sysError( (uint32) outstream.lastError() ) );
   }

   // Declaring the VM idle from now on.
   VMachine::Pauser pauser( vm );

   byte buffer[4096];
   int count = 0;
   while( ( count = instream.read( buffer, 4096) ) > 0 )
   {
      if ( outstream.write( buffer, count ) < 0 )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ ).
            sysError( (uint32) outstream.lastError() ) );
      }
   }

   if ( count < 0 )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ ).
            sysError( (uint32) instream.lastError() ) );
   }
}

/*#
   @function dirRemove
   @brief Removes an empty directory.
   @param dir The path to the directory to be removed.
   @raise IoError on system error.

   The function removes an empty directory.
   On failure an IoError with code 1012 will be raised.
*/
FALCON_FUNC  dirRemove ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_rmdir( *strName, fsError ) ) {
      throw new IoError( ErrorParam( 1012, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot remove directory" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}

/*#
   @function dirChange
   @brief Changes the current working directory.
   @param newDir The new working directory.
   @raise IoError on system error.

   On success, the working directory is changed to the specified one. The path
   must be indicated in Falcon conventions (forward slashes separating
   directories). On failure, an IoError is raised.
*/

FALCON_FUNC  dirChange ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      throw  new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_chdir( *strName, fsError ) ) {
      throw  new IoError( ErrorParam( 1013, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change working directory" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) );
   }
   else
      vm->retnil();
}

/*#
   @function dirCurrent
   @brief Returns the current working directory.
   @return A string representing the current working directory.
   @raise IoError on system error.

   On success, a string containing the current working directory
   is returned. The path is in Falcon convention (forward slashes).
   If any system error occurs, an IoError is raised.
*/

FALCON_FUNC  dirCurrent ( ::Falcon::VMachine *vm )
{
   int32 fsError;
   CoreString *ret = new CoreString;
   if( ! Sys::fal_getcwd( *ret, fsError ) ) {
      throw  new IoError( ErrorParam( 1014, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot read current working directory" ).
         sysError( (uint32) Sys::_lastError() ) );
   }
   else
      vm->retval( ret );
}


/*#
   @function fileRemove
   @brief Removes a file from the system.
   @param filename The name, relative or absolute path of the file to be removed.
   @raise IoError on system error.

   This function tries to remove a file from the filesystem. If unsuccessful, an
   error with code 1015 is raised.
*/

FALCON_FUNC  fileRemove ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_unlink( *strName, fsError ) ) {
      throw new IoError( ErrorParam( 1015, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot remove target file" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}

/*#
   @function fileMove
   @brief Renames a file locally.
   @param sourcePath The path of the file to be moved.
   @param destPath The path of the destination file.
   @raise IoError on system error.

   This function actually renames a file. Usually, it will file if the
   caller tries to move the file across filesystems (i.e. on different discs).

   On failure, an IoError is raised.
*/

FALCON_FUNC  fileMove ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *dest = vm->param(1);

   if ( name == 0 || ! name->isString() || dest == 0 || ! dest->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) );
      return;
   }

   String *strName = name->asString();
   String *strDest = dest->asString();

   int32 fsError;
   if( ! Sys::fal_move( *strName, *strDest, fsError ) ) {
      throw new IoError( ErrorParam( 1016, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot move target file" ).extra( *strName + " -> " + *strDest ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}


/*#
   @function fileChmod
   @brief Changes UNIX access right to a directory entry.
   @param path A file or otherwise valid directory entry.
   @param mode The new access mode.
   @raise IoError on system error.

   This function will work only on POSIX systems. The file access mode will be
   set along the octal access right defined by POSIX. See man 3 chmod for details.
*/

FALCON_FUNC  fileChmod ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *mode = vm->param(1);

   if ( name == 0 || ! name->isString() || mode == 0 || ! mode->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
      origin( e_orig_runtime ) );
   }

   if( ! Sys::fal_chmod( *name->asString(), (uint32) mode->forceInteger() ) )
   {
      throw new IoError( ErrorParam( 1016, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file mode" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}


/*#
   @function fileChown
   @brief Changes UNIX owner to a directory entry.
   @param path A file or otherwise valid directory entry.
   @param ownerId The new ownerId.
   @raise IoError on system error.

   This function changes the user ownership of the given file on POSIX systems.
*/

FALCON_FUNC  fileChown ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *mode = vm->param(1);

   if ( name == 0 || ! name->isString() || mode == 0 || ! mode->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) );
      return;
   }

   if( Sys::fal_chown( *name->asString(), (int32) mode->forceInteger() ) )
   {
      throw new IoError( ErrorParam( 1017, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file owner" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}

/*#
   @function fileChgroup
   @brief Changes UNIX group to a directory entry.
   @param path A file or otherwise valid directory entry.
   @param groupId The new group id.
   @raise IoError on system error.

   This function changes the group ownership of the given file on POSIX systems.
*/

FALCON_FUNC  fileChgroup ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   Item *mode = vm->param(1);

   if ( name == 0 || ! name->isString() || mode == 0 || ! mode->isOrdinal() )
   {
      throw  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) );
   }

   if( Sys::fal_chgrp( *name->asString(), (int32) mode->forceInteger() ) )
   {
      throw  new IoError( ErrorParam( 1018, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file owner" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) );
   }
}

}
}

/* end of dir_ext.cpp */
