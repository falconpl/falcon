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
#include <falcon/uri.h>

#include "falcon_rtl_ext.h"
#include "rtl_messages.h"

#include <string.h>

/*#
   @beginmodule falcon_rtl
*/

/*#
   @begingroup rtl_syssupport
*/

/*#
   @funset rtl_dir_funcs Directory functions
   @brief Directory and file names functions.

   Directory functions are currently under development. The general principle is
   that they should be, where possible, multiplatform and portable, with some
   extensions working only on specific OSes.  Also, the general idea on failure is
   that they should raise an error when success is expected, and return an error
   value when failure is likely. However, the behavior of the function listed in
   this section is subject to sudden change, as the contribution of the community
   (even if just in form of suggestion) is vital.

   @beginset rtl_dir_funcs
*/


namespace Falcon {

namespace Ext {

FileStatManager::InnerData::InnerData()
{
   // we'll fill them on request.
   m_cache_mtime.setNil();
   m_cache_atime.setNil();
   m_cache_mtime.setNil();
}

FileStatManager::InnerData::InnerData( const InnerData &other ):
   m_fsdata( other.m_fsdata )
{
   // we'll fill them on request.
   m_cache_mtime.setNil();
   m_cache_atime.setNil();
   m_cache_mtime.setNil();
}

FileStatManager::InnerData::~InnerData()
{
}

void *FileStatManager::onInit( VMachine *vm )
{
   return new InnerData;
}

void FileStatManager::onGarbageMark( VMachine *vm, void *data )
{
   InnerData *id = static_cast<InnerData *>(data);
   vm->memPool()->markItemFast( id->m_cache_mtime );
   vm->memPool()->markItemFast( id->m_cache_atime );
   vm->memPool()->markItemFast( id->m_cache_ctime );
}

void FileStatManager::onDestroy( VMachine *vm, void *user_data )
{
   InnerData *id = static_cast<InnerData *>(user_data);
   delete id;
}


void* FileStatManager::onClone( VMachine *vm, void *user_data )
{
   InnerData *id = static_cast<InnerData *>(user_data);
   InnerData *clone = new InnerData( *id );

   // don't clone the timestamps now; let them lazy
   return clone;
}


void FileStats_type_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatManager::InnerData *id = static_cast<FileStatManager::InnerData *>(user_data);
   property = (int64) id->m_fsdata.m_type;
}

void FileStats_ctime_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatManager::InnerData *id = static_cast<FileStatManager::InnerData *>(user_data);

   // is read only
   if ( id->m_cache_ctime.isNil() ) {
      Item *ts_class = instance->origin()->findWKI( "TimeStamp" );
      //if we wrote the std module, can't be zero.
      fassert( ts_class != 0 );
      id->m_cache_ctime = ts_class->asClass()->createInstance( new TimeStamp(*id->m_fsdata.m_ctime) );
   }
   else {
      TimeStamp *ts = static_cast<TimeStamp *>( id->m_cache_ctime.asObject()->getUserData());
      *ts = *id->m_fsdata.m_ctime;
   }

   property = id->m_cache_ctime;
}

void FileStats_mtime_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatManager::InnerData *id = static_cast<FileStatManager::InnerData *>(user_data);

   // is read only
   if ( id->m_cache_mtime.isNil() ) {
      Item *ts_class = instance->origin()->findWKI( "TimeStamp" );
      //if we wrote the std module, can't be zero.
      fassert( ts_class != 0 );
      id->m_cache_mtime = ts_class->asClass()->createInstance( new TimeStamp(*id->m_fsdata.m_mtime) );
   }
   else {
      TimeStamp *ts = static_cast<TimeStamp *>( id->m_cache_mtime.asObject()->getUserData());
      *ts = *id->m_fsdata.m_mtime;
   }

   property = id->m_cache_mtime;
}

void FileStats_atime_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   FileStatManager::InnerData *id = static_cast<FileStatManager::InnerData *>(user_data);
   // is read only
   if ( id->m_cache_atime.isNil() ) {
      Item *ts_class = instance->origin()->findWKI( "TimeStamp" );
      //if we wrote the std module, can't be zero.
      fassert( ts_class != 0 );
      id->m_cache_atime = ts_class->asClass()->createInstance( new TimeStamp(*id->m_fsdata.m_atime) );
   }
   else {
      TimeStamp *ts = static_cast<TimeStamp *>( id->m_cache_atime.asObject()->getUserData());
      *ts = *id->m_fsdata.m_atime;
   }

   property = id->m_cache_atime;
}


/*#
   @function FileReadStats
   @param filename Relative or absolute path to a file for which stats must be read
   @return A new @a FileStat instance or nil.

   On success, retunrs a new instance of the FileStat class; on failure, nil is returned.
*/

FALCON_FUNC FileReadStats( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   // if the name is not given, we expect a readStats to be called later on
   if ( name == 0 )
      return;

   if ( ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // create the timestamps
   Item *fs_class = vm->findWKI( "FileStat" );
   //if we wrote the std module, can't be zero.
   fassert( fs_class != 0 );

   CoreObject *self = fs_class->asClass()->createInstance();

   // CreateInstance will prepare our inner data.
   FileStatManager::InnerData *id = static_cast<FileStatManager::InnerData *>( self->getUserData() );

   if ( ! Sys::fal_stats( *name->asString(), id->m_fsdata ) )
   {
      String fname =  *name->asString();
      vm->raiseModError( new IoError( ErrorParam( 1001, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot read stats for file" ).extra( fname ).
         sysError( (uint32) Sys::_lastError() ) ) );
      return;
   }

   vm->retval( self );
}

/*#
   @class FileStat
   @brief Class holding informations on system files.

   The FileStat class holds informations on a single directory entry. It is
   returned by the @a FileReadStats factory function, but it can be also instantiated
   directly. Then the @a FileStat.readStats method can be used to fill the contents of
   the instance with actual data from the file system.

   @prop access POSIX access mode
   @prop atime Last access time, expressed as a @a TimeStamp instance.
   @prop attribs DOS Attributes
   @prop ctime Creation time or last attribute change time, expressed as a @a TimeStamp instance.
   @prop group Group ID of the given file.
   @prop mtime Last modify time, expressed as a @a TimeStamp instance.
   @prop owner Owner ID of the given file.
   @prop size File size.
   @prop ftype File type; can be one of the following constants (declared in this class):
      - NORMAL
      - DIR
      - PIPE
      - LINK
      - DEVICE
      - SOCKET
      - UNKNOWN

   Both access and attribs properties are given a value respectively only on
   POSIX or MS-Windows systems; their value is the underlying numeric
   value the system provides. The ctime property has a different meaning
   in MS-Windows and POSIX system. In the former, is the time at which the
   file has been created; in the latter is the time when the file ownership flags
   have been last changed, which may or may not be the same as file creation time.

   Times are returned as a @a TimeStamp class instance; the time is always expressed
   as local system time.
*/

/*#
   @method readStats FileStat
   @brief Fills the data in this instance reading them from a system file.
   @param filename Relative or absolute path to a file for which stats must be read
   @return True on success, false if the file cannot be queried.

   Fills the contents of this object with informations on the given file.
   If the stats of the required file can be read, the function returns true.
*/

FALCON_FUNC FileStat_readStats ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   FileStat fstats;
   CoreObject *self = vm->self().asObject();
   FileStatManager::InnerData *id = static_cast<FileStatManager::InnerData *>( self->getUserData() );
   vm->regA().setBoolean( Sys::fal_stats( *name->asString(), id->m_fsdata ) );
}

/*#
   @function fileType
   @brief Deterimnes the type of a file.
   @param filename Relative or absolute path to a file.
   @return A valid file type or FILE_TYPE_NOTFOUND if not found.

   This function is useful to know what of what kind of system entry
   is a certain file, or if it exists at all, without paying the overhead
   for a full FileStat object being instantiated.

   Returned values may be:
      - FILE_TYPE_NORMAL
      - FILE_TYPE_DIR
      - FILE_TYPE_PIPE
      - FILE_TYPE_LINK
      - FILE_TYPE_DEVICE
      - FILE_TYPE_SOCKET
      - FILE_TYPE_UNKNOWN

   or FILE_TYPE_NOTFOUND if the file doesn't exist.
*/
FALCON_FUNC  fileType( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);

   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   FileStat::e_fileType type;

   Sys::fal_fileType( *name->asString(), type );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String temp;
   if ( ! Sys::fal_readlink( *name->asString(), temp ) ) {
      vm->retnil();
   }
   else {
      String *ret = new GarbageString( vm );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   Path path( *name->asString() );
   if ( ! path.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_invalid_path ) ) ) );
      return;
   }

   CoreArray *parts = new CoreArray( vm, 4 );
   parts->length( 4 );
   String *res = new GarbageString( vm );
   String *loc = new GarbageString( vm );
   String *file = new GarbageString( vm );
   String *ext = new GarbageString( vm );

   path.split( *res, *loc, *file, *ext );
   parts->at(0) = res;
   parts->at(1) = loc;
   parts->at(2) = file;
   parts->at(3) = ext;

   vm->retval( parts );
}


/*
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_invalid_path ) ) ) );
      return;
   }

   vm->retval( new GarbageString( vm, p.get() ) );
}

/*
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *name = filename->asString();
   int32 len = name->length()-1;
   int32 pos = len;
   while( pos >= 0 )
   {
      if ( name->getCharAt( pos ) == '/' )
      {
         vm->retval( new GarbageString( vm, *name, pos + 1 ) );
         return;
      }
      pos--;
   }

   // shallow copy
   vm->retval( *filename );
}


/*
   @function filePath
   @brief Return the path specification part in a complete filename.
   @param fullpath A string containing a path.
   @return The path part.

   The function determines the filename part of a complete path name. The
   returned filename includes the host or disk specification, if present. The
   filename does not need to represent a file actually existing in the system.

   @note This function is an interal shortcut to the @a Path class.
*/

FALCON_FUNC  filePath ( ::Falcon::VMachine *vm )
{
   Item *filename = vm->param(0);
   if ( filename == 0 || ! filename->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }


   String *name = filename->asString();
   int32 len = name->length();
   int32 pos = len-1;

   while( pos > 0 ) {
      if ( name->getCharAt( pos ) == '/' ) {
         vm->retval( new GarbageString( vm, *name, 0, pos ) );
         return;
      }
      pos--;
   }

   if ( name->getCharAt( pos ) == '/' )
      vm->retval( new GarbageString( vm, "/" ) );
   else
      vm->retval( new GarbageString( vm ) );
}

/*
   @function DirectoryOpen
   @brief Opens a directory and returns a directory object.
   @param dirname A relative or absolute path to a directory
   @return An instance of the @a Directory class.
   @raise IoError on failure.

   If the function is successful, an instance of the Directory
   class is returned. The instance can be used to retrieve the
   directory entries one at a time.

   On failure, an IoError instance with code 1010 is raised.
*/

FALCON_FUNC  DirectoryOpen ( ::Falcon::VMachine *vm )
{
   Item *name = vm->param(0);
   if ( name == 0 || ! name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 fsError;
   DirEntry *dir = Sys::fal_openDir( *name->asString(), fsError );

   if( dir != 0 ) {
      Item *dir_class = vm->findWKI( "Directory" );
      //if we wrote the std module, can't be zero.
      fassert( dir_class != 0 );
      CoreObject *self = dir_class->asClass()->createInstance();
      self->setUserData( dir );
      vm->retval( self );
   }
   else {
      vm->raiseModError( new IoError( ErrorParam( 1010, __LINE__ ).
         origin( e_orig_runtime ).desc( "Can't open directory" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

/*#
   @class Directory
   @brief Special iterator to access directory listings.

   The Directory class is used by DirectoryOpen() function to return an
   object that the user can iterate upon. It should not be created directly,
   but only through @a DirectoryOpen .

   The caller should repeatedly call the read() method until nil is returned. In
   case an error is raised, the error() method may be called to get informations on
   the cause that raised the error.

   After the read is complete, the caller should call close() to free the resources
   associated with the object. The garbage collector will eventually take care of
   it, but it is better to close the object as soon as possible.
*/

/*#
   @method read Directory
   @brief Returns the next entry in the directory.
   @return A string representing the next entry, or nil when no new entries are left.
*/
FALCON_FUNC  Directory_read ( ::Falcon::VMachine *vm )
{
   DirEntry *dir = static_cast<DirEntry *>(vm->self().asObject()->getUserData());

   String reply;
   if ( dir->read( reply ) ) {
      String *ret = new GarbageString( vm );
      ret->bufferize( reply );
      vm->retval( ret );
   }
   else {
      if ( dir->lastError() != 0 ) {
         vm->raiseModError( new IoError( ErrorParam( 1010, __LINE__ ).
            origin( e_orig_runtime ).desc( "Can't read directory" ).
            sysError( (uint32) Sys::_lastError() ) ) );
      }
      vm->retnil();
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
   DirEntry *dir = static_cast<DirEntry *>(vm->self().asObject()->getUserData());
   dir->close();
   if ( dir->lastError() != 0 ) {
      vm->raiseModError( new IoError( ErrorParam( 1010, __LINE__ ).
         origin( e_orig_runtime ).desc( "Can't close directory" ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

/*#
   @method error Directory
   @brief Returns the last system error code that the directory operation causes.
   @return A system error code.

   The error code may be rendered into a string using the @a systemErrorDescription function.
*/

FALCON_FUNC  Directory_error( ::Falcon::VMachine *vm )
{
   DirEntry *dir = static_cast<DirEntry *>(vm->self().asObject()->getUserData());
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
   lead to the directory that should be created. In example:

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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   const String &strName = *name->asString();
   bool descend = vm->param(1) == 0 ? false : vm->param(1)->isTrue();

   int32 fsError = 0;
   if ( descend )
   {
      // find /.. sequences
      uint32 pos = strName.find( "/" );
      while( true )
      {
         String strPath( strName, 0, pos );

         // stat the file
         FileStat fstats;
         // if the file exists...
         if ( (! Sys::fal_stats( strPath, fstats )) ||
              fstats.m_type != FileStat::t_dir )
         {
            // if it's not a directory, try to create the directory.
            if ( ! Sys::fal_mkdir( strPath, fsError ) )
               break;
         }

         // last loop?
         if ( pos == String::npos )
            break;

         pos = strName.find( "/", pos + 1 );
       }
   }
   else
   {
      // Just one try; succeed or fail
      Sys::fal_mkdir( strName, fsError );
   }

   if ( fsError != 0 )
   {
      vm->raiseModError( new IoError( ErrorParam( 1011, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot create directory" ).extra( strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_rmdir( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1012, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot remove directory" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_chdir( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1013, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change working directory" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
   String *ret = new GarbageString( vm );
   if( ! Sys::fal_getcwd( *ret, fsError ) ) {
      delete ret;
      vm->raiseModError( new IoError( ErrorParam( 1014, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot read current working directory" ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }
   String *strName = name->asString();

   int32 fsError;
   if( ! Sys::fal_unlink( *strName, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1015, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot remove target file" ).extra( *strName ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   String *strName = name->asString();
   String *strDest = dest->asString();

   int32 fsError;
   if( ! Sys::fal_move( *strName, *strDest, fsError ) ) {
      vm->raiseModError( new IoError( ErrorParam( 1016, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot move target file" ).extra( *strName + " -> " + *strDest ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
      origin( e_orig_runtime ) ) );
      return;
   }

   if( ! Sys::fal_chmod( *name->asString(), (uint32) mode->forceInteger() ) )
   {
      vm->raiseModError( new IoError( ErrorParam( 1016, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file mode" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   if( Sys::fal_chown( *name->asString(), (int32) mode->forceInteger() ) )
   {
      vm->raiseModError( new IoError( ErrorParam( 1017, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file owner" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   if( Sys::fal_chgrp( *name->asString(), (int32) mode->forceInteger() ) )
   {
      vm->raiseModError( new IoError( ErrorParam( 1018, __LINE__ ).
         origin( e_orig_runtime ).desc( "Cannot change target file owner" ).extra( *name->asString() ).
         sysError( (uint32) Sys::_lastError() ) ) );
   }
}

//====================================================
// (reflexive) Class path
//

/*# @class Path
   @brief Interface to local filesystem path definition.
   @optparam path The path that will be used as initial path.
   @raise ParamError in case the inital path is malformed.

   This class offers an object oriented interface to access
   path elements given a complete path, or to build a path from its elements.
*/

/*# @property unit Path
   @brief Unit specificator.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the unit specificator (disk name) used in some filesystems.
   It is separated by the rest of the path via a ":". According to
   RFC 3986 it always starts with a "/", which is automatically added
   if absent.
*/

/*# @property location Path
   @brief Location specificator.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the "path to file". It can start with a "/" or not; if
   it starts with a "/" it is considered absolute.
*/

/*# @property file Path
   @brief File part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the part of the path that identifies an element in a directory.
   It includes everything after the last "/" path separator.
*/

/*# @property filename Path
   @brief File name part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This element coresponds to the first part of the file element, if it is
   divided into a filename and an extension by a "." dot.
*/

/*# @property extension Path
   @brief File extension part.
   @raise ParamError if assigned to a value that makes the path invalid.

   This element coresponds to the first last of the file element, if it is
   divided into a filename and an extension by a "." dot.
*/

/*# @property path Path
   @brief Complete path.
   @raise ParamError if assigned to a value that makes the path invalid.

   This is the complete path referred by this object.
*/

void* PathManager::onInit( VMachine *vm )
{
   return 0;
}

void  PathManager::onDestroy( VMachine *vm, void *user_data )
{
   delete static_cast<Path* >( user_data );
}

void* PathManager::onClone( VMachine *vm, void *user_data )
{
   return new Path( *static_cast<Path* >( user_data ) );
}

bool PathManager::onObjectReflectTo( CoreObject *reflector, void *user_data )
{
   Path &path = *static_cast<Path *>( user_data );

   Item *property = reflector->cachedProperty( "unit" );
   if ( ! property->isString() )
      goto complain;

   path.setResource( *property->asString() );

   property = reflector->cachedProperty( "location" );
   if ( ! property->isString() )
      goto complain;

   path.setLocation( *property->asString() );

   property = reflector->cachedProperty( "file" );
   if ( ! property->isString() )
      goto complain;

   path.setFile( *property->asString() );

   property = reflector->cachedProperty( "extension" );
   if ( ! property->isString() )
      goto complain;

   path.setExtension( *property->asString() );

   if ( ! path.isValid() )
   {
      reflector->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( reflector->origin()->moduleString( rtl_invalid_path ) ) ) );
   }

   return true;

complain:
   reflector->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
      origin( e_orig_runtime ).extra( "S" ) ) );
   return true;
}

bool PathManager::onObjectReflectFrom( CoreObject *reflector, void *user_data )
{
   Path &path = *static_cast<Path *>( user_data );

   reflector->cacheStringProperty( "unit", path.getResource() );
   reflector->cacheStringProperty( "location", path.getLocation() );
   reflector->cacheStringProperty( "file", path.getFile() );
   reflector->cacheStringProperty( "extension", path.getExtension() );
   reflector->cacheStringProperty( "filename", path.getFilename() );

   // TODO: reflect URI
   return true;
}

// Reflective URI method
void Path_path_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );

   instance->reflectTo( user_data );

   if ( ! path.isValid() )
   {
      instance->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( instance->origin()->moduleString( rtl_invalid_path ) ) ) );
   }

   // And now set the property
   if ( property.isString() )
      property.asString()->bufferize( path.get() );
   else
      property = new GarbageString( instance->origin(), path.get() );
}

void Path_path_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );

   // We're setting the URI, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), set );
   instance->reflectFrom( user_data );

   if ( ! path.isValid() )
   {
      instance->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( instance->origin()->moduleString( rtl_invalid_path ) ) ) );
   }
}

// Reflective URI method
void Path_filename_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getFilename );
}

void Path_filename_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );

   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setFilename );

   if ( ! path.isValid() )
   {
      instance->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( instance->origin()->moduleString( rtl_invalid_path ) ) ) );
   }
}

// Reflective path method
void Path_file_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getFile );
}

void Path_file_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setFile );

   if ( ! path.isValid() )
   {
      instance->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( instance->origin()->moduleString( rtl_invalid_path ) ) ) );
   }
}

// Reflective path method
void Path_extension_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   FALCON_REFLECT_STRING_FROM( (&path), getExtension );
}

void Path_extension_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   Path &path = *static_cast<Path *>( user_data );
   // We're setting the path, that is, the property has been written.
   FALCON_REFLECT_STRING_TO( (&path), setExtension );

   if ( ! path.isValid() )
   {
      instance->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( instance->origin()->moduleString( rtl_invalid_path ) ) ) );
   }
}

/*#
   @init Path
   @brief Constructor for the Path class.
   @raise ParamError in case the inital path is malformed.

   Builds the path object, optionally using the given parameter
   as a complete path constructor.

   If the parameter is an array, it must have at least four
   string elements, and it will be used to build the path from
   its constituents. In example:

   @code
      unit = "C"
      location = "/a/path/to"
      file = "somefile"
      ext = "anext"
      p = Path( [ unit, location, file, ext ] )
   @endocde

   @b nil can be passed if some part of the specification is not used.

   @note Use the fileNameMerge() function to simply merge elements of a path
   specification into a string.
   @see fileNameMerge
*/

FALCON_FUNC  Path_init ( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param(0);
   // we need anyhow a carrier.
   CoreObject *self = vm->self().asObject();
   Path *path = new Path;
   self->setUserData( path );

   if ( p0 == 0 || ( ! p0->isString() && ! p0->isArray() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S|A" ) ) );
      return;
   }

   if ( p0->isString() )
   {
      path->set( *p0->asString() );
   }
   else {
      const String *unitspec = 0;
      const String *fname = 0;
      const String *fpath = 0;
      const String *fext = 0;

      String sDummy;

      const CoreArray &array = *p0->asArray();
      if( array.length() >= 0 && array[0].isString() )
         unitspec = array[0].asString();
      else
         unitspec = &sDummy;

      if( array.length() >= 1 && array[1].isString() )
         fpath = array[1].asString();
      else
         fpath = &sDummy;

      if( array.length() >= 2 && array[2].isString() )
         fname = array[2].asString();
      else
         fname = &sDummy;

      if( array.length() >= 3 && array[3].isString() )
         fext = array[3].asString();
      else
         fext = &sDummy;

      path->join( *unitspec, *fpath, *fname, *fext );
   }

   self->reflectFrom( path );

   if ( ! path->isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_invalid_path ) ) ) );
   }
}

}
}

/* end of dir.cpp */
